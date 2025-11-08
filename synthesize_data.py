# synth_benchmark_dataset.py
# Run with: spark-submit --driver-memory 8g --executor-memory 8g synth_benchmark_dataset.py

import re, unicodedata, random, json
from typing import List, Tuple
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, monotonically_increasing_id, size
from pyspark.sql.types import *
from pyspark.ml.feature import RegexTokenizer, IDF, HashingTF
from pyspark.ml import Pipeline
from glob import glob

# ------------------- CONFIG -------------------
SEED = 42
random.seed(SEED)

INPUT_PATH = "/path/to/base_corpus.parquet"  # columns: doc_id(long), text(string)
OUTPUT_DOCS = "/path/to/out/documents.parquet"
OUTPUT_PAIRS = "/path/to/out/pairs.parquet"

# Produce exact dupes and near-dupes in several Jaccard similarity bands on k-shingles: e.g.
# D0: exact (J≈1.0)
# D1: very-near (0.9–1.0)
# D2: near (0.8–0.9)
# D3: similar (0.65–0.8)
# D4: borderline (0.5–0.65)

# get total number of samples in the source dataset
json_files = glob(INPUT_PATH + "/*.json")
total_docs = 0
for jf in json_files:
    with open(jf, 'r') as f:
        data = f.leadlines()
        total_docs += len(data)

target_dups_frac = 0.6  # fraction of total docs to use as dup bases
total_target_dups = int(total_docs * target_dups_frac)
# allocate per-band fractions, [D0, D1, D2, D3, D4] summing to 1.0
frac_per_band = [0.10, 0.25, 0.25, 0.20, 0.20]

# target counts (tune to your scale)
TARGET = {
    f"D{i}": int(total_target_dups * frac) for i, frac in enumerate(frac_per_band)
}

K_SHINGLE = 3

# ------------------- NORMALIZE/TOKENIZE -------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    return s

word_re = re.compile(r"[A-Za-z0-9]+", re.U)

def tokenize(s: str) -> List[str]:
    s = normalize_text(s).lower()
    return word_re.findall(s)

def shingles(tokens: List[str], k: int) -> List[str]:
    if len(tokens) < k:
        return []
    return [" ".join(tokens[i:i+k]) for i in range(len(tokens)-k+1)]

def jaccard_shingle(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

# ------------------- CORRUPTION OPERATORS -------------------
STOP = set("""
a an the and or for to in of with on at from by as is are was were be been being
""".split())

SYN = {
    "good": ["excellent","fine","decent"],
    "new": ["recent", "fresh","modern"],
    "first": ["1st","initial","primary"],
    "last": ["final","ultimate","concluding"],
    "long": ["lengthy","extended","prolonged"],
    "little": ["small","tiny","mini"],
    "own": ["personal","private","individual"],
    "other": ["different","alternative","additional"],
    "old": ["ancient","aged","antique"],
    "right": ["correct","accurate","true"],
    "quick": ["fast","rapid","swift"],
    "buy": ["purchase","acquire","get"],
    "price": ["cost","pricing","rate"],
    "great": ["excellent","good","superb"],
    "small": ["little","compact","mini"],
    "big": ["large","huge","massive"],
    "different": ["distinct","diverse","varied"],
    "important": ["significant","crucial","vital"],
    "bad": ["poor","inferior","subpar"]
}

def flip_case(s: str, rng: random.Random) -> str:
    modes = [str.lower, str.upper, str.title]
    f = rng.choice(modes)
    return f(s)

def punct_jitter(s: str, rng: random.Random) -> str:
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    # randomly drop some commas/periods
    s = re.sub(r",", lambda m: "" if rng.random()<0.3 else ",", s)
    s = re.sub(r"\.", lambda m: "" if rng.random()<0.2 else ".", s)
    return s

def stopword_tidy(tokens: List[str], rng: random.Random) -> List[str]:
    out=[]
    for t in tokens:
        if t in STOP and rng.random()<0.25:
            continue
        out.append(t)
    # maybe insert "the" occasionally
    if rng.random()<0.2 and len(out)>5:
        pos = rng.randrange(1, len(out))
        out.insert(pos, "the")
    return out

def synonym_sub(tokens: List[str], rng: random.Random, rate=0.1) -> List[str]:
    out=[]
    for t in tokens:
        if t in SYN and rng.random()<rate:
            out.append(rng.choice(SYN[t]))
        else:
            out.append(t)
    return out

def sentence_split(s: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", s)
    return [p for p in parts if p.strip()]

def sentence_join(parts: List[str]) -> str:
    return " ".join(parts)

def sentence_shuffle(s: str, rng: random.Random) -> str:
    parts = sentence_split(s)
    if len(parts) > 1:
        i = rng.randrange(len(parts))
        j = rng.randrange(len(parts))
        parts[i], parts[j] = parts[j], parts[i]
    return sentence_join(parts)

def truncate_tokens(tokens: List[str], rng: random.Random, frac=0.1) -> List[str]:
    n = len(tokens)
    cut = max(1, int(n*frac))
    if rng.random()<0.5:
        return tokens[cut:]
    else:
        return tokens[:-cut] if n>cut else tokens

def expand_boilerplate(s: str, rng: random.Random) -> str:
    bp = [
        "© 2025 Example Inc. All rights reserved.",
        "For more info visit example.com/support.",
        "Terms apply. See website for details."
    ]
    return s + " " + rng.choice(bp)

def ocr_noise(s: str, rng: random.Random, rate=0.02) -> str:
    table = {"l":"1","1":"l","O":"0","0":"O","S":"5","5":"S"}
    chars=list(s)
    for i,c in enumerate(chars):
        if rng.random()<rate and c in table:
            chars[i] = table[c]
    return "".join(chars)

def html_wrap(s: str, rng: random.Random) -> str:
    wraps = [
        ("<p>","</p>"),
        ("<div>","</div>"),
        (""," &amp; "),
    ]
    left,right = rng.choice(wraps)
    return f"{left}{s}{right}"

# compose per-band recipes
def apply_band_ops(text: str, band: str, rng: random.Random) -> str:
    t0 = text
    if band=="D0":
        return t0
    # tokenize as needed
    toks = tokenize(t0)
    if band=="D1":
        if rng.random()<0.5: t0 = flip_case(t0, rng)
        t0 = punct_jitter(t0, rng)
        if rng.random()<0.6: toks = stopword_tidy(toks, rng)
        return " ".join(toks) if rng.random()<0.7 else t0

    if band in ("D2","D3","D4"):
        if rng.random()<0.4: t0 = sentence_shuffle(t0, rng)
        toks = tokenize(t0)
        rate = {"D2":0.05,"D3":0.10,"D4":0.18}[band]
        toks = synonym_sub(toks, rng, rate=rate)
        if rng.random()<0.5: toks = stopword_tidy(toks, rng)
        if rng.random()<0.6: toks = truncate_tokens(toks, rng, frac={"D2":0.05,"D3":0.1,"D4":0.2}[band])
        s = " ".join(toks)
        if rng.random()<0.4: s = expand_boilerplate(s, rng)
        if rng.random()<0.3: s = ocr_noise(s, rng, rate={"D2":0.005,"D3":0.01,"D4":0.02}[band])
        if rng.random()<0.2: s = html_wrap(s, rng)
        return s

    return t0
# ------------------- END CORRUPTIONS -------------------

# ---------- Spark setup ----------
spark = SparkSession.builder \
    .appName("DataSynthesis") \
    .master("local[*]") \
    .config("spark.driver.memory", "200g") \
    .config("spark.executor.memory", "200g") \
    .config("spark.driver.memoryOverhead", "10g") \
    .config("spark.executor.memoryOverhead", "10g") \
    .config("spark.sql.shuffle.partitions", "768") \
    .config("spark.local.dir", "/home/amd/workspace/tmp") \
    .getOrCreate()

base = spark.read.json(INPUT_PATH)
base = base.withColumnsRenamed({"id":"doc_id", "token_count":"length_tokens"})
# basic sanity filter
base = base.filter(col("text").isNotNull() & (col("text")!=""))

# Add length buckets to mimic real distributions
tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+")
pipe = Pipeline(stages=[tokenizer]).fit(base)
base = pipe.transform(base).withColumn("length_tokens", size(col("tokens"))).drop("tokens")

# stratify by length
short = base.filter(col("length_tokens")<100)
med   = base.filter(col("length_tokens").between(100,499))
long  = base.filter(col("length_tokens").between(500,1000))
xlong = base.filter(col("length_tokens")>1000)

def sample_df(df, n):
    return df.orderBy(monotonically_increasing_id()).limit(n)

# Allocate per band across length buckets (40% med, 40% short, 20% long)
def alloc(n):
    a_short = int(n*0.4)
    a_med   = int(n*0.4)
    a_long  = n - a_short - a_med
    return a_short, a_med, a_long

from itertools import chain
band_samples = {}
for band, n in TARGET.items():
    s,m,l = alloc(n)
    band_samples[band] = (sample_df(short, s)
                          .unionByName(sample_df(med, m))
                          .unionByName(sample_df(long, l))
                          .withColumn("band_target", lit(band)))

# Build a union of all bases used for variants/negatives (and keep a map)
bases = None
for band, df in band_samples.items():
    bases = df if bases is None else bases.unionByName(df)
bases = bases.dropDuplicates(["doc_id"])

# UDFs to apply ops & compute jaccard
rng_broadcast_seed = SEED
@udf(StringType())
def make_variant(text, band, id_):
    rng = random.Random(rng_broadcast_seed + int(id_))
    return apply_band_ops(text, band, rng)

def jaccard_tokens_udf(k=K_SHINGLE):
    def f(a, b):
        ta = tokenize(a)
        tb = tokenize(b)
        return float(jaccard_shingle(shingles(ta, k), shingles(tb, k)))
    return udf(f, DoubleType())

# Create variants for positives (D0..D4)
pos_variants = []
for band in ["D0","D1","D2","D3","D4"]:
    df = band_samples[band]
    dfv = (df.withColumn("variant_text", make_variant(col("text"), col("band_target"), col("doc_id")))
             .withColumn("variant_of", col("doc_id")))
    pos_variants.append(dfv)

pos_all = None
for d in pos_variants:
    pos_all = d if pos_all is None else pos_all.unionByName(d)

# Compute true Jaccard and finalize bands by measured value
jacc_udf = jaccard_tokens_udf(K_SHINGLE)
pos_all = (pos_all
           .withColumn("jaccard_true", jacc_udf(col("text"), col("variant_text"))))

# Band from measured Jaccard
def band_from_jacc(j):
    if j >= 0.999: return "D0"
    if j >= 0.90:  return "D1"
    if j >= 0.80:  return "D2"
    if j >= 0.65:  return "D3"
    if j >= 0.50:  return "D4"
    return "N"
@udf(StringType())
def band_assign(j):
    return band_from_jacc(j)

pos_all = pos_all.withColumn("band", band_assign(col("jaccard_true")))

# Assign new variant ids
variants_docs = (pos_all
                 .select(col("variant_of").alias("source_id"),
                         col("variant_text").alias("text"),
                         col("band"),
                         col("band_target"))
                 .withColumn("variant_id", monotonically_increasing_id()))

# Build documents table (bases + variants)
docs_base = (bases.select(col("doc_id"),
                          col("doc_id").alias("source_id"),
                          col("text"),
                          col("length_tokens"))
                  .withColumn("variant_of", lit(None).cast(LongType()))
                  .withColumn("band", lit("ORIG"))
                  .withColumn("band_target", lit("ORIG")))

docs_var = (variants_docs
            .select(col("variant_id").alias("doc_id"),
                    col("source_id"),
                    col("text"),
                    )
            .withColumn("variant_of", col("source_id"))
            .withColumn("band", col("band"))
            .withColumn("band_target", col("band_target")))

documents = (docs_base.select("doc_id","source_id","variant_of","text","band","band_target")
             .unionByName(docs_var.select("doc_id","source_id","variant_of","text","band","band_target")))

# Recompute token length for variants
documents = Pipeline(stages=[RegexTokenizer(inputCol="text", outputCol="toks", pattern="\\W+")])\
    .fit(documents).transform(documents).withColumn("length_tokens", size(col("toks"))).drop("toks")

# Positive pairs (source_id, variant_id)
pairs_pos = (variants_docs
             .select(col("source_id").alias("id_a"),
                     col("variant_id").alias("id_b"),
                     lit(1).alias("label"),
                     col("band"),
                     col("band_target"))
             )

# ------------------- HARD NEGATIVES -------------------
# Build a light TF-IDF to find topical neighbors with low Jaccard
# (You can skip or replace with your own category-based sampler.)
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

tf_tokenizer = Tokenizer(inputCol="text", outputCol="w")
tf_hash = HashingTF(inputCol="w", outputCol="tf", numFeatures=1<<18)
idf = IDF(inputCol="tf", outputCol="tfidf")
tf_pipe = Pipeline(stages=[tf_tokenizer, tf_hash, idf]).fit(bases)
bases_tfidf = tf_pipe.transform(bases)

# For speed, we’ll sample a subset of bases to seed negatives
neg_seed = bases_tfidf.orderBy(monotonically_increasing_id()).limit(int(TARGET["N"] * 1.2))

# Create random partner pool (approximation for demo; replace with ANN or exact cosine if desired)
pool = bases.select(col("doc_id").alias("doc_id_b"), col("text").alias("text_b"))

cand = (neg_seed
        .select(col("doc_id").alias("id_a"), col("text").alias("text_a"))
        .join(pool, how="cross"))

# Compute Jaccard and keep < 0.5 for negatives, but subsample to TARGET["N"]
cand = cand.withColumn("jacc", jacc_udf(col("text_a"), col("text_b")))
neg = (cand.filter(col("jacc") < 0.5)
            .select("id_a", col("doc_id_b").alias("id_b"))
            .dropDuplicates(["id_a","id_b"])
            .limit(TARGET["N"])
            .withColumn("label", lit(0))
            .withColumn("band", lit("N"))
            .withColumn("band_target", lit("N")))

# Build final pairs
pairs = (pairs_pos
         .unionByName(neg)
         .withColumn("id_min", (col("id_a") < col("id_b")).cast("int"))
         .withColumn("a", col("id_a"))
         .withColumn("b", col("id_b"))
         .withColumn("id_a", (col("a") * col("id_min") + col("b") * (1-col("id_min"))).cast("long"))
         .withColumn("id_b", (col("b") * col("id_min") + col("a") * (1-col("id_min"))).cast("long"))
         .drop("a","b","id_min"))

# Compute true Jaccard for all pairs using the *final* documents
docs_for_jacc = documents.select("doc_id","text").withColumnRenamed("text","text_final")
pairs = (pairs
         .join(docs_for_jacc.withColumnRenamed("doc_id","aid"), pairs.id_a==col("aid"))
         .join(docs_for_jacc.withColumnRenamed("doc_id","bid"), pairs.id_b==col("bid"))
         .withColumn("jaccard_true", jacc_udf(col("text_final"), col("text_final_1")))
         .drop("aid","bid","text_final","text_final_1"))

# Save outputs
# documents.write.mode("overwrite").parquet(OUTPUT_DOCS)
# pairs.write.mode("overwrite").parquet(OUTPUT_PAIRS)

spark.stop()
