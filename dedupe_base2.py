from pyspark.sql import SparkSession
from pyspark.sql.functions import transform, size, col, floor, sha2, broadcast, explode, udf
from pyspark.sql.functions import conv, substring, lit, expr
from pyspark.sql.types import BooleanType
from pyspark.ml.feature import MinHashLSH, HashingTF, RegexTokenizer, NGram
from pyspark.ml.functions import vector_to_array
import os
from typing import List
import argparse
from time import time

class TextCorpusDeduplicator:
    def __init__(self, spark_session, args):
        """
        Initialize the deduplicator with Spark session and MinHash parameters.
        
        Args:
            spark_session: Spark session
            num_hash_tables: Number of hash tables for MinHash LSH
        """
        
        self.spark = spark_session
        self.args = args
        self.shingle_size = args.shingle_size
        self.num_hash_tables = args.num_hash_tables
        self.jaccard_threshold = args.jaccard_threshold
        
    def load_json_files(self, input_path: str):
        """
        Load multiple JSON files and create a DataFrame.
        
        Args:
            input_path: Path to directory containing JSON files or single JSON file
            
        Returns:
            DataFrame with source_id, text, token_count columns
        """
        # Read JSON files
        try:
            df = self.spark.read.json(input_path)
        except Exception as e:
            raise RuntimeError(f"Error reading JSON files from {input_path}: {e}")
        
        # Rename 'id' to 'source_id' if necessary
        df = df.withColumnRenamed("id", "source_id")
        # use sha2 function to generate unique ids based on source_id
        df_with_id = df.withColumn("doc_id", sha2(col("source_id"), 256))
                
        return df_with_id.select("doc_id", "source_id", "text", "token_count")
    
    def create_shards(self, df):
        """
        Create shards based on token count with specified incremental size.
        
        Args:
            df: Input DataFrame            
        Returns:
            DataFrame with shard_id column added
        """
        # Calculate shard_id based on token_count
        if self.args.shard_method == "hash":
            num_shards = self.args.shard_count
            df_with_shards = df.withColumn(
                "shard_id", 
                (conv(substring("doc_id", 1, 15), 16, 10).cast("long") % lit(num_shards)).cast("int")
            )
        elif self.args.shard_method == "token_count":
            shard_bin_size = self.args.shard_bin_size
            df_with_shards = df.withColumn(
                "shard_id", 
                floor(col("token_count") / shard_bin_size)
            )
        
        # df_with_shards = df_with_shards.persist()
        # Show shard distribution
        shard_counts = df_with_shards.groupBy("shard_id").count().orderBy("shard_id")
        print("Shard distribution:")
        shard_counts.show()
        
        return df_with_shards
    
    def preprocess_text(self, df):
        """
        Preprocess text for MinHash LSH by tokenizing and creating feature vectors.
        
        Args:
            df: DataFrame with text column
            
        Returns:
            DataFrame with features column containing sparse vectors
        """
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tok", pattern="\\W+")
        ngrams = NGram(inputCol="tok", outputCol="shingles", n=self.shingle_size)
        tf_bin = HashingTF(inputCol="shingles", outputCol="features", numFeatures=1<<21, binary=True)

        # Run first two stages manually so we can filter without a Python UDF
        df_tok = tokenizer.transform(df)
        df_ngrams = ngrams.transform(df_tok)

        # Keep only rows that produced at least one shingle (avoids empty feature vectors)
        df_ngrams = df_ngrams.filter(size(col("shingles")) > 0)

        # Apply hashing
        docs_with_features = tf_bin.transform(df_ngrams)
        
        # print schema
        # docs_with_features.printSchema()

        return docs_with_features.select("doc_id", "source_id", "text", "token_count", "shard_id", "features")
    

    def deduplicate_shard(self, shard_df, output_path):
        """
        For a shard, identify duplicate pairs using LSH, remove duplicates and write to output.
        Banding: group consecutive hash values into band keys (size = band_size).
        return the number of documents removed.
        """
        if shard_df.rdd.isEmpty():
            return 0

        band_size = self.args.band_size
        shard_df = shard_df.persist()
        # Apply MinHash LSH
        mh_lsh = MinHashLSH(
            inputCol="features",
            outputCol="hashes",
            numHashTables=self.num_hash_tables
        )
        model = mh_lsh.fit(shard_df)
        hashed_df = model.transform(shard_df).drop("features")
        # hashed_df.show(5, truncate=False)

        # Flatten hashes (array<vector(1)>) -> array<long>
        hashed_df = hashed_df.withColumn(
            "hash_values",
            transform(col("hashes"), lambda h: vector_to_array(h)[0].cast("bigint"))
        ).select("doc_id", "hash_values").persist()
        # hashed_df.printSchema()


        # Build band keys expression dynamically
        # band_keys: array<string>
        band_expr = (
            "transform("
            f"sequence(0, int(floor(size(hash_values)/{band_size})) - 1), "
            "i -> concat_ws('_'"
            + "".join([f", cast(hash_values[i*{band_size}+{j}] as string)" for j in range(band_size)])
            + "))"
        )

        banded_df = hashed_df.select(
            "doc_id",
            "hash_values",
            expr(band_expr).alias("band_keys")
        ).filter(size(col("band_keys")) > 0)
        # banded_df.show(5, truncate=False)
        # Explode band keys to buckets
        buckets = banded_df.select(
            col("doc_id"),
            explode(col("band_keys")).alias("band_id")
        )

        # Optional: drop oversized band buckets
        MAX_BUCKET_SIZE = self.args.max_bucket_size
        bucket_sizes = buckets.groupBy("band_id").count()
        overlarge_bands = bucket_sizes.filter(col("count") > MAX_BUCKET_SIZE).select("band_id").persist()
        docs_to_remove = None
        if not overlarge_bands.rdd.isEmpty():
            # add docs in the overlarge bands to removal list
            docs_to_remove = buckets.join(broadcast(overlarge_bands), "band_id", "inner").select("doc_id").distinct()
            # remove overlarge bands from buckets
            pruned_buckets = buckets.join(broadcast(overlarge_bands), "band_id", "left_anti")
            overlarge_bands.unpersist()
        else:
            pruned_buckets = buckets

        pruned_buckets = pruned_buckets.repartition(col("band_id")).persist()
        # Generate candidate pairs (each pair shares at least one band)
        a = pruned_buckets.alias("a")
        b = pruned_buckets.alias("b")
        candidate_pairs = a.join(
            b,
            (col("a.band_id") == col("b.band_id")) & (col("a.doc_id") < col("b.doc_id")),
            "inner"
        ).select(
            col("a.doc_id").alias("doc_id_a"),
            col("b.doc_id").alias("doc_id_b")
        ).distinct()

        if not candidate_pairs.rdd.isEmpty():
            # Attach hash values to compute Jaccard estimate (proportion of equal minhashes)
            hv = hashed_df.alias("hv")
            pairs_with_hashes = candidate_pairs \
                .join(hv, col("doc_id_a") == col("hv.doc_id")) \
                .select(col("doc_id_a"), col("doc_id_b"), col("hash_values").alias("hash_values_a")) \
                .join(hv, col("doc_id_b") == col("hv.doc_id")) \
                .select(col("doc_id_a"), col("doc_id_b"),
                        col("hash_values_a"),
                        col("hash_values").alias("hash_values_b"))
            # pairs_with_hashes.show(5, truncate=False)
            # matches = number of positions with equal hash; jaccard_est = matches / total
            similarity_scored = pairs_with_hashes.select(
                "doc_id_a",
                "doc_id_b",
                expr("""
                aggregate(
                    zip_with(hash_values_a, hash_values_b, (x,y) -> int(x=y)),
                    0,
                    (acc,x) -> acc + x
                )
                """).alias("matches"),
                expr("size(hash_values_a)").alias("total")
            ).withColumn("jaccard_est", col("matches") / col("total"))

            # similarity_scored.show(5, truncate=False)
            # Filter by threshold
            duplicate_pairs = similarity_scored.filter(
                col("jaccard_est") >= lit(self.jaccard_threshold)
            ).select("doc_id_a", "doc_id_b")

            # debug: show some duplicate pairs with text
            # duplicate_pairs.join(
            #     shard_df.select("doc_id", "text").alias("dfa"),
            #     col("doc_id_a") == col("dfa.doc_id"),
            #     "left"
            # ).join(
            #     shard_df.select("doc_id", "text").alias("dfb"),
            #     col("doc_id_b") == col("dfb.doc_id"),
            #     "left"
            # ).show(5, truncate=False)

            # Choose to remove doc_id_b (larger id) per pair
            if docs_to_remove is not None:
                docs_to_remove = docs_to_remove.union(
                    duplicate_pairs.select(col("doc_id_b").alias("doc_id")).distinct()
                )
            else:
                docs_to_remove = duplicate_pairs.select(col("doc_id_b").alias("doc_id")).distinct()
        
        if docs_to_remove is None or docs_to_remove.rdd.isEmpty():
            # no documents to remove
            survivors = shard_df.select(col("source_id").alias("id"), "text", "token_count")
            removal_count = 0
        else:
            docs_to_remove = docs_to_remove.persist()
            removal_count = docs_to_remove.count()
            # remove duplicates from shard_df and write to output
            survivors = shard_df.alias("a").join(
                    broadcast(docs_to_remove.alias("b")),
                    col("a.doc_id") == col("b.doc_id"),
                    "left_anti"
                ).select(col("source_id").alias("id"), "text", "token_count")
        
        # Write survivors to output path
        survivors.coalesce(self.args.output_files_per_shard) \
         .write.mode("append") \
         .json(output_path)
        
        # clean up
        docs_to_remove.unpersist()
        pruned_buckets.unpersist()
        hashed_df.unpersist()
        shard_df.unpersist()
        # return the number of documents removed
        return removal_count

    def process_corpus(self, input_path: str, output_path: str):

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        else:
            # clear existing files in output_path
            print(f"{os.path.abspath(output_path)} exists. overwriting this folder")
            for filename in os.listdir(output_path):
                file_path = os.path.join(output_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        self.start_time = time()
        print("Loading JSON files...")
        df = self.load_json_files(input_path)
        self.original_count = df.count()
        self.removal_count = 0
        print(f"Loaded {self.original_count} documents")

        print("Creating shards...")
        df_sharded = self.create_shards(df)

        print("Preprocessing text for MinHash LSH...")
        df_preprocessed = self.preprocess_text(df_sharded).cache()

        shard_ids = [r["shard_id"] for r in df_preprocessed.select("shard_id").distinct().collect()]

        for shard_id in shard_ids:
            print(f"\nProcessing shard {shard_id}...")
            shard_df = df_preprocessed.filter(col("shard_id") == shard_id)
            shard_removals = self.deduplicate_shard(shard_df, output_path)
            print(f"Shard {shard_id}: removed {shard_removals} documents")
            self.removal_count += shard_removals

        df_preprocessed.unpersist()
        print(f"Total documents removed: {self.removal_count} out of {self.original_count}")
        self.final_count = self.original_count - self.removal_count
        print(f"Final dataset contains {self.final_count} documents")



        print("Deduplication completed!")
        self.end_time = time()
        self.show_statistics()
      
    def show_statistics(self):
        """Show deduplication statistics."""
        original_count = self.original_count
        final_count = self.final_count
        removed_count = original_count - final_count
        removal_percentage = (removed_count / original_count) * 100
        
        print(f"\n=== Deduplication Statistics ===")
        print(f"Original documents: {original_count}")
        print(f"Final documents: {final_count}")
        print(f"Removed duplicates: {removed_count}")
        print(f"Removal percentage: {removal_percentage:.2f}%")
        print(f"total run time: {self.end_time - self.start_time:.2f} seconds")

def main():
    """Example usage of the TextCorpusDeduplicator."""

    arg_parser = argparse.ArgumentParser(description="Deduplicate text corpus using MinHash LSH.")
    arg_parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON files (directory or single file).")
    arg_parser.add_argument("--output_path", type=str, required=True, help="Path to save deduplicated results.")
    arg_parser.add_argument("--shard_bin_size", type=int, default=100, help="Token count bin size for sharding (default: 100).")
    arg_parser.add_argument("--num_hash_tables", type=int, default=6, help="Number of hash tables for MinHash LSH (default: 5).")
    arg_parser.add_argument("--jaccard_threshold", type=float, default=0.5, help="Jaccard similarity threshold for deduplication (default: 0.5).")
    arg_parser.add_argument("--shingle_size", type=int, default=3, help="Shingle size for tokenization (default: 3).")
    arg_parser.add_argument("--shard_method", type=str, default="hash", choices=["hash", "token_count"], help="Sharding method: 'hash' or 'token_count' (default: 'hash').")
    arg_parser.add_argument("--shard_count", type=int, default=50, help="Number of shards if using hash-based sharding (default: 50).")
    arg_parser.add_argument("--band_size", type=int, default=2,
                            help="Number of consecutive hash values combined into one band (default: 2).")
    arg_parser.add_argument("--max_bucket_size", type=int, default=200,
                            help="Maximum allowed size of a band bucket before it is dropped (default: 200).")
    arg_parser.add_argument("--output_files_per_shard", type=int, default=4, 
                            help="Number of output files per shard (default: 4).")
    args = arg_parser.parse_args()
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("DocumentTokenCount") \
        .master("local[*]") \
        .config("spark.driver.memory", "500g") \
        .config("spark.executor.memory", "500g") \
        .config("spark.driver.memoryOverhead", "10g") \
        .config("spark.executor.memoryOverhead", "10g") \
        .config("spark.sql.shuffle.partitions", "384") \
        .config("spark.local.dir", "/home/amd/workspace/tmp") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.skewJoin.skewedPartitionThresholdInBytes", "256MB") \
        .config("spark.sql.autoBroadcastJoinThreshold", "64MB") \
        .getOrCreate()    

    # Initialize the deduplicator
    deduplicator = TextCorpusDeduplicator(spark, args)
    
    # Process the corpus
    deduplicator.process_corpus(
        input_path=args.input_path,
        output_path=args.output_path
    )
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()