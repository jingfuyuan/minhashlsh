from pyspark.sql import SparkSession
from pyspark.sql.functions import size, col, floor, sha2, broadcast, explode, udf, conv, substring, lit
from pyspark.sql.types import BooleanType
from pyspark.ml.feature import MinHashLSH, HashingTF, RegexTokenizer, NGram
from pyspark.ml import Pipeline
import json
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
        tf_bin = HashingTF(inputCol="shingles", outputCol="features", numFeatures=1<<22, binary=True)

        # Run first two stages manually so we can filter without a Python UDF
        df_tok = tokenizer.transform(df)
        df_ngrams = ngrams.transform(df_tok)

        # Keep only rows that produced at least one shingle (avoids empty feature vectors)
        df_ngrams = df_ngrams.filter(size(col("shingles")) > 0)

        # Apply hashing
        docs_with_features = tf_bin.transform(df_ngrams)

        return docs_with_features


    def deduplicate_shard(self, shard_df):
        """
        Perform MinHash LSH deduplication on a single shard.
        
        Args:
            shard_df: DataFrame containing a single shard
            
        Returns:
            DataFrame with duplicates removed
        """
        # shard_df = shard_df.persist()
        initial_count = shard_df.count()
        print(f"Processing shard with {initial_count} documents")

        if initial_count == 0:
            return shard_df
        
        # Initialize MinHash LSH
        mh_lsh = MinHashLSH(
            inputCol="features", 
            outputCol="hashes", 
            numHashTables=self.num_hash_tables
        )
        
        # Fit the model
        model = mh_lsh.fit(shard_df)
        
        # Transform to get hash values
        hashed_df = model.transform(shard_df).persist()

        # identify overlarge buckets and drop them
        buckets = hashed_df.select(
            col("doc_id"),
            explode(col("hashes")).alias("bucket_id")
        )
        bucket_size = buckets.groupBy("bucket_id").count()
        MAX_BUCKET_SIZE = 500
        overlarge_buckets = bucket_size.filter(col("count") > MAX_BUCKET_SIZE).select("bucket_id")
        docs_in_overlarge_buckets = (
            buckets.join(overlarge_buckets, "bucket_id", "inner")
                .select("doc_id")
                .distinct()
        )

        # Remove documents in overlarge buckets
        hashed_df = (
            hashed_df.join(broadcast(docs_in_overlarge_buckets), "doc_id", "left_anti")
        )

        print(f"After dropping overlarge buckets, {hashed_df.count()} documents remain in shard")

        # Find similar pairs using approximate similarity join
        # We use the same dataframe for both sides of the join
        similar_pairs = model.approxSimilarityJoin(
            hashed_df, 
            hashed_df, 
            threshold=self.jaccard_threshold,
            distCol="jaccard_distance"
        )
        # Filter out self-joins and ensure we don't have duplicate pairs
        duplicate_pairs = similar_pairs.filter(
            col("datasetA.doc_id") < col("datasetB.doc_id")
        ).select(
            col("datasetA.doc_id").alias("doc_id_a"),
            col("datasetB.doc_id").alias("doc_id_b"),
            col("jaccard_distance")
        ).persist()
        # print("\nSimilar pairs found:\n")
        # similar_pairs.filter(
        #     col("datasetA.doc_id") < col("datasetB.doc_id")
        #     ).select(col("datasetA.text").alias("text_a"),
        #              col("datasetB.text").alias("text_b"),
        #              col("jaccard_distance")).show(truncate=False)

        # Collect IDs of documents to remove (keep the one with smaller doc_id)
        docs_to_remove = duplicate_pairs.select("doc_id_b").distinct().persist()
        
        print(f"Found {duplicate_pairs.count()} duplicate pairs")
        print(f"Removing {docs_to_remove.count()} duplicate documents")
        
        # Remove duplicates by anti-joining
        deduplicated_df = hashed_df.join(
            broadcast(docs_to_remove), 
            hashed_df.doc_id == docs_to_remove.doc_id_b, 
            "left_anti"
        )

        # Remove depulicates using filtering with isin
        # remove_list = [row["doc_id_b"] for row in docs_to_remove.collect()]
        # deduplicated_df = shard_df.filter(~col("doc_id").isin(remove_list))
        
        return deduplicated_df.drop("features", "hashes")
    
    def process_corpus(self, input_path: str, output_path: str):
        """
        Complete pipeline to process the text corpus.
        
        Args:
            input_path: Path to input JSON files
            output_path: Path to save deduplicated results
            shard_size: Token count increment for sharding
        """
        self.start_time = time()
        print("Loading JSON files...")
        df = self.load_json_files(input_path)
        self.original_count = df.count()
        print(f"Loaded {self.original_count} documents")
        
        print("Creating shards...")
        df_sharded = self.create_shards(df)
        
        print("Preprocessing text for MinHash LSH...")
        df_preprocessed = self.preprocess_text(df_sharded)
        
        # Process each shard separately
        shard_ids = df_preprocessed.select("shard_id").distinct().collect()
        
        deduplicated_shards = []
        self.final_count = 0
        for row in shard_ids:
            shard_id = row["shard_id"]
            print(f"\nProcessing shard {shard_id}...")
            
            # Filter data for current shard
            shard_df = df_preprocessed.filter(col("shard_id") == shard_id)
            
            # Deduplicate the shard
            deduplicated_shard = self.deduplicate_shard(shard_df)
            self.final_count += deduplicated_shard.count()
            deduplicated_shards.append(deduplicated_shard)
        
        # Combine all deduplicated shards
        print("\nCombining deduplicated shards...")
        final_df = deduplicated_shards[0]
        for shard in deduplicated_shards[1:]:
            final_df = final_df.union(shard)
        print(f"Final dataset contains {self.final_count} documents")
        
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        # Save results
        print(f"Saving results to {output_path}...")
        final_df.select("doc_id", "source_id", "text", "token_count") \
               .write \
               .mode("overwrite") \
               .json(output_path)
        
        print("Deduplication completed!")
        self.end_time = time()
        # Show statistics
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
    arg_parser.add_argument("--num_hash_tables", type=int, default=5, help="Number of hash tables for MinHash LSH (default: 5).")
    arg_parser.add_argument("--jaccard_threshold", type=float, default=0.5, help="Jaccard similarity threshold for deduplication (default: 0.5).")
    arg_parser.add_argument("--shingle_size", type=int, default=3, help="Shingle size for tokenization (default: 3).")
    arg_parser.add_argument("--shard_method", type=str, default="hash", choices=["hash", "token_count"], help="Sharding method: 'hash' or 'token_count' (default: 'hash').")
    arg_parser.add_argument("--shard_count", type=int, default=50, help="Number of shards if using hash-based sharding (default: 50).")
    args = arg_parser.parse_args()
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("DocumentTokenCount") \
        .master("local[*]") \
        .config("spark.driver.memory", "600g") \
        .config("spark.executor.memory", "600g") \
        .config("spark.driver.memoryOverhead", "10g") \
        .config("spark.executor.memoryOverhead", "10g") \
        .config("spark.sql.shuffle.partitions", "768") \
        .config("spark.local.dir", "/home/amd/workspace/tmp") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.skweJoin.skewedPartitionThresholdInBytes", "256MB") \
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