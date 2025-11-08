# filepath: \home\amd\workspace\minhashlsh\doc_count.py
# use pyspark to read json files from a directory
# load the "id" and "text" fields from each json file
# count the number of tokens in the "text" field for each row
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, split
import sys
import os

def main(input_path, output_path):
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("DocumentTokenCount") \
        .master("local[*]") \
        .config("spark.driver.memory", "200g") \
        .config("spark.executor.memory", "200g") \
        .config("spark.driver.memoryOverhead", "10g") \
        .config("spark.executor.memoryOverhead", "10g") \
        .config("spark.sql.shuffle.partitions", "768") \
        .config("spark.local.dir", "/home/amd/workspace/tmp") \
        .getOrCreate()
    
    try:
        # Read JSON files from directory
        df = spark.read.json(input_path)
        
        # Select only id and text columns
        df_selected = df.select("id", "text")
        
        # Count tokens by splitting text on whitespace and getting array size
        df_with_token_count = df_selected.withColumn(
            "token_count", 
            size(split(col("text"), "\\s+"))
        )
        # keep the rows where token_count >= 20 and token_count <= 4000 
        df_with_token_count = df_with_token_count.filter(
            (col("token_count") >= 20) & (col("token_count") <= 2000)
        )
        # count how many rows df_with_token_count has
        row_count = df_with_token_count.count()
        print(f"Total number of rows in the dataset before deduplication: {row_count}")

        # remove deduplicate texts
        df_with_token_count = df_with_token_count.dropDuplicates(["text"])

        # count how many rows df_with_token_count has
        row_count = df_with_token_count.count()
        print(f"Total number of rows in the dataset after deduplication: {row_count}")

        # Show some results
        print("Sample data with token counts:")
        df_with_token_count.show(10, truncate=True)

        # create a histogram of token counts with 40 bins
        histogram = df_with_token_count.select("token_count").rdd.flatMap(lambda x: x).histogram(50)
        print("Token count histogram (bins and counts):")
        print(histogram)
        
        # Write results to output path
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        df_with_token_count.write \
            .mode("overwrite") \
            .json(output_path)
        
        print(f"Results written to: {output_path}")
        
    except Exception as e:
        print(f"Error processing files: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python doc_count.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)