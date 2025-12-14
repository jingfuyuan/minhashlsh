# MinHash LSH Deduplication Benchmarking (under development)

## 📘 Overview
This project is to benchmark the performance and quality of **MinHash LSH** (Locality Sensitive Hashing) for approximate text deduplication.  
It simulates large‐scale text deduplication as used in **LLM pretraining corpora**, where we aim to identify near‐duplicate documents and paragraphs efficiently.

The system is implemented with **PySpark**

## To do
- Synthetic dataset generation with controlled duplicate/near‐duplicate text.
- MinHash LSH–based deduplication pipeline.
- Performance benchmarking (runtime, scalability).
- Quality evaluation (precision, recall, F1).
- Impact of different parameters
- hardware utlization analysis
