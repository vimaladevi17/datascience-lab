'''
Experiment Title

Loading and Processing Large Datasets using PySpark (RDD and DataFrame Operations)

Aim

To load and process large-scale datasets using PySpark by implementing Resilient Distributed Dataset (RDD) and DataFrame operations in a distributed computing environment.

Description

Big Data technologies are used to process large volumes of structured and unstructured data efficiently. Apache Spark is a distributed computing framework designed for high-speed data processing. PySpark is the Python API for Apache Spark.

In this experiment, large datasets are loaded into Spark, and processing is performed using two core abstractions:

RDD (Resilient Distributed Dataset)
DataFrame

Both approaches are used to perform transformations and actions on distributed datasets.

1️⃣ Apache Spark

Apache Spark is an open-source distributed computing system designed for fast computation. It supports:

In-memory computation
Distributed data processing
Fault tolerance
Parallel execution

Spark runs on clusters and distributes tasks across multiple nodes.

2️⃣ Resilient Distributed Dataset (RDD)

RDD is the fundamental data abstraction in Spark.

Key Features:
Immutable distributed collection of objects
Fault tolerant
Lazy evaluation
Supports parallel operations
RDD Operations:

Transformations (Lazy)

map()
filter()
flatMap()
reduceByKey()

Actions (Trigger Execution)

collect()
count()
take()
reduce()

RDD is low-level but provides fine-grained control over distributed computation.

3️⃣ DataFrame

DataFrame is a higher-level abstraction built on top of RDD.

Key Features:
Structured data format (like tables)
Schema support
Optimized execution using Catalyst Optimizer
Faster than RDD for structured data

DataFrames support SQL queries and optimized execution plans.

4️⃣ Processing Steps in the Experiment
Initialize SparkSession
Load large dataset (CSV/JSON)
Perform RDD operations:
Mapping
Filtering
Aggregation
Perform DataFrame operations:
Select columns
Filter rows
GroupBy and aggregation
SQL queries
Compare performance of RDD vs DataFrame

5️⃣ Advantages of PySpark for Large Data
Handles terabytes of data
Distributed memory processing
Faster than traditional MapReduce
Scalable and fault-tolerant
'''

# ==========================================================
# Load and Process Large Dataset using PySpark
# RDD and DataFrame Operations
# ==========================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum as spark_sum

# ==========================================================
# 1. Initialize Spark Session
# ==========================================================

spark = SparkSession.builder \
    .appName("Large Dataset Processing") \
    .getOrCreate()

print("Spark Session Created Successfully")

# ==========================================================
# 2. Load Dataset (CSV File)
# ==========================================================

# Replace with your dataset path
file_path = r"C:\Users\VIMALA\Downloads\datascience lab\cleaned_data.csv"

df = spark.read.csv(file_path, header=True, inferSchema=True)

print("Dataset Loaded Successfully")
df.printSchema()
df.show(5)

# ==========================================================
# 3. RDD OPERATIONS
# ==========================================================

print("\n=========== RDD OPERATIONS ===========")

# Convert DataFrame to RDD
rdd = df.rdd

# Example 1: Count total records
total_records = rdd.count()
print("Total Records:", total_records)

# Example 2: Filter rows where target = 1
# Note: This example assumes a 'target' column exists. If not, it will fail.
# For california_housing_train.csv, 'median_house_value' could be used instead.
filtered_rdd = rdd.filter(lambda row: row["target"] == 1)
print("Count where target = 1:", filtered_rdd.count())

# Example 3: Map operation (Select specific column)
# Note: Assuming 'feature_0' exists or using an appropriate column from california_housing_train.csv
feature_rdd = rdd.map(lambda row: row["age"])

print("First 5 age values:", feature_rdd.take(5))

# Example 4: Compute average of feature_0 using RDD
feature_sum = feature_rdd.reduce(lambda a, b: a + b)
feature_avg = feature_sum / total_records

print("Average of age (RDD):", feature_avg)

# ==========================================================
# 4. DATAFRAME OPERATIONS
# ==========================================================

print("\n=========== DATAFRAME OPERATIONS ===========")

# Example 1: Select Columns
df.select("age", "chol").show(5)

# Example 2: Filter Condition
# df.filter(col("target") == 1).show(5) # Original, will be commented out
df.filter(col("chol") > 200).show(5)

# Example 3: GroupBy and Aggregation
# df.groupBy("target").count().show() # Original, will be commented out
df.groupBy("target").count().show()

# Example 4: Average calculation using DataFrame
df.select(avg("age")).show()

# Example 5: Aggregation using multiple functions
# df.groupBy("target").agg(
#     avg("feature_0").alias("Average_feature_0"),
#     spark_sum("feature_1").alias("Total_feature_1")
# ).show() # Original, will be commented out
df.groupBy("target").agg(
    avg("age").alias("Average_Age"),
    spark_sum("trestbps").alias("Total_trestbps")
).show()

# ==========================================================
# 5. SQL OPERATIONS (Optional Advanced Part)
# ==========================================================

print("\n=========== SQL OPERATIONS ===========")

df.createOrReplaceTempView("data_table")

result = spark.sql("""
    SELECT target, COUNT(*) as total_count,
           AVG(age) as avg_age
    FROM data_table
    GROUP BY target
""")

result.show()

# ==========================================================
# 6. Stop Spark Session
# ==========================================================

spark.stop()

print("Spark Session Stopped Successfully")