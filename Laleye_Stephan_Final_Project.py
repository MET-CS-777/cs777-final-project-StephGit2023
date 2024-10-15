##... CS777 Project Code...
#
##... 10 / 13 / 24
#
##... Stephan Laleye


from __future__ import print_function
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, row_number
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

if __name__ == "__main__":
    if len(sys.argv) != 4: 
        print("Usage: <output_file_1> <output_file_2> <output_file_3>", file=sys.stderr)
        exit(-1)

    output_file_1 = sys.argv[1]
    output_file_2 = sys.argv[2]
    output_file_3 = sys.argv[3]

    # Initialize Spark session
    sc = SparkContext(appName = "Final-Project")
    spark = SparkSession(sc)

    # Load Parquet file from Google Cloud Storage
    proj_df = spark.read.parquet("gs://cs777_final_proj_2024/CS777_Project_Dataset.parquet")

    # Preprocessing: Encoding and Feature Engineering
    ethnicity_indexer = StringIndexer(inputCol="Ethnicity", outputCol="ethnicity_index")
    ethnicity_encoder = OneHotEncoder(inputCol="ethnicity_index", outputCol="ethnicity_encoded")
    assembler = VectorAssembler(inputCols=["ethnicity_encoded", "Rank", "Count"], outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    pipeline = Pipeline(stages=[ethnicity_indexer, ethnicity_encoder, assembler, scaler])
    proj_df = pipeline.fit(proj_df).transform(proj_df)

    # Split data into train (80%) and test (20%) sets
    train_df, test_df = proj_df.randomSplit([0.8, 0.2], seed=1234)

    # K-means clustering on the training set
    kmeans = KMeans(featuresCol="scaled_features", k=5, seed=1)
    model = kmeans.fit(train_df)

    # Apply the model to the test set
    predictions = model.transform(test_df)

    # Register the DataFrame as a SQL temporary table
    predictions.createOrReplaceTempView("baby_names")

    # ---- Output 1: Define Ethnicity for Each Cluster ----
    ethnicity_per_cluster = predictions.groupBy("prediction", "Ethnicity").count()
    ethnicity_cluster_df = ethnicity_per_cluster.withColumn("rank", row_number().over(
        Window.partitionBy("prediction").orderBy(col("count").desc()))).filter(col("rank") == 1)

    # Output: Cluster to Ethnicity mapping
    ethnicity_cluster_df.select("prediction", "Ethnicity").coalesce(1).write.csv(output_file_1, mode="overwrite", header=True)

    # ---- Output 2: Top 10 Names Found Within Each Cluster/Ethnicity ----
    top_names = spark.sql("""
    SELECT prediction, Ethnicity, `Child's First Name`, COUNT(*) as name_count
    FROM baby_names
    GROUP BY prediction, Ethnicity, `Child's First Name`
    ORDER BY prediction, Ethnicity, name_count DESC
    """)

    top_names.createOrReplaceTempView("top_names")
    
    # Select top 10 names within each cluster/ethnicity
    top_names_df = spark.sql("""
    SELECT prediction, Ethnicity, `Child's First Name`, name_count
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY prediction, Ethnicity ORDER BY name_count DESC) as rank
        FROM top_names
    ) tmp
    WHERE rank <= 10
    """)

    # Output: Top 10 names per cluster/ethnicity
    top_names_df.coalesce(1).write.csv(output_file_2, mode="overwrite", header=True)

    # ---- Output 3: Names Found Across Two or More Clusters ----
    influence_df = predictions.groupBy("prediction", "Child's First Name").agg(countDistinct("ethnicity_index").alias("distinct_ethnicities"))
    influential_names = influence_df.filter(col("distinct_ethnicities") > 1)

    # Select top influential names for each cluster
    influential_names.createOrReplaceTempView("influential_names")
    top_influential_names = spark.sql("""
    SELECT prediction, `Child's First Name`, distinct_ethnicities
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY prediction ORDER BY distinct_ethnicities DESC) as rank
        FROM influential_names
    ) tmp
    WHERE rank <= 5
    """)

    # Output: Names found across multiple clusters
    top_influential_names.coalesce(1).write.csv(output_file_3, mode="overwrite", header=True)

    # Stop the Spark session
    spark.stop()
