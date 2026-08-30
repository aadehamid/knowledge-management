title: Comparing Performance of Big Data File Formats: A Practical Guide | Towards Data Science
description: Parquet vs ORC vs Avro vs Delta Lake
author: Sarthak Sarbahi

# Comparing Performance of Big Data File Formats: A Practical Guide | Towards Data Science

![Photo by Viktor Talashuk on Unsplash](https://assets.insightmediagroup.io/media/wp-content/uploads/2024/01/0OzMvdkpL9ah7z3HF-scaled.jpg "Photo by Viktor Talashuk on Unsplash")

The big data world is full of various storage systems, heavily influenced by different file formats. These are key in nearly all data pipelines, allowing for efficient data storage and easier querying and information extraction. They are designed to handle the challenges of big data like size, speed, and structure.

Data engineers often face a plethora of choices. It's crucial to know which file format fits which scenario. This tutorial is designed to help with exactly that. You'll explore four widely used file formats: **Parquet**, **ORC**, **Avro**, and **Delta Lake**.

The tutorial starts with setting up the environment for these file formats. Then you'll learn to read and write data in each format. You'll also compare their performance while handling ***10 million*** records. And finally, you'll understand the appropriate scenarios for each. So let's get started!

### Table of contents {#table-of-contents}

1. [Environment setup](#66df)
2. [Working with Parquet](#f49e)
3. [Working with ORC](#f042)
4. [Working with Avro](#517c)
5. [Working with Delta Lake](#5cb3)
6. [When to use which file format?](#c715)

---

### Environment setup {#environment-setup}

In this guide, we're going to use JupyterLab with Docker and MinIO. Think of Docker as a handy tool that simplifies running applications, and MinIO as a flexible storage solution perfect for handling lots of different types of data. Here's how we'll set things up:

I'm not diving deep into every step here since there's already a great [tutorial](https://towardsdatascience.com/seamless-data-analytics-workflow-from-dockerized-jupyterlab-and-minio-to-insights-with-spark-sql-3c5556a18ce6) for that. I suggest checking it out first, then coming back to continue with this one.

> [**Seamless Data Analytics Workflow: From Dockerized JupyterLab and MinIO to Insights with Spark SQL**](https://towardsdatascience.com/seamless-data-analytics-workflow-from-dockerized-jupyterlab-and-minio-to-insights-with-spark-sql-3c5556a18ce6)

Once everything's ready, we'll start by preparing our sample data. Open a new Jupyter notebook to begin.

First up, we need to install the `s3fs` Python package, essential for working with MinIO in Python.

python

```
!pip install s3fs
```

Following that, we'll import the necessary dependencies and modules.

python

```
import osimport s3fsimport pysparkfrom pyspark.sql import SparkSessionfrom pyspark import SparkContextimport pyspark.sql.functions as Ffrom pyspark.sql import Rowimport pyspark.sql.types as Timport datetimeimport time
```

We'll also set some environment variables that will be useful when interacting with MinIO.

python

```
# Define environment variablesos.environ["MINIO_KEY"] = "minio"os.environ["MINIO_SECRET"] = "minio123"os.environ["MINIO_ENDPOINT"] = "http://minio1:9000"
```

Then, we'll set up our Spark session with the necessary settings.

python

```
# Create Spark sessionspark = SparkSession.builder     .appName("big_data_file_formats")     .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.11.1026,org.apache.spark:spark-avro_2.12:3.5.0,io.delta:delta-spark_2.12:3.0.0")     .config("spark.hadoop.fs.s3a.endpoint", os.environ["MINIO_ENDPOINT"])     .config("spark.hadoop.fs.s3a.access.key", os.environ["MINIO_KEY"])     .config("spark.hadoop.fs.s3a.secret.key", os.environ["MINIO_SECRET"])     .config("spark.hadoop.fs.s3a.path.style.access", "true")     .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")     .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")     .enableHiveSupport()     .getOrCreate()
```

Let's simplify this to understand it better.

- `spark.jars.packages`: Downloads the required JAR files from the [Maven repository](https://mvnrepository.com/). A Maven repository is a central place used for storing build artifacts like JAR files, libraries, and other dependencies that are used in Maven-based projects.
- `spark.hadoop.fs.s3a.endpoint`: This is the endpoint URL for MinIO.
- `spark.hadoop.fs.s3a.access.key` and `spark.hadoop.fs.s3a.secret.key`: This is the access key and secret key for MinIO. Note that it is the same as the username and password used to access the MinIO web interface.
- `spark.hadoop.fs.s3a.path.style.access`: It is set to true to enable path-style access for the MinIO bucket.
- `spark.hadoop.fs.s3a.impl`: This is the implementation class for S3A file system.
- `spark.sql.extensions`: Registers Delta Lake's SQL commands and configurations within the Spark SQL parser.
- `spark.sql.catalog.spark_catalog`: Sets the Spark catalog to Delta Lake's catalog, allowing table management and metadata operations to be handled by Delta Lake.

Choosing the right JAR version is crucial to avoid errors. Using the same Docker image, the JAR version mentioned here should work fine. If you encounter setup issues, feel free to leave a comment. I'll do my best to assist you :)

> [**GitHub - sarthak-sarbahi/big-data-file-formats**](https://github.com/sarthak-sarbahi/big-data-file-formats/tree/main)

Our next step is to create a big Spark dataframe. It'll have 10 million rows, divided into ten columns - half are text, and half are numbers.

python

```
# Generate sample datanum_rows = 10000000df = spark.range(0, num_rows)# Add columnsfor i in range(1, 10):  # Since we already have one column    if i % 2 == 0:        # Integer column        df = df.withColumn(f"int_col_{i}", (F.randn() * 100).cast(T.IntegerType()))    else:        # String column        df = df.withColumn(f"str_col_{i}", (F.rand() * num_rows).cast(T.IntegerType()).cast("string"))df.count()
```

Let's peek at the first few entries to see what they look like.

python

```
# Show rows from sample datadf.show(10,truncate = False)+---+---------+---------+---------+---------+---------+---------+---------+---------+---------+|id |str_col_1|int_col_2|str_col_3|int_col_4|str_col_5|int_col_6|str_col_7|int_col_8|str_col_9|+---+---------+---------+---------+---------+---------+---------+---------+---------+---------+|0  |7764018  |128      |1632029  |-15      |5858297  |114      |1025493  |-88      |7376083  ||1  |2618524  |118      |912383   |235      |6684042  |-115     |9882176  |170      |3220749  ||2  |6351000  |75       |3515510  |26       |2605886  |89       |3217428  |87       |4045983  ||3  |4346827  |-70      |2627979  |-23      |9543505  |69       |2421674  |-141     |7049734  ||4  |9458796  |-106     |6374672  |-142     |5550170  |25       |4842269  |-97      |5265771  ||5  |9203992  |23       |4818602  |42       |530044   |28       |5560538  |-75      |2307858  ||6  |8900698  |-130     |2735238  |-135     |1308929  |22       |3279458  |-22      |3412851  ||7  |6876605  |-35      |6690534  |-41      |273737   |-178     |8789689  |88       |4200849  ||8  |3274838  |-42      |1270841  |-62      |4592242  |133      |4665549  |-125     |3993964  ||9  |4904488  |206      |2176042  |58       |1388630  |-63      |9364695  |78       |2657371  |+---+---------+---------+---------+---------+---------+---------+---------+---------+---------+only showing top 10 rows
```

To understand the structure of our dataframe, we'll use `df.printSchema()` to see the types of data it contains. After this, we'll create four CSV files. These will be used for Parquet, Avro, ORC, and Delta Lake. We're doing this to avoid any bias in performance testing - using the same CSV lets Spark cache and optimize things in the background.

python

```
# Write 4 CSVs for comparing performance for every file typedf.write.csv("s3a://mybucket/ten_million_parquet.csv")df.write.csv("s3a://mybucket/ten_million_avro.csv")df.write.csv("s3a://mybucket/ten_million_orc.csv")df.write.csv("s3a://mybucket/ten_million_delta.csv")
```

Now, we'll make four separate dataframes from these CSVs, each one for a different file format.

python

```
# Read all four CSVs to create dataframesschema = T.StructType([    T.StructField("id", T.LongType(), nullable=False),    T.StructField("str_col_1", T.StringType(), nullable=True),    T.StructField("int_col_2", T.IntegerType(), nullable=True),    T.StructField("str_col_3", T.StringType(), nullable=True),    T.StructField("int_col_4", T.IntegerType(), nullable=True),    T.StructField("str_col_5", T.StringType(), nullable=True),    T.StructField("int_col_6", T.IntegerType(), nullable=True),    T.StructField("str_col_7", T.StringType(), nullable=True),    T.StructField("int_col_8", T.IntegerType(), nullable=True),    T.StructField("str_col_9", T.StringType(), nullable=True)])df_csv_parquet = spark.read.format("csv").option("header",True).schema(schema).load("s3a://mybucket/ten_million_parquet.csv")df_csv_avro = spark.read.format("csv").option("header",True).schema(schema).load("s3a://mybucket/ten_million_avro.csv")df_csv_orc = spark.read.format("csv").option("header",True).schema(schema).load("s3a://mybucket/ten_million_orc.csv")df_csv_delta = spark.read.format("csv").option("header",True).schema(schema).load("s3a://mybucket/ten_million_delta.csv")
```

And that's it! We're all set to explore these big data file formats.

### Working with Parquet {#working-with-parquet}

Parquet is a column-oriented file format that meshes really well with Apache Spark, making it a top choice for handling big data. It shines in analytical scenarios, particularly when you're sifting through data column by column.

One of its neat features is the ability to store data in a compressed format, with **snappy compression** being the go-to choice. This not only saves space but also enhances performance.

Another cool aspect of Parquet is its flexible approach to data schemas. You can start off with a basic structure and then smoothly expand by adding more columns as your needs grow. This adaptability makes it super user-friendly for evolving data projects.

Now that we've got a handle on Parquet, let's put it to the test. We're going to write 10 million records into a Parquet file and keep an eye on how long it takes. Instead of using the `%timeit` Python function, which runs multiple times and can be heavy on resources for big data tasks, we'll just measure it once.

python

```
# Write data as Parquetstart_time = time.time()df_csv_parquet.write.parquet("s3a://mybucket/ten_million_parquet2.parquet")end_time = time.time()print(f"Time taken to write as Parquet: {end_time - start_time} seconds")
```

For me, this task took ***15.14 seconds***, but remember, this time can change depending on your computer. For example, on a less powerful PC, it took longer. So, don't sweat it if your time is different. What's important here is comparing the performance across different file formats.

Next up, we'll run an aggregation query on our Parquet data.

python

```
# Perfom aggregation query using Parquet datastart_time = time.time()df_parquet = spark.read.parquet("s3a://mybucket/ten_million_parquet2.parquet")df_parquet .select("str_col_5","str_col_7","int_col_2") .groupBy("str_col_5","str_col_7") .count() .orderBy("count") .limit(1) .show(truncate = False)end_time = time.time()print(f"Time taken for query: {end_time - start_time} seconds")+---------+---------+-----+|str_col_5|str_col_7|count|+---------+---------+-----+|1        |6429997  |1    |+---------+---------+-----+
```

This query finished in ***12.33 seconds***. Alright, now let's switch gears and explore the ORC file format.

### Working with ORC {#working-with-orc}

The ORC file format, another column-oriented contender, might not be as well-known as Parquet, but it has its own perks. One standout feature is its ability to compress data even more effectively than Parquet, while using the same snappy compression algorithm.

It's a hit in the Hive world, thanks to its support for ACID operations in Hive tables. ORC is also tailor-made for handling large streaming reads efficiently.

Plus, it's just as flexible as Parquet when it comes to schemas - you can begin with a basic structure and then add more columns as your project grows. This makes ORC a robust choice for evolving big data needs.

Let's dive into testing ORC's writing performance.

python

```
# Write data as ORCstart_time = time.time()df_csv_orc.write.orc("s3a://mybucket/ten_million_orc2.orc")end_time = time.time()print(f"Time taken to write as ORC: {end_time - start_time} seconds")
```

It took me ***12.94 seconds*** to complete the task. Another point of interest is the size of the data written to the MinIO bucket. In the `ten_million_orc2.orc` folder, you'll find several partition files, each of a consistent size. Every partition ORC file is about ***22.3 MiB***, and there are 16 files in total.

![ORC partition files (Image by author)](https://assets.insightmediagroup.io/media/wp-content/uploads/2024/01/1QF4lLb_lAkwW6TfDQomEbg.png "ORC partition files (Image by author)")

Comparing this to Parquet, each Parquet partition file is around ***26.8 MiB***, also totaling 16 files. This shows that ORC indeed offers better compression than Parquet.

Next, we'll test how ORC handles an aggregation query. We're using the same query for all file formats to keep our benchmarking fair.

python

```
# Perform aggregation using ORC datadf_orc = spark.read.orc("s3a://mybucket/ten_million_orc2.orc")start_time = time.time()df_orc .select("str_col_5","str_col_7","int_col_2") .groupBy("str_col_5","str_col_7") .count() .orderBy("count") .limit(1) .show(truncate = False)end_time = time.time()print(f"Time taken for query: {end_time - start_time} seconds")+---------+---------+-----+|str_col_5|str_col_7|count|+---------+---------+-----+|1        |2906292  |1    |+---------+---------+-----+
```

The ORC query finished in ***13.44 seconds***, a tad longer than Parquet's time. With ORC checked off our list, let's move on to experimenting with Avro.

### Working with Avro {#working-with-avro}

Avro is a row-based file format with its own unique strengths. While it doesn't compress data as efficiently as Parquet or ORC, it makes up for this with a faster writing speed.

What really sets Avro apart is its excellent schema evolution capabilities. It handles changes like added, removed, or altered fields with ease, making it a go-to choice for scenarios where data structures evolve over time.

> Avro is particularly well-suited for workloads that involve a lot of data writing.

Now, let's check out how Avro does with writing data.

python

```
# Write data as Avrostart_time = time.time()df_csv_avro.write.format("avro").save("s3a://mybucket/ten_million_avro2.avro")end_time = time.time()print(f"Time taken to write as Avro: {end_time - start_time} seconds")
```

It took me ***12.81 seconds***, which is actually quicker than both Parquet and ORC. Next, we'll look at Avro's performance with an aggregation query.

python

```
# Perform aggregation using Avro datadf_avro = spark.read.format("avro").load("s3a://mybucket/ten_million_avro2.avro")start_time = time.time()df_avro .select("str_col_5","str_col_7","int_col_2") .groupBy("str_col_5","str_col_7") .count() .orderBy("count") .limit(1) .show(truncate = False)end_time = time.time()print(f"Time taken for query: {end_time - start_time} seconds")+---------+---------+-----+|str_col_5|str_col_7|count|+---------+---------+-----+|1        |6429997  |1    |+---------+---------+-----+
```

This query took about ***15.42 seconds***. So, when it comes to querying, Parquet and ORC are ahead in terms of speed. Alright, it's time to explore our final and newest file format - Delta Lake.

### Working with Delta Lake {#working-with-delta-lake}

Delta Lake is a new star in the big data file format universe, closely related to Parquet in terms of storage size - it's like Parquet but with some extra features.

When writing data, Delta Lake takes a bit longer than Parquet, mostly because of its `_delta_log` folder, which is key to its advanced capabilities. These capabilities include ACID compliance for reliable transactions, time travel for accessing historical data, and small file compaction to keep things tidy.

While it's a newcomer in the big data scene, Delta Lake has quickly become a favorite on cloud platforms that run Spark, outpacing its use in on-premises systems.

Let's move on to testing Delta Lake's performance, starting with a data writing test.

python

```
# Write data as Deltastart_time = time.time()df_csv_delta.write.format("delta").save("s3a://mybucket/ten_million_delta2.delta")end_time = time.time()print(f"Time taken to write as Delta Lake: {end_time - start_time} seconds")
```

The write operation took ***17.78 seconds***, which is a bit longer than the other file formats we've looked at. A neat thing to notice is that in the `ten_million_delta2.delta` folder, each partition file is actually a Parquet file, similar in size to what we observed with Parquet. Plus, there's the `_delta_log` folder.

![Writing data as Delta Lake (Image by author)](https://assets.insightmediagroup.io/media/wp-content/uploads/2024/01/1mHUFOG0pK_N1Mu6f0o8-0Q.png "Writing data as Delta Lake (Image by author)")

The `_delta_log` folder in the Delta Lake file format plays a critical role in how Delta Lake manages and maintains data integrity and versioning. It's a key component that sets Delta Lake apart from other big data file formats. Here's a simple breakdown of its function:

1. **Transaction Log**: The `_delta_log` folder contains a transaction log that records every change made to the data in the Delta table. This log is a series of JSON files that detail the additions, deletions, and modifications to the data. It acts like a comprehensive diary of all the data transactions.
2. **ACID Compliance**: This log enables ACID (Atomicity, Consistency, Isolation, Durability) compliance. Every transaction in Delta Lake, like writing new data or modifying existing data, is atomic and consistent, ensuring data integrity and reliability.
3. **Time Travel and Auditing**: The transaction log allows for "time travel", which means you can easily view and restore earlier versions of the data. This is extremely useful for data recovery, auditing, and understanding how data has evolved over time.
4. **Schema Enforcement and Evolution**: The `_delta_log` also keeps track of the schema (structure) of the data. It enforces the schema during data writes and allows for safe evolution of the schema over time without corrupting the data.
5. **Concurrency and Merge Operations**: It manages concurrent reads and writes, ensuring that multiple users can access and modify the data at the same time without conflicts. This makes it ideal for complex operations like merge, update, and delete.

In summary, the `_delta_log` folder is the brain behind Delta Lake's advanced data management features, offering robust transaction logging, version control, and reliability enhancements that are not typically available in simpler file formats like Parquet or ORC.

Now, it's time to see how Delta Lake fares with an aggregation query.

python

```
# Perform aggregation using Delta datadf_delta = spark.read.format("delta").load("s3a://mybucket/ten_million_delta2.delta")start_time = time.time()df_delta .select("str_col_5","str_col_7","int_col_2") .groupBy("str_col_5","str_col_7") .count() .orderBy("count") .limit(1) .show(truncate = False)end_time = time.time()print(f"Time taken for query: {end_time - start_time} seconds")+---------+---------+-----+|str_col_5|str_col_7|count|+---------+---------+-----+|1        |2906292  |1    |+---------+---------+-----+
```

This query finished in about ***15.51 seconds***. While this is a tad slower compared to Parquet and ORC, it's pretty close. It suggests that Delta Lake's performance in real-world scenarios is quite similar to that of Parquet.

Awesome! We've wrapped up all our experiments. Let's recap our findings in the next section.

### When to use which file format? {#when-to-use-which-file-format}

We've wrapped up our testing, so let's bring all our findings together. For data writing, Avro takes the top spot. That's really what it's best at in practical scenarios.

When it comes to reading and running aggregation queries, Parquet leads the pack. However, this doesn't mean ORC and Delta Lake fall short. As columnar file formats, they perform admirably in most situations.

![Performance comparison (Image by author)](https://assets.insightmediagroup.io/media/wp-content/uploads/2024/01/1jNqCcwmKjzADkJi80JMOGw.png "Performance comparison (Image by author)")

Here's a quick rundown:

- Choose ORC for the best compression, especially if you're using Hive and Pig for analytical tasks.
- Working with Spark? Parquet and Delta Lake are your go-to choices.
- For scenarios with lots of data writing, like landing zone areas, Avro is the best fit.

And that's a wrap on this tutorial!

---

### Conclusion {#conclusion}

In this guide, we put the four big hitters of big data file formats - Parquet, ORC, Avro, and Delta Lake - to the test. We checked how they handle writing data and then how they manage an aggregation query. This helped us see each format's overall performance and how they differ in terms of data size. We also dove into what makes Delta Lake unique, especially its `_delta_log` folder.

> You can find the full notebook on [GitHub](https://github.com/sarthak-sarbahi/big-data-file-formats/blob/main/big_data_file_formats.ipynb).

I sincerely hope this guide was beneficial for you. Should you have any questions, please don't hesitate to drop them in the comments below.

### References {#references}
