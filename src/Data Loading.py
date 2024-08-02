import warnings
warnings.filterwarnings("ignore")

!pip install pyspark

file = '../input/our-world-in-data-covid19-dataset/owid-covid-data.csv'

#After initializing a PySpark session, the data are transformed into a PySpark DataFrame.

import pyspark
from pyspark.sql import SparkSession, SQLContext

spark = SparkSession.builder.appName("Covid Data Mining").config('spark.sql.debug.maxToStringFields', 2000).getOrCreate()
full_df = spark.read.csv(file, header=True, inferSchema=True)

#Let's first identify the total number of samples, as well as the number of each sample's features.

print(f"The total number of samples is {full_df.count()}, with each sample corresponding to {len(full_df.columns)} features.")

#In order to identify each feature, as well as its type, `full_df.dtypes` can be used. Alternatively, they are available as part of the Schema's information via the following:

full_df.printSchema()

full_df.select("iso_code","location","continent","date","tests_units").show(5)

from pyspark.sql import functions as F

miss_vals = full_df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in full_df.columns]).collect()[0].asDict()
miss_vals = dict(sorted(miss_vals.items(), reverse=True, key=lambda item: item[1]))

import pandas as pd

pd.DataFrame.from_records([miss_vals])

