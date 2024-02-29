# COVID-19-Data-Analysis-With-Pyspark
# Introduction
The goal of this project is the loading, preprocessing and exploratory data analysis of SARS-Cov-2 data on a global scale using PySpark, Apache Spark's Python API. The dataset was curated and is maintained by Our World in Data (OWiD). Typically, Spark is utilized when dealing with much larger datasets than the one seen here. In fact, all the tasks performed below could be performed using Pandas, in a somewhat cleaner and more familiar fashion. Nonetheless, the purpose of this notebook is not the analysis itself (especially since Covid-19 data have been extensively analyzed before), but the introduction of the reader to PySpark. In this sense, the dataset is more of a workhorse serving an educational purpose, so Pandas will be used only when absolutely necessary.

# 1. Data Loading & Overview

# 2. Data Preprocessing

 2.1. Handling Missing Values
 
 2.2. Outlier Detection
 
 2.3. Duplicate Entries

# 3. Exploratory Data Analysis

 3.1. Evolution of top countries with respect to mortality
 
 3.2. Evolution of top countries with respect to total cases per million
 
 3.3. Hospitalized Patients and ICU Admissions
 
 3.4. Geographic Heatmap of Total Cases
 
 3.5. Geographic Correlation of Excess Mortality
 
 3.6. Reproduction Rate on the Continent Level
 
 3.7. Correlation between different features
 
 3.8. Covid and general health conditions on the country level
 
 3.9. k-Means Clustering
