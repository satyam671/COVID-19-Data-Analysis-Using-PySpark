# COVID-19 Data Analysis Using PySpark
![1_y2SsUh89eQShLdIipSO4eg](https://github.com/user-attachments/assets/0a8b8416-ba41-4d6a-a588-129f4f6026cd)

## Decoding the Pandemic: A Big Data Approach

This project leverages the power of PySpark to analyze global COVID-19 data, providing insights into the pandemic's progression and impact across different countries and continents.

## 🦠 About the Project

In this data engineering endeavor, we dive deep into the SARS-CoV-2 dataset curated by Our World in Data (OWiD). While the dataset size doesn't necessarily warrant big data tools, this project serves as an educational gateway into the world of distributed computing with PySpark.

## 🛠 Technologies Used

- PySpark: Apache Spark's Python API for distributed data processing
- SQL: For query-based data manipulation
- Python: For additional data handling and visualization

## 🔍 Project Structure: What We Explore -

1. **Data Loading & Overview**: Initial steps in big data processing
2. **Data Preprocessing**: 
   - Handling missing values in large datasets
   - Outlier detection at scale
   - Efficient duplicate entry management
3. **Exploratory Data Analysis**:
   - Mortality trends across countries
   - Case rates per million population
   - Hospital and ICU admission patterns
   - Geographic visualizations of case distribution
   - Excess mortality correlations
   - Continent-level reproduction rates
   - Feature correlation analysis
   - Health condition impacts on COVID-19 spread
   - K-means clustering for country grouping

## 🎯 Project Goals

- Demonstrate proficiency in PySpark for big data analysis
- Showcase data engineering skills in cleaning and preprocessing large datasets
- Provide valuable insights into the global COVID-19 situation
- Explore the potential of distributed computing in epidemiological research

## 🚀 Getting Started

Instructions on how to set up and run the project:

### Prerequisites
1. **Python** (version 3.7+ recommended)
2. **Apache Spark** (version 3.0+)
3. **PySpark** (version compatible with the installed Apache Spark)
4. **Jupyter Notebook**

### Installation
1. **Install Python**: Follow the instructions on the [official Python website](https://www.python.org/downloads/).
2. **Install Apache Spark**: Follow the instructions on the [official Apache Spark website](https://spark.apache.org/downloads.html).
3. **Install PySpark**: Use the following command:
    ```bash
    pip install pyspark
    ```
4. **Install Jupyter Notebook**: Use the following command:
    ```bash
    pip install notebook
    ```

### Running the Project
1. **Clone the Repository**: 
    ```bash
    git clone https://github.com/satyam671/COVID-19-Data-Analysis-Using-PySpark.git
    ```
2. **Navigate to the Project Directory**:
    ```bash
    cd <project_directory>
    ```
3. **Start Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
4. **Open the Notebook**: Locate and open the `analyzing-covid-19-data-with-pyspark.ipynb` file in Jupyter Notebook.
5. **Run the Cells**: Execute the notebook cells sequentially for analysis.


## 📊 Results and Insights

### Key Findings
1. **Global Trends**: Analysis of COVID-19 case trends globally, identifying peak periods and significant changes over time.
2. **Country-Specific Insights**: Detailed exploration of COVID-19 impacts on specific countries, including case numbers, death rates, and recovery rates.
3. **Vaccination Analysis**: Insights into vaccination rollouts and their correlation with case and death rates.
4. **Human Development Index (HDI) vs. Reproduction Rate**: Scatterplots and analysis showing the relationship between HDI and COVID-19 reproduction rates.

### Visualizations
- Time series plots of COVID-19 cases and deaths.
- Geographical distribution maps of COVID-19 spread.
- Scatterplots of various metrics such as HDI vs. reproduction rate.

For a detailed overview of the findings, please take a look at the results section in the Jupyter Notebook.

## 🛠️ Issues Page

### Known Issues
1. **Data Quality**: Some datasets may have missing or inconsistent data points. Ensure proper data cleaning steps are included in your workflow.
2. **Performance**: Large datasets may require substantial memory and processing power. Optimize PySpark jobs and consider using a cluster if necessary.
3. **Dependencies**: Compatibility issues between different versions of PySpark, Spark, and Python. Ensure all dependencies are compatible.

### Future Improvements
1. **Enhanced Visualizations**: Integrate interactive visualizations using libraries like Plotly.
2. **Real-Time Analysis**: Implement real-time data analysis for more timely insights.
3. **Extended Metrics**: Include more health and economic metrics to provide a comprehensive analysis of COVID-19 impacts.

### 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/satyam671/COVID-19-Data-Analysis-Using-PySpark/issues) if you want to contribute.

### 📝 License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](https://github.com/satyam671/COVID-19-Data-Analysis-Using-PySpark/blob/main/LICENSE) file for details.

### 🙏 Acknowledgements

1. **Our World in Data (OWiD) for providing the COVID-19 dataset**
2. **Apache Spark community for PySpark**

---

This project is part of a data engineering portfolio, demonstrating skills in big data processing, analysis, and visualization using industry-standard tools.
