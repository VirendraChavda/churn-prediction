# Telco Customer Churn Prediction and Analysis
### Author: Virendrasinh Chavda

<p align="justify">
This repository contains the code and resources for predicting and analyzing customer churn in the telecom industry. Built with a robust tech stack that includes Streamlit, <strong>Scikit-Learn</strong>, and <strong>SHAP</strong>, this project offers insights into churn patterns using machine learning models. It integrates an interactive dashboard for segmentation, visualization, and in-depth analysis of customer behavior.
</p>

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Detailed Implementation](#detailed-implementation)
5. [Results](#results)
6. [Setup and Installation](#setup-and-installation)
7. [Usage](#usage)
8. [Future Enhancements](#future-enhancements)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview
<p align="justify">
The Telco Customer Churn Prediction system identifies customers likely to churn based on their demographics, usage patterns, and contract details. This project utilizes <strong>machine learning models</strong> and provides actionable insights through an <strong>interactive dashboard</strong>. It also highlights the key factors influencing customer churn using <strong>SHAP analysis</strong>.
</p>

---

## Features
- <strong>Churn Prediction</strong>: Predicts the likelihood of customer churn using machine learning models.
- <strong>Interactive Dashboard</strong>: Built with Streamlit for visualizing churn patterns, segmentation, and analysis.
- <strong>Segmentation Analysis</strong>:
  - <strong>Contract and Tenure</strong>: Churn rate by contract type and tenure.
  - <strong>Service Bundles</strong>: Churn patterns across service packages.
  - <strong>Payment Methods</strong>: Impact of payment methods on churn.
  - <strong>Tenure and Monthly Charges</strong>: Trends in tenure, charges, and churn.
- <strong>SHAP Analysis</strong>: Explains model predictions by identifying features that influence churn.
- <strong>Geographical Mapping</strong>: Visualizes churned customers on an interactive map.

---

## Technologies Used
- <strong>Streamlit</strong>: For building the interactive web dashboard.
- <strong>Scikit-Learn</strong>: For machine learning model training and preprocessing.
- <strong>SHAP</strong>: For explainable AI and feature impact visualization.
- <strong>Seaborn & Matplotlib</strong>: For data visualization and analytics.
- <strong>Pandas</strong>: For data manipulation and analysis.

---

## Detailed Implementation

### Data Preparation
- <strong>Dataset</strong>: The dataset includes customer demographics, services, and payment details.
- <strong>Cleaning</strong>: Handled missing values and ensured consistent numeric formatting for features like `Total Charges`.

### Machine Learning Models
- <strong>Random Forest</strong>: Used for churn prediction.
- <strong>XGBoost</strong>: Enhanced model with higher performance for interpretability and accuracy.
- <strong>Feature Scaling</strong>: Applied scaling techniques (e.g., MinMaxScaler) for numeric attributes.

### Dashboard Features
- <strong>Filters</strong>: Interactive filters for city, tenure, and other attributes.
- <strong>Segmentation</strong>: Drill-down analysis by customer attributes.
- <strong>Visualizations</strong>: Includes bar charts, line plots, and SHAP summary plots.

---

## Results
The project achieves robust performance in churn prediction with the following key metrics:

| <strong>Model<strong>          | <strong>Metric<strong>       | <strong>Value<strong>  |
|---------------------|------------------|------------|
| <strong>Random Forest<strong>   | Accuracy         | 96.7%      |
|                     | F1-Score         | 91.3%      |
| <strong>XGBoost<strong>         | Accuracy         | 94.2%      |
|                     | F1-Score         | 87.1%      |

---

## Setup and Installation

1. <strong>Clone the Repository</strong>
   ```bash
   git clone https://github.com/VirendraChavda/churn-prediction.git
   cd churn-prediction
   ```
2. <strong>Install Dependencies</strong>
   ```bash
   pip install -r requirements.txt
   ```
3. <strong>Prepare Data Place the new_data.csv file in the Data/ folder.<strong>
4. <strong>Run the Application</strong>
   ```bash
   streamlit run app.py
   ```
## Usage
<p align="justify">
The application provides an intuitive dashboard for customer churn analysis. Users can:
</p>

  1. Apply filters to explore churn patterns across different customer segments.
  2. View visualizations like churn by contract type, tenure, and service bundles.
  3. Analyze SHAP plots to understand the key factors influencing churn.
  4. Access a map of churned customers to identify geographic patterns.


---

## Future Enhancements
- <strong>Enhanced Models</strong>: Incorporate deep learning models for improved prediction accuracy.
- <strong>Advanced Segmentation</strong>: Enable multi-level drill-down into customer attributes.
- <strong>Forecasting</strong>: Predict future churn trends based on historical data.
- <strong>Real-Time Data</strong>: Integrate with live customer data systems.

---

## Contributing
<p align="justify">
We welcome contributions to enhance this project! Please submit issues or pull requests. For significant changes, discuss your ideas in an issue before implementation.
</p>

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```bash
This `README.md` provides a comprehensive description of your project, including its features, technologies, setup instructions, and results. Let me know if you'd like any modifications or additional details!
```
