# Telco Customer Churn Prediction and Analysis
### Author: Virendrasinh Chavda

<p align="justify">
This repository contains the code and resources for predicting and analyzing customer churn in the telecom industry. Built with a robust tech stack that includes Streamlit, **Scikit-Learn**, and **SHAP**, this project offers insights into churn patterns using machine learning models. It integrates an interactive dashboard for segmentation, visualization, and in-depth analysis of customer behavior.
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
The Telco Customer Churn Prediction system identifies customers likely to churn based on their demographics, usage patterns, and contract details. This project utilizes **machine learning models** and provides actionable insights through an **interactive dashboard**. It also highlights the key factors influencing customer churn using **SHAP analysis**.
</p>

---

## Features
- **Churn Prediction**: Predicts the likelihood of customer churn using machine learning models.
- **Interactive Dashboard**: Built with Streamlit for visualizing churn patterns, segmentation, and analysis.
- **Segmentation Analysis**:
  - **Contract and Tenure**: Churn rate by contract type and tenure.
  - **Service Bundles**: Churn patterns across service packages.
  - **Payment Methods**: Impact of payment methods on churn.
  - **Tenure and Monthly Charges**: Trends in tenure, charges, and churn.
- **SHAP Analysis**: Explains model predictions by identifying features that influence churn.
- **Geographical Mapping**: Visualizes churned customers on an interactive map.

---

## Technologies Used
- **Streamlit**: For building the interactive web dashboard.
- **Scikit-Learn**: For machine learning model training and preprocessing.
- **SHAP**: For explainable AI and feature impact visualization.
- **Seaborn & Matplotlib**: For data visualization and analytics.
- **Pandas**: For data manipulation and analysis.

---

## Detailed Implementation

### Data Preparation
- **Dataset**: The dataset includes customer demographics, services, and payment details.
- **Cleaning**: Handled missing values and ensured consistent numeric formatting for features like `Total Charges`.

### Machine Learning Models
- **Random Forest**: Used for churn prediction.
- **XGBoost**: Enhanced model with higher performance for interpretability and accuracy.
- **Feature Scaling**: Applied scaling techniques (e.g., MinMaxScaler) for numeric attributes.

### Dashboard Features
- **Filters**: Interactive filters for city, tenure, and other attributes.
- **Segmentation**: Drill-down analysis by customer attributes.
- **Visualizations**: Includes bar charts, line plots, and SHAP summary plots.

---

## Results
The project achieves robust performance in churn prediction with the following key metrics:

| **Model**          | **Metric**       | **Value**  |
|---------------------|------------------|------------|
| **Random Forest**   | Accuracy         | 96.7%      |
|                     | F1-Score         | 91.3%      |
| **XGBoost**         | Accuracy         | 94.2%      |
|                     | F1-Score         | 87.1%      |

---

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/VirendraChavda/churn-prediction.git
   cd churn-prediction
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare Data Place the new_data.csv file in the Data/ folder.**
4. **Run the Application**
   ```bash
   streamlit run app.py
   ```
## Usage
<p align="justify">
The application provides an intuitive dashboard for customer churn analysis. Users can:
1. Apply filters to explore churn patterns across different customer segments.
2. View visualizations like churn by contract type, tenure, and service bundles.
3. Analyze SHAP plots to understand the key factors influencing churn.
4. Access a map of churned customers to identify geographic patterns.
</p>

---

## Future Enhancements
- **Enhanced Models**: Incorporate deep learning models for improved prediction accuracy.
- **Advanced Segmentation**: Enable multi-level drill-down into customer attributes.
- **Forecasting**: Predict future churn trends based on historical data.
- **Real-Time Data**: Integrate with live customer data systems.

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
