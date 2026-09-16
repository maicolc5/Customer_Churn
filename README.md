# Customer Churn Prediction - AdventureWorks2025

## Problem Definition

**Objective**: Predict if a customer will make another purchase within the next 6 months based on their historical transaction data.

**Business Value**: Identify at-risk customers to implement retention strategies and increase revenue.

## Project Structure

```
Customer_Churn/
├── 01_data_extraction.py      # Extract data from SQL Server
├── 02_eda.ipynb               # Exploratory Data Analysis
├── 03_preprocessing.py        # Data cleaning and preprocessing
├── 04_feature_engineering.py  # Feature creation
├── 05_model_selection.py      # Algorithm comparison
├── 06_training.py             # Model training
├── 07_evaluation.py           # Model evaluation
├── 08_conclusion.ipynb        # Final analysis and insights
├── data/                      # Raw and processed data
├── models/                    # Saved models
└── requirements.txt           # Dependencies
```

## Dataset

- **Source**: AdventureWorks2025 SQL Server Database
- **Tables**: SalesOrderHeader, SalesOrderDetail, Customer, Product
- **Target**: `will_buy_again` (1 = returns, 0 = churned)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract data (requires SQL Server connection)
python 01_data_extraction.py

# 3. Run the pipeline in order
# Or run all at once
python main.py
```

## Results

See `08_conclusion.ipynb` for final model performance and business recommendations.
