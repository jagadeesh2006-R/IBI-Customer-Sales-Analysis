# IBI-Customer-Sales-Analysis

# IBI Internship Project – Customer & Sales Data Analysis

**By:** Jagadeeswar R.  
**Internship:** Shuruwat Foundation – Data Science Track

## Objective

This project aims to perform in-depth analysis on customer and sales data for:
- Identifying customer segments
- Forecasting future sales trends
- Predicting customer churn
- Recommending targeted business strategies

##  Project Structure

IBI-Customer-Sales-Analysis/
+-- data/                          # Dataset (.csv or link)
+-- notebooks/                    # Jupyter Notebook with full project
+-- reports/                      # Final report (PDF/Word)
+-- models/                       # Trained ML models (.pkl)
+-- images/                       # Visualizations (charts/plots)
+-- README.md                     # Project overview


##  How to Run This Project

### 1. Clone or Download the Repo

git clone https://github.com/jagadeesh2006-R/IBI-Customer-Sales-Analysis.git
cd IBI-Customer-Sales-Analysis

### 2. Install Required Packages

pip install -r requirements.txt


> If `requirements.txt` is not present, manually install:


pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost prophet tensorflow mlxtend


### 3. Run the Notebook

Open the notebook:

notebooks/ibi_customer_sales_analysis.ipynb

##  Models

This project includes:

* `churn_random_forest.pkl`: Predicts if a customer will churn
* `customer_segmentation_gmm.pkl`: Segments customers using GMM

Use `model_saver.py` in the `models/` folder to regenerate these.

##  Visualizations

Visuals include:

* Monthly Sales Trend
* Product Category Sales
* Recency vs Monetary (RFM)

Located in `images/` and used in the report.


##  Final Report

Find the project summary in:

* `reports/IBI_Project_Report_Jagadeeswar.docx`

##  Dataset Source

You can use any publicly available Kaggle retail dataset. A link or file path is specified in the notebook.



##  Status

Project submitted as part of IBI Internship – Advanced Customer & Sales Analytics.
All deliverables are complete and reproducible.
