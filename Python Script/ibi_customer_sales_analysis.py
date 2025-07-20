# IBI Internship Project – Customer & Sales Data Analysis
# Author: Jagadeeswar R.

# ===============================
# 1. SETUP & LIBRARY IMPORTS
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from statsmodels.tsa.seasonal import STL
from prophet import Prophet
import tensorflow as tf
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")
import re
import csv # Import the csv module
import os # Import the os module

# ===============================
# 2. LOAD DATA
# ===============================
# Load the data using the csv module due to complex formatting issues
data = []
processed_lines = 0
skipped_lines = 0
with open('Retail_Transaction_Dataset.csv', 'r', encoding='utf-8') as f: # Specify encoding
    reader = csv.reader(f)
    next(reader) # Skip the first line
    for i, row in enumerate(reader):
        if not row:
            skipped_lines += 1
            continue # Skip empty rows

        # Expected columns: CustomerID, TransactionType, Quantity, Price, Date, PaymentMethod, ShippingAddress, ProductCategory, Discount, OtherValue (10 columns)
        if len(row) == 10:
            data.append(row)
            processed_lines += 1
        else:
            skipped_lines += 1
            # Optional: print problematic rows
            # print(f"Skipping row {i+2} (incorrect number of fields: {len(row)}): {row}")


df = pd.DataFrame(data, columns=['CustomerID', 'TransactionType', 'Quantity', 'Price', 'Date', 'PaymentMethod', 'ShippingAddress', 'ProductCategory', 'Discount', 'OtherValue'])

print(f"Processed {processed_lines} lines, skipped {skipped_lines} lines.")
print("Initial Dataset Info:")
print(df.info())
print(df.head())

# Print column names to inspect for issues
print("\nDataFrame columns:")
print(df.columns)

# ===============================
# 3. DATA PREPROCESSING
# ===============================
# Convert relevant columns to numeric, coercing errors
for col in ['Quantity', 'Price', 'Discount', 'OtherValue']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3.1 Missing Values Imputation
imp_mice = IterativeImputer()
num_cols = df.select_dtypes(include=np.number).columns
df_imputed = pd.DataFrame(imp_mice.fit_transform(df[num_cols]), columns=num_cols)
df[num_cols] = df_imputed

# 3.2 Outlier Handling (Quantity)
Q1 = df['Quantity'].quantile(0.25)
Q3 = df['Quantity'].quantile(0.75)
IQR = Q3 - Q1
df['Quantity'] = df['Quantity'].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

# 3.3 Normalization
scaler = MinMaxScaler()
# Assuming 'Age' is a valid column name based on earlier comments, if not, replace with actual numerical columns
# Only normalize numerical columns that are not identifiers
numerical_cols_to_normalize = ['Price', 'Discount', 'Quantity', 'OtherValue'] # Added Quantity and OtherValue
df[numerical_cols_to_normalize] = scaler.fit_transform(df[numerical_cols_to_normalize])


# ===============================
# 4. FEATURE ENGINEERING
# ===============================
df['TotalPrice'] = df['Quantity'] * df['Price'] * (1 - df['Discount'])
df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Convert with errors='coerce'
df['Recency'] = (pd.Timestamp('2025-07-01') - df['Date']).dt.days

rfm = df.groupby('CustomerID').agg({
    'Date': 'nunique',
    'TotalPrice': 'sum',
    'Recency': 'min'
}).rename(columns={'Date': 'Frequency', 'TotalPrice': 'Monetary'})

rfm['CLV'] = rfm['Monetary']
# Handle potential division by zero or very small numbers in Recency max
recency_max = rfm['Recency'].max()
epsilon = 1e-6  # Small value to avoid division by zero
rfm['LoyaltyScore'] = (rfm['Frequency'] / rfm['Frequency'].max()) * 50 + \
                      (1 - rfm['Recency'] / (recency_max if recency_max > epsilon else epsilon)) * 50

# ===============================
# 5. EXPLORATORY DATA ANALYSIS
# ===============================
plt.figure(figsize=(10, 4))
# Drop rows with NaT in 'Date' before resampling
monthly_sales = df.dropna(subset=['Date']).resample('M', on='Date')['TotalPrice'].sum()
sns.lineplot(data=monthly_sales)
plt.title('Monthly Sales Trend')
plt.ylabel('Total Sales')
plt.xlabel('Month')
plt.tight_layout()
# Create the 'images' directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig("images/monthly_sales_trend.png")

# ===============================
# 6. CUSTOMER SEGMENTATION
# ===============================
# Check for and handle infinite values before scaling
rfm.replace([np.inf, -np.inf], np.nan, inplace=True)
rfm.dropna(subset=['Frequency', 'Monetary', 'Recency'], inplace=True) # Drop rows with NaNs in these columns after handling inf

segment_data = scaler.fit_transform(rfm[['Frequency', 'Monetary', 'Recency']])
gmm = GaussianMixture(n_components=4, random_state=42)
rfm['Segment_GMM'] = gmm.fit_predict(segment_data)

# ===============================
# 7. SALES FORECASTING
# ===============================
# Drop rows with NaT in 'Date' before resampling
sales_ts = df.dropna(subset=['Date']).set_index('Date').resample('D')['TotalPrice'].sum().reset_index()
sales_ts.columns = ['ds', 'y']

model = Prophet()
model.fit(sales_ts)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

fig1 = model.plot(forecast)
fig1.savefig("sales_forecast.png")

# ===============================
# 8. CHURN PREDICTION
# ===============================
rfm['Churned'] = rfm['Recency'] > 90

X = rfm[['Frequency', 'Monetary', 'Recency', 'LoyaltyScore']]
y = rfm['Churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

print("Random Forest Accuracy (Churn):", clf.score(X_test, y_test))

# ===============================
# 9. MARKET BASKET ANALYSIS
# ===============================
# Using 'CustomerID' as the transaction identifier
basket = df.groupby(['CustomerID', 'ProductCategory'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

print("Top Association Rules:")
print(rules.sort_values('confidence', ascending=False).head())
