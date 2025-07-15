# model_saver.py
# Save trained models into the models/ directory

import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Simulate sample data similar to RFM structure
np.random.seed(42)
n_customers = 200
rfm = pd.DataFrame({
    'CustomerID': np.arange(1, n_customers+1),
    'Frequency': np.random.poisson(5, n_customers),
    'Monetary': np.random.randint(100, 10000, n_customers),
    'Recency': np.random.randint(1, 120, n_customers)
})

# Loyalty Score based on Frequency and Recency
rfm['LoyaltyScore'] = (rfm['Frequency'] / rfm['Frequency'].max()) * 50 + \
                      (1 - rfm['Recency'] / rfm['Recency'].max()) * 50

# Define churn (Recency > 90)
rfm['Churned'] = rfm['Recency'] > 90

# 🎯 Churn Model: Random Forest
X_churn = rfm[['Frequency', 'Monetary', 'Recency', 'LoyaltyScore']]
y_churn = rfm['Churned']
X_train, X_test, y_train, y_test = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 👥 Segmentation Model: GMM
X_seg = rfm[['Frequency', 'Monetary', 'Recency']]
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_seg)

os.makedirs("models", exist_ok=True)
with open("models/churn_random_forest.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("models/customer_segmentation_gmm.pkl", "wb") as f:
    pickle.dump(gmm, f)
