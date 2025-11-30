# features.py
import pandas as pd
from datetime import timedelta

def create_customer_features(df):
    """
    Create customer-level features (RFM + behavior features).
    """
    ref_date = df['InvoiceDate'].max() + timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                              # Frequency
        'TotalPrice': 'sum'                                  # Monetary
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Add churn label: if no purchase in last 90 days → churn
    rfm['Churn'] = (rfm['Recency'] > 90).astype(int)

    return rfm

if __name__ == "__main__":
    from etl import load_and_clean_data
    df = etl.load_and_clean_data()
    features = create_customer_features(df)
    print(features.head())