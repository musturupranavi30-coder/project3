# etl.py
import pandas as pd

def load_and_clean_data():
    df = pd.read_csv("OnlineRetail.csv", encoding='unicode_escape')

    # Example cleaning steps
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    return df