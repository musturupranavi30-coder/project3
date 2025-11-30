import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from etl import load_and_clean_data
from features import create_customer_features

st.set_page_config(page_title="Customer Churn & Sales Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction and Sales Dashboard")

@st.cache_data
def load_data():
    return load_and_clean_data()

@st.cache_data
def compute_features(df):
    return create_customer_features(df)

# Load data
df = load_data()
features = compute_features(df)

# --- Sales Trends ---
st.header("💰 Sales Trends Over Time")

df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
monthly_sales = df.groupby('Month')['TotalPrice'].sum().reset_index()

fig_sales = px.line(monthly_sales, x='Month', y='TotalPrice', title="Monthly Sales Trend")
st.plotly_chart(fig_sales, use_container_width=True)

# --- Top Products ---
st.header("🏆 Top Products")
top_products = df.groupby('Description')['TotalPrice'].sum().nlargest(10).reset_index()
fig_products = px.bar(top_products, x='Description', y='TotalPrice', title="Top 10 Products")
st.plotly_chart(fig_products, use_container_width=True)

# --- Churn Prediction ---
st.header("🔮 Customer Churn Prediction")

try:
    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except:
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

customer_id = st.selectbox("Select Customer ID", features['CustomerID'])
row = features[features['CustomerID'] == customer_id][['Recency', 'Frequency', 'Monetary']]

scaled = scaler.transform(row)
prob = model.predict_proba(scaled)[0][1]

st.metric("Churn Probability", f"{prob:.2f}")