import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import os
import json

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Forecasting Insights", layout="wide")
st.title("AI-Enhanced Forecasting Tool")
st.markdown("*Generate narrative insights from financial data using GPT-4o*")

# ------------------------------------------------------------------
# OPENAI SETUP
# ------------------------------------------------------------------
openai_api_key = st.sidebar.text_input("OpenAI API Key")
if not openai_api_key:
    st.info("Enter your OpenAI API key in the sidebar to enable AI insights.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv", encoding="cp1252")
    df.columns = df.columns.str.strip().str.upper()
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")
    df["REVENUE"] = pd.to_numeric(df["QUANTITYORDERED"], errors="coerce") * pd.to_numeric(df["PRICEEACH"], errors="coerce")
    df = df.dropna(subset=["ORDERDATE", "REVENUE"])
    return df

df = load_data()

# ------------------------------------------------------------------
# FILTERS
# ------------------------------------------------------------------
st.sidebar.header("Filters")
product_line = st.sidebar.multiselect("Product Line", df["PRODUCTLINE"].unique(), default=df["PRODUCTLINE"].unique()[:2])
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df["ORDERDATE"].min().date(), df["ORDERDATE"].max().date()),
    min_value=df["ORDERDATE"].min().date(),
    max_value=df["ORDERDATE"].max().date()
)

mask = (
    (df["ORDERDATE"].dt.date >= date_range[0]) &
    (df["ORDERDATE"].dt.date <= date_range[1]) &
    (df["PRODUCTLINE"].isin(product_line))
)
filtered = df[mask].copy()

# ------------------------------------------------------------------
# KPI + CHART
# ------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
total_rev = filtered["REVENUE"].sum()
col1.metric("Total Revenue", f"${total_rev:,.0f}")
col2.metric("Orders", len(filtered))
col3.metric("Avg Order", f"${filtered['REVENUE'].mean():,.0f}")

monthly = filtered.resample("ME", on="ORDERDATE")["REVENUE"].sum().reset_index()
fig = px.line(monthly, x="ORDERDATE", y="REVENUE", title="Monthly Revenue Trend", markers=True)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# AI INSIGHTS (GPT-4o)
# ------------------------------------------------------------------
if st.button("Generate AI Executive Summary", type="primary"):
    with st.spinner("Asking GPT-4o for insights..."):
        # Prepare data summary
        stats = {
            "total_revenue": round(total_rev, 0),
            "total_orders": len(filtered),
            "avg_order_value": round(filtered["REVENUE"].mean(), 2),
            "top_product": filtered.groupby("PRODUCTLINE")["REVENUE"].sum().idxmax(),
            "peak_month": monthly.loc[monthly["REVENUE"].idxmax(), "ORDERDATE"].strftime("%B %Y"),
            "growth_trend": "up" if monthly["REVENUE"].iloc[-1] > monthly["REVENUE"].iloc[0] else "down"
        }

        prompt = f"""
You are a senior financial analyst. Write a concise, executive-level summary (3-4 sentences) of the following sales performance:

- Total Revenue: ${stats['total_revenue']:,}
- Total Orders: {stats['total_orders']:,}
- Avg Order Value: ${stats['avg_order_value']:,}
- Top Product Line: {stats['top_product']}
- Peak Month: {stats['peak_month']}
- Trend: {stats['growth_trend']}ward

Focus on business impact and strategic implications. Keep tone professional.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            insight = response.choices[0].message.content.strip()
            st.success("AI Summary Generated!")
            st.markdown(f"**Executive Summary (GPT-4o):**  \n{insight}")
        except Exception as e:
            st.error(f"OpenAI Error: {e}")

# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.markdown("---")
st.caption(
    "GitHub: [github.com/s3achan/ai-forecasting](https://github.com/s3achan/ai-forecasting)"
)