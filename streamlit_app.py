import streamlit as st
import pandas as pd
import numpy as np # Adding numpy for general data handling best practice

# ===============================================================
# PAGE CONFIG — MUST BE FIRST STREAMLIT COMMAND
# ===============================================================
st.set_page_config(page_title="Marketing Performance Dashboard", layout="wide")

# ===============================================================
# LOAD DATA
#
# CRITICAL: The relative path to the CSV file is correct for
# Streamlit deployment via GitHub, assuming the file is in the
# same folder as this Python script.
# ===============================================================
@st.cache_data
def load_data():
    # Use the relative path to the CSV file in the GitHub repository
    df = pd.read_csv("marketing_performance_dashboard_full_v4.csv", parse_dates=["Date"])

    # Derived KPIs for richer insights, handling potential division by zero
    df["Cost_per_Conversion($)"] = np.where(df["Conversions"] > 0, df["Spend($)"] / df["Conversions"], 0)
    df["Revenue_per_Conversion($)"] = np.where(df["Conversions"] > 0, df["Revenue($)"] / df["Conversions"], 0)
    df["Profit_per_Conversion($)"] = np.where(df["Conversions"] > 0, df["Profit($)"] / df["Conversions"], 0)
    # Handle division by zero for ratio, setting to 0 or NaN if Revenue is 0
    df["Spend_to_Revenue_Ratio"] = np.where(df["Revenue($)"] > 0, df["Spend($)"] / df["Revenue($)"], np.nan) 
    
    return df

df = load_data()

# ===============================================================
# HEADER
# ===============================================================
st.title("📊 Marketing Performance Dashboard")
st.write("**Designed for both CMO and CFO perspectives**")

# ===============================================================
# SIDEBAR FILTERS
# ===============================================================
st.sidebar.header("🔍 Filters")

# Filter Options (using cached unique values for stability)
platforms = df["Platform"].unique()
regions = df["Region"].unique()
objectives = df["Campaign_Objective"].unique()
segments = df["Audience_Segment"].unique()

platform = st.sidebar.multiselect("Platform", sorted(platforms), sorted(platforms))
region = st.sidebar.multiselect("Region", sorted(regions), sorted(regions))
objective = st.sidebar.multiselect("Campaign Objective", sorted(objectives), sorted(objectives))
segment = st.sidebar.multiselect("Audience Segment", sorted(segments), sorted(segments))

# Date Range Filter
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
# Ensure date_input receives Python date objects
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Convert date_range back to pandas datetime objects for filtering
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# Apply filters
filtered_df = df[
    (df["Platform"].isin(platform)) &
    (df["Region"].isin(region)) &
    (df["Campaign_Objective"].isin(objective)) &
    (df["Audience_Segment"].isin(segment)) &
    (df["Date"].between(start_date, end_date))
]

# Check if the filtered DataFrame is empty
if filtered_df.empty:
    st.error("No data matches the selected filters. Please adjust the filter criteria.")
    # Stop execution if no data is present
    st.stop() 


# ===============================================================
# TABS: CMO + CFO
# ===============================================================
tab1, tab2 = st.tabs(["🎯 CMO Dashboard", "💰 CFO Dashboard"])

# ===============================================================
# TAB 1 — CMO DASHBOARD
# ===============================================================
with tab1:
    st.header("🎯 CMO Dashboard — Marketing Performance View")
    st.write("Focus: Reach, Engagement, and Conversion Efficiency")

    # Aggregate key metrics for KPI row
    total_impressions = filtered_df['Impressions'].sum()
    total_reach = filtered_df['Reach'].sum()
    avg_ctr = filtered_df['CTR(%)'].mean()
    avg_conversion_rate = filtered_df['Conversion_Rate(%)'].mean()
    avg_roas = filtered_df['ROAS'].mean()
    avg_engagement_rate = filtered_df['Engagement_Rate(%)'].mean()

    # KPI ROW
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Impressions", f"{total_impressions:,.0f}")
    col2.metric("Total Reach", f"{total_reach:,.0f}")
    col3.metric("Avg CTR (%)", f"{avg_ctr:.2f}")
    col4.metric("Avg Conversion Rate (%)", f"{avg_conversion_rate:.2f}")
    col5.metric("Avg ROAS", f"{avg_roas:.2f}")
    col6.metric("Avg Engagement Rate (%)", f"{avg_engagement_rate:.2f}")

    st.markdown("---")

    # CTR & Conversion Trend
    st.subheader("📈 CTR & Conversion Rate Over Time")
    chart_df = filtered_df.groupby("Date")[["CTR(%)", "Conversion_Rate(%)"]].mean()
    st.line_chart(chart_df)

    # Spend vs Conversions
    st.subheader("📊 Spend vs Conversions by Platform")
    spend_conv = filtered_df.groupby("Platform")[["Spend($)", "Conversions"]].sum()
    st.bar_chart(spend_conv)

    # Regional Reach
    st.subheader("🌍 Regional Reach")
    reach_region = filtered_df.groupby("Region")[["Reach"]].sum()
    # Use st.bar_chart for simple visualization, reset index for better labeling
    st.bar_chart(reach_region)

    # Device Split
    st.subheader("📱 Device Type Spend Share")
    device_spend = filtered_df.groupby("Device_Type")[["Spend($)"]].sum()
    # Can use a pie chart or donut chart for better visual on share, but sticking to bar_chart for simplicity
    st.bar_chart(device_spend)

    # Campaign Breakdown Table
    st.subheader("🔍 Campaign-Level Breakdown")
    st.dataframe(
        filtered_df.groupby("Campaign_Name")[["CTR(%)", "CPC($)", "CPM($)", "Conversions", "ROAS"]]
        .mean()
        .sort_values("ROAS", ascending=False)
        .round(2)
        .style.format({
            "CTR(%)": "{:.2f}",
            "CPC($)": "${:.2f}",
            "CPM($)": "${:.2f}",
            "Conversions": "{:,.0f}",
            "ROAS": "{:.2f}"
        })
    )

    st.info("💡 **Insight Tip:** High CTR but low Conversion Rate may indicate strong creative performance but poor post-click experience. Check landing page speed and relevance!")

# ===============================================================
# TAB 2 — CFO DASHBOARD
# ===============================================================
with tab2:
    st.header("💰 CFO Dashboard — Financial Efficiency View")
    st.write("Focus: Spend Optimization, Profitability, ROI, and Cost Efficiency")

    # Aggregate key metrics for KPI row
    total_spend = filtered_df['Spend($)'].sum()
    total_revenue = filtered_df['Revenue($)'].sum()
    total_profit = filtered_df['Profit($)'].sum()
    avg_roi = filtered_df['ROI(%)'].mean()
    avg_cac = filtered_df['CAC($)'].mean()
    # Calculate weighted average profit margin by total revenue if needed, but mean of percentage is fine for quick view
    avg_profit_margin = filtered_df['Profit_Margin'].mean()

    # KPI ROW
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Spend ($)", f"${total_spend:,.0f}")
    col2.metric("Total Revenue ($)", f"${total_revenue:,.0f}")
    col3.metric("Total Profit ($)", f"${total_profit:,.0f}")
    col4.metric("Avg ROI (%)", f"{avg_roi:.2f}")
    col5.metric("Avg CAC ($)", f"${avg_cac:.2f}")
    col6.metric("Avg Profit Margin (%)", f"{avg_profit_margin * 100:.2f}")

    st.markdown("---")

    # Spend vs Revenue
    st.subheader("📈 Spend vs Revenue Over Time")
    time_trend = filtered_df.groupby("Date")[["Spend($)", "Revenue($)"]].sum()
    st.line_chart(time_trend)

    # Profit & ROI by Platform
    st.subheader("💹 Profit & ROI by Platform")
    # Using sum for Profit and mean for ROI
    roi_platform = filtered_df.groupby("Platform").agg(
        {'Profit($)': 'sum', 'ROI(%)': 'mean'}
    ).sort_values("Profit($)", ascending=False)
    st.bar_chart(roi_platform)

    # ROI by Region
    st.subheader("🌍 ROI by Region")
    roi_region = filtered_df.groupby("Region")[["ROI(%)"]].mean().sort_values("ROI(%)", ascending=False)
    st.bar_chart(roi_region)

    # Platform Summary Table
    st.subheader("🔍 Platform Financial Summary")
    st.dataframe(
        filtered_df.groupby("Platform")[["Spend($)", "Revenue($)", "Profit($)", "ROI(%)", "CAC($)", "ROAS"]]
        .sum() # Sum all metrics for a comprehensive platform view
        .round(2)
        .sort_values("ROI(%)", ascending=False)
        .style.format({
            "Spend($)": "${:,.0f}",
            "Revenue($)": "${:,.0f}",
            "Profit($)": "${:,.0f}",
            "ROI(%)": "{:.2f}",
            "CAC($)": "${:.2f}",
            "ROAS": "{:.2f}"
        })
    )

    st.info("💡 **CFO Insight Tip:** High ROI with low CAC indicates efficient marketing spend. Analyze region-wise ROI to reallocate future budgets. Consider cost-per-conversion and profit-per-conversion metrics (available in the raw data) for deeper unit economics.")
