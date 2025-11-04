import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px # Import Plotly for richer visualizations

# ===============================================================
# PAGE CONFIG — MUST BE FIRST STREAMLIT COMMAND
# ===============================================================
st.set_page_config(page_title="Marketing Performance Dashboard", layout="wide")

# ===============================================================
# LOAD DATA
# ===============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("marketing_performance_dashboard_full_v4.csv", parse_dates=["Date"])

    # Derived KPIs, handling potential division by zero (already in place)
    df["Cost_per_Conversion($)"] = np.where(df["Conversions"] > 0, df["Spend($)"] / df["Conversions"], 0)
    df["Revenue_per_Conversion($)"] = np.where(df["Conversions"] > 0, df["Revenue($)"] / df["Conversions"], 0)
    df["Profit_per_Conversion($)"] = np.where(df["Conversions"] > 0, df["Profit($)"] / df["Conversions"], 0)
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
# (No changes needed here, as the filtering logic is robust)
# ===============================================================
st.sidebar.header("🔍 Filters")

platforms = df["Platform"].unique()
regions = df["Region"].unique()
objectives = df["Campaign_Objective"].unique()
segments = df["Audience_Segment"].unique()

platform = st.sidebar.multiselect("Platform", sorted(platforms), sorted(platforms))
region = st.sidebar.multiselect("Region", sorted(regions), sorted(regions))
objective = st.sidebar.multiselect("Campaign Objective", sorted(objectives), sorted(objectives))
segment = st.sidebar.multiselect("Audience Segment", sorted(segments), sorted(segments))

min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

filtered_df = df[
    (df["Platform"].isin(platform)) &
    (df["Region"].isin(region)) &
    (df["Campaign_Objective"].isin(objective)) &
    (df["Audience_Segment"].isin(segment)) &
    (df["Date"].between(start_date, end_date))
]

if filtered_df.empty:
    st.error("No data matches the selected filters. Please adjust the filter criteria.")
    st.stop() 


# ===============================================================
# TABS: CMO + CFO
# ===============================================================
tab1, tab2 = st.tabs(["🎯 CMO Dashboard", "💰 CFO Dashboard"])

# ===============================================================
# TAB 1 — CMO DASHBOARD (Enhanced)
# ===============================================================
with tab1:
    st.header("🎯 CMO Dashboard — Marketing Performance View (Enhanced)")
    st.write("Focus: Reach, Engagement, and Conversion Efficiency")

    # Aggregate key metrics for KPI row
    total_impressions = filtered_df['Impressions'].sum()
    total_reach = filtered_df['Reach'].sum()
    avg_ctr = filtered_df['CTR(%)'].mean()
    avg_conversion_rate = filtered_df['Conversion_Rate(%)'].mean()
    avg_roas = filtered_df['ROAS'].mean()
    avg_cpa = filtered_df['Cost_per_Conversion($)'].mean() # NEW KPI
    
    # KPI ROW - Row 1 (Focus on Volume/Cost)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Impressions", f"{total_impressions:,.0f}")
    col2.metric("Total Reach", f"{total_reach:,.0f}")
    col3.metric("Avg CPA ($)", f"${avg_cpa:.2f}") # CPA is added here

    # KPI ROW - Row 2 (Focus on Efficiency/Quality)
    col4, col5, col6 = st.columns(3)
    col4.metric("Avg CTR (%)", f"{avg_ctr:.2f}")
    col5.metric("Avg Conversion Rate (%)", f"{avg_conversion_rate:.2f}")
    col6.metric("Avg Engagement Rate (%)", f"{filtered_df['Engagement_Rate(%)'].mean():.2f}")


    st.markdown("---")

    # NEW: CONVERSION FUNNEL (Visualization for the CMO)
    st.subheader("📉 Full Funnel Performance: Clicks, Conversions, and Revenue")
    funnel_data = filtered_df[["Clicks", "Conversions", "Revenue($)"]].sum().reset_index()
    funnel_data.columns = ['Metric', 'Value']

    col_funnel, col_placeholder = st.columns([2, 1])

    with col_funnel:
        fig_funnel = px.bar(
            funnel_data, 
            x='Metric', 
            y='Value', 
            title='Funnel Volume Breakdown (Clicks and Conversions)',
            color='Metric',
            color_discrete_map={'Clicks': 'rgb(245, 130, 48)', 'Conversions': 'rgb(70, 130, 180)', 'Revenue($)': 'rgb(60, 179, 113)'},
            text='Value'
        )
        # Customizing text display for better formatting
        fig_funnel.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_funnel.update_layout(showlegend=False, xaxis_title="", yaxis_title="Volume/Value")
        st.plotly_chart(fig_funnel, use_container_width=True)


    # CTR & Conversion Trend (Using original Streamlit chart for variety)
    st.subheader("📈 Efficiency Trend: CTR & Conversion Rate Over Time")
    chart_df = filtered_df.groupby("Date")[["CTR(%)", "Conversion_Rate(%)"]].mean()
    st.line_chart(chart_df)

    # Spend vs Conversions
    st.subheader("📊 Spend vs Conversions by Platform")
    spend_conv = filtered_df.groupby("Platform")[["Spend($)", "Conversions"]].sum()
    st.bar_chart(spend_conv)

    # Campaign Breakdown Table (No change, as it is already good)
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

    st.info("💡 **CMO Insight Tip:** High CPA may indicate inefficiency. Use the funnel chart to identify if the cost is driven by low CTR (bad ad) or low Conversion Rate (bad landing page).")

# ===============================================================
# TAB 2 — CFO DASHBOARD (Enhanced)
# ===============================================================
with tab2:
    st.header("💰 CFO Dashboard — Financial Efficiency View (Enhanced)")
    st.write("Focus: Spend Optimization, Profitability, ROI, and Unit Economics")

    # Aggregate key metrics for KPI row
    total_spend = filtered_df['Spend($)'].sum()
    total_revenue = filtered_df['Revenue($)'].sum()
    total_profit = filtered_df['Profit($)'].sum()
    avg_roi = filtered_df['ROI(%)'].mean()
    avg_cac = filtered_df['CAC($)'].mean()
    # Weighted average calculation is more accurate for these ratios
    avg_profit_per_conv = (filtered_df['Profit($)'].sum() / filtered_df['Conversions'].sum()) if filtered_df['Conversions'].sum() > 0 else 0
    avg_spend_to_revenue_ratio = (total_spend / total_revenue) if total_revenue > 0 else 0


    # KPI ROW - Row 1 (Focus on Totals)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue ($)", f"${total_revenue:,.0f}")
    col2.metric("Total Spend ($)", f"${total_spend:,.0f}")
    col3.metric("Total Profit ($)", f"${total_profit:,.0f}")

    # KPI ROW - Row 2 (Focus on Unit/Financial Efficiency)
    col4, col5, col6 = st.columns(3)
    col4.metric("Avg ROI (%)", f"{avg_roi:.2f}")
    col5.metric("Avg CAC ($)", f"${avg_cac:.2f}")
    col6.metric("Profit per Conversion ($)", f"${avg_profit_per_conv:.2f}") # NEW KPI

    st.markdown("---")

    # ENHANCED: Spend vs Revenue vs Profit over time
    st.subheader("📈 Financial Trend: Revenue, Spend, and Profit Over Time")
    time_trend = filtered_df.groupby("Date")[["Spend($)", "Revenue($)", "Profit($)"]].sum()
    st.line_chart(time_trend)
    
    # Unit Economics Table and Chart in a single row
    st.subheader("💹 Platform Unit Economics vs ROI")
    col_ratio, col_roi_chart = st.columns(2)

    with col_ratio:
        # Unit Economics: Cost_per_Conversion, Revenue_per_Conversion, Profit_per_Conversion
        unit_econ_df = filtered_df.groupby("Platform").agg(
            {'Cost_per_Conversion($)': 'mean', 'Profit_per_Conversion($)': 'mean', 'Spend_to_Revenue_Ratio': 'mean'}
        ).sort_values('Profit_per_Conversion($)', ascending=False)
        
        st.dataframe(
            unit_econ_df
            .round(2)
            .style.format({
                "Cost_per_Conversion($)": "${:.2f}",
                "Profit_per_Conversion($)": "${:.2f}",
                "Spend_to_Revenue_Ratio": "{:.2f}"
            })
        )
        st.info(f"Global Avg Spend/Revenue Ratio: {avg_spend_to_revenue_ratio:.2f}")

    with col_roi_chart:
        # ROI by Region (using Plotly for better aesthetics)
        roi_region = filtered_df.groupby("Region")[["ROI(%)"]].mean().reset_index()
        fig_roi_region = px.bar(
            roi_region, 
            x='Region', 
            y='ROI(%)', 
            title='Avg ROI (%) by Region',
            color='ROI(%)',
            color_continuous_scale=px.colors.sequential.Teal,
        )
        fig_roi_region.update_layout(xaxis_title="", yaxis_title="Average ROI (%)")
        st.plotly_chart(fig_roi_region, use_container_width=True)


    # Platform Summary Table (Updated to use sum for consistency)
    st.subheader("🔍 Platform Financial Summary (Totals)")
    st.dataframe(
        filtered_df.groupby("Platform")[["Spend($)", "Revenue($)", "Profit($)", "ROI(%)", "CAC($)", "ROAS"]]
        .sum() # Sum all financial metrics for platform totals
        .round(2)
        .sort_values("Profit($)", ascending=False)
        .style.format({
            "Spend($)": "${:,.0f}",
            "Revenue($)": "${:,.0f}",
            "Profit($)": "${:,.0f}",
            "ROI(%)": "{:.2f}", # Keep this as a general percentage
            "CAC($)": "${:.2f}", # CAC from the raw data is averaged here, but should technically be calculated from total spend/conversions
            "ROAS": "{:.2f}"
        })
    )

    st.info("💡 **CFO Insight Tip:** Focus on platforms and regions with the highest **Profit per Conversion** and the lowest **Spend to Revenue Ratio** to maximize efficient budget allocation.")
