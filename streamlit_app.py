import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ===============================================================
# PAGE CONFIG
# ===============================================================
st.set_page_config(
    page_title="Marketing Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================
# CUSTOM CSS (updated: slightly smaller text sizes)
# ===============================================================
st.markdown("""
    <style>
    /* Reduce overall app font size slightly */
    html, body, .streamlit-container {
        font-size: 14px; /* slightly smaller */
    }

    /* Headings slightly reduced */
    h1, .stApp h1 {
        font-size: 24px !important;
    }
    h2, .stApp h2 {
        font-size: 20px !important;
    }
    h3, .stApp h3 {
        font-size: 16px !important;
    }

    /* Markdown / paragraph body */
    .stMarkdown, .stText, p {
        font-size: 14px !important;
        line-height: 1.35;
    }

    /* Metric / card adjustments */
    .metric-card {
        background-color: #f0f2f6;
        padding: 14px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        font-size: 14px;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 8px 0;
        font-size: 14px;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 8px 0;
        font-size: 14px;
    }

    /* Sidebar and caption smaller */
    .stSidebar, .stSidebar .block-container, .stCaption {
        font-size: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================================================
# LOAD DATA
# ===============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("marketing_performance_dashboard_realistic_v5.csv", parse_dates=["Date"])
    
    # Ensure all numeric calculations are safe
    df["Cost_per_Conversion($)"] = np.where(
        df["Conversions"] > 0, 
        df["Spend($)"] / df["Conversions"], 
        0
    )
    df["Revenue_per_Conversion($)"] = np.where(
        df["Conversions"] > 0, 
        df["Revenue($)"] / df["Conversions"], 
        0
    )
    df["Profit_per_Conversion($)"] = np.where(
        df["Conversions"] > 0, 
        df["Profit($)"] / df["Conversions"], 
        0
    )
    
    return df

df = load_data()

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================
def calculate_weighted_avg(df, metric_col, weight_col):
    """Calculate weighted average for metrics like ROI, ROAS"""
    total_weight = df[weight_col].sum()
    if total_weight == 0:
        return 0
    return (df[metric_col] * df[weight_col]).sum() / total_weight

def calculate_period_comparison(filtered_df, date_col="Date"):
    """Calculate period-over-period comparison"""
    if len(filtered_df) == 0:
        return None
    
    # Split into two periods
    mid_point = filtered_df[date_col].min() + (filtered_df[date_col].max() - filtered_df[date_col].min()) / 2
    period_1 = filtered_df[filtered_df[date_col] <= mid_point]
    period_2 = filtered_df[filtered_df[date_col] > mid_point]
    
    if len(period_1) == 0 or len(period_2) == 0:
        return None
    
    metrics = {
        'revenue': (period_2['Revenue($)'].sum() / period_1['Revenue($)'].sum() - 1) * 100 if period_1['Revenue($)'].sum() > 0 else 0,
        'spend': (period_2['Spend($)'].sum() / period_1['Spend($)'].sum() - 1) * 100 if period_1['Spend($)'].sum() > 0 else 0,
        'conversions': (period_2['Conversions'].sum() / period_1['Conversions'].sum() - 1) * 100 if period_1['Conversions'].sum() > 0 else 0,
    }
    
    return metrics

def format_large_number(num):
    """Format large numbers with K, M suffixes"""
    if num >= 1_000_000:
        return f"${num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"${num/1_000:.1f}K"
    else:
        return f"${num:.0f}"

# ===============================================================
# SIDEBAR FILTERS
# ===============================================================
st.sidebar.header("🔍 Filters")

platforms = df["Platform"].unique()
regions = df["Region"].unique()
objectives = df["Campaign_Objective"].unique()
segments = df["Audience_Segment"].unique()
devices = df["Device_Type"].unique()

platform = st.sidebar.multiselect("Platform", sorted(platforms), sorted(platforms))
region = st.sidebar.multiselect("Region", sorted(regions), sorted(regions))
objective = st.sidebar.multiselect("Campaign Objective", sorted(objectives), sorted(objectives))
segment = st.sidebar.multiselect("Audience Segment", sorted(segments), sorted(segments))
device = st.sidebar.multiselect("Device Type", sorted(devices), sorted(devices))

min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range", 
    [min_date, max_date], 
    min_value=min_date, 
    max_value=max_date
)

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1]) if len(date_range) > 1 else pd.to_datetime(date_range[0])

# Apply filters
filtered_df = df[
    (df["Platform"].isin(platform)) &
    (df["Region"].isin(region)) &
    (df["Campaign_Objective"].isin(objective)) &
    (df["Audience_Segment"].isin(segment)) &
    (df["Device_Type"].isin(device)) &
    (df["Date"].between(start_date, end_date))
]

if filtered_df.empty:
    st.error("⚠️ No data matches the selected filters. Please adjust the filter criteria.")
    st.stop()

# ===============================================================
# HEADER WITH EXECUTIVE SUMMARY
# ===============================================================
st.title("📊 Marketing Performance Dashboard")
st.markdown("**Multi-Stakeholder View: CMO & CFO Analytics**")

# Period comparison
comparison = calculate_period_comparison(filtered_df)

if comparison:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Revenue Trend", 
            format_large_number(filtered_df['Revenue($)'].sum()),
            f"{comparison['revenue']:+.1f}%" if comparison else None
        )
    with col2:
        st.metric(
            "Spend Trend", 
            format_large_number(filtered_df['Spend($)'].sum()),
            f"{comparison['spend']:+.1f}%" if comparison else None
        )
    with col3:
        st.metric(
            "Conversions Trend", 
            f"{filtered_df['Conversions'].sum():,.0f}",
            f"{comparison['conversions']:+.1f}%" if comparison else None
        )
    with col4:
        weighted_roas = calculate_weighted_avg(filtered_df, 'ROAS', 'Spend($)')
        st.metric("Weighted ROAS", f"{weighted_roas:.2f}x")

st.markdown("---")

# ===============================================================
# TABS: Executive Summary, CMO, CFO
# ===============================================================
tab_exec, tab_cmo, tab_cfo = st.tabs(["📋 Executive Summary", "🎯 CMO Dashboard", "💰 CFO Dashboard"])

# ===============================================================
# TAB 0 — EXECUTIVE SUMMARY
# ===============================================================
with tab_exec:
    st.header("📋 Executive Summary")
    st.write("**Key Performance Highlights & Strategic Insights**")
    
    # Top performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performing Campaigns")
        top_campaigns = filtered_df.groupby('Campaign_Name').agg({
            'Revenue($)': 'sum',
            'Spend($)': 'sum',
            'Profit($)': 'sum',
            'ROAS': 'mean'
        }).sort_values('Profit($)', ascending=False).head(5)
        
        top_campaigns['ROI(%)'] = ((top_campaigns['Revenue($)'] - top_campaigns['Spend($)']) / top_campaigns['Spend($)'] * 100)
        
        st.dataframe(
            top_campaigns[['Revenue($)', 'Profit($)', 'ROAS', 'ROI(%)']].style.format({
                'Revenue($)': '${:,.0f}',
                'Profit($)': '${:,.0f}',
                'ROAS': '{:.2f}x',
                'ROI(%)': '{:.1f}%'
            }),
            use_container_width=True
        )
    
    with col2:
        st.subheader("⚠️ Underperforming Campaigns")
        bottom_campaigns = filtered_df.groupby('Campaign_Name').agg({
            'Revenue($)': 'sum',
            'Spend($)': 'sum',
            'Profit($)': 'sum',
            'ROAS': 'mean'
        }).sort_values('Profit($)', ascending=True).head(5)
        
        bottom_campaigns['ROI(%)'] = ((bottom_campaigns['Revenue($)'] - bottom_campaigns['Spend($)']) / bottom_campaigns['Spend($)'] * 100)
        
        st.dataframe(
            bottom_campaigns[['Revenue($)', 'Profit($)', 'ROAS', 'ROI(%)']].style.format({
                'Revenue($)': '${:,.0f}',
                'Profit($)': '${:,.0f}',
                'ROAS': '{:.2f}x',
                'ROI(%)': '{:.1f}%'
            }),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Platform comparison
    st.subheader("📊 Platform Performance Comparison")
    
    platform_metrics = filtered_df.groupby('Platform').agg({
        'Spend($)': 'sum',
        'Revenue($)': 'sum',
        'Profit($)': 'sum',
        'Conversions': 'sum',
        'Impressions': 'sum',
        'Clicks': 'sum'
    })
    
    platform_metrics['ROAS'] = platform_metrics['Revenue($)'] / platform_metrics['Spend($)']
    platform_metrics['ROI(%)'] = (platform_metrics['Revenue($)'] - platform_metrics['Spend($)']) / platform_metrics['Spend($)'] * 100
    platform_metrics['CPA($)'] = platform_metrics['Spend($)'] / platform_metrics['Conversions']
    platform_metrics['CTR(%)'] = platform_metrics['Clicks'] / platform_metrics['Impressions'] * 100
    
    # Create radar chart for platform comparison
    fig_radar = go.Figure()
    
    metrics_for_radar = ['ROAS', 'ROI(%)', 'CTR(%)']
    
    for platform_name in platform_metrics.index:
        values = []
        for metric in metrics_for_radar:
            # Normalize to 0-100 scale for better visualization
            val = platform_metrics.loc[platform_name, metric]
            if metric == 'ROAS':
                val = min(val * 20, 100)  # Scale ROAS
            elif metric == 'ROI(%)':
                val = min(val, 100)
            elif metric == 'CTR(%)':
                val = min(val * 20, 100)  # Scale CTR
            values.append(val)
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_for_radar,
            fill='toself',
            name=platform_name
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Platform Performance Comparison (Normalized)"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # ===============================================================
# KEY INSIGHTS (Dark + Light Mode Readable)
# ===============================================================
st.subheader("💡 Key Insights")

best_platform = platform_metrics.sort_values('Profit($)', ascending=False).index[0]
best_profit = platform_metrics.loc[best_platform, 'Profit($)']
best_roas = platform_metrics.loc[best_platform, 'ROAS']

worst_platform = platform_metrics.sort_values('Profit($)', ascending=True).index[0]
worst_profit = platform_metrics.loc[worst_platform, 'Profit($)']

# Best-performing box (blue)
st.markdown(f"""
<div style="
    background-color:#1e88e5;
    color:#ffffff;
    padding:14px 18px;
    border-radius:10px;
    border-left:5px solid #1565c0;
    margin:8px 0;
    font-size:15px;
">
<strong>✅ Best Performing Platform:</strong> 
{best_platform} generated <strong>${best_profit:,.0f}</strong> in profit 
with a ROAS of <strong>{best_roas:.2f}x</strong>.
</div>
""", unsafe_allow_html=True)

# Underperforming box (amber)
st.markdown(f"""
<div style="
    background-color:#ffb300;
    color:#0d1117;
    padding:14px 18px;
    border-radius:10px;
    border-left:5px solid #f57c00;
    margin:8px 0;
    font-size:15px;
">
<strong>⚠️ Attention Required:</strong> 
{worst_platform} generated only <strong>${worst_profit:,.0f}</strong> in profit. 
Consider budget reallocation or campaign optimization.
</div>
""", unsafe_allow_html=True)


# ===============================================================
# TAB 1 — CMO DASHBOARD
# ===============================================================
with tab_cmo:
    st.header("🎯 CMO Dashboard — Marketing Performance View")
    st.write("**Focus: Reach, Engagement, Conversion Efficiency, and Audience Insights**")
    
    # KPI Row 1
    total_impressions = filtered_df['Impressions'].sum()
    total_reach = filtered_df['Reach'].sum()
    total_clicks = filtered_df['Clicks'].sum()
    avg_cpa = (filtered_df['Spend($)'].sum() / filtered_df['Conversions'].sum()) if filtered_df['Conversions'].sum() > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Impressions", f"{total_impressions:,.0f}")
    col2.metric("Total Reach", f"{total_reach:,.0f}")
    col3.metric("Total Clicks", f"{total_clicks:,.0f}")
    col4.metric("Avg CPA", f"${avg_cpa:.2f}")
    
    # KPI Row 2
    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    avg_conv_rate = (filtered_df['Conversions'].sum() / total_clicks * 100) if total_clicks > 0 else 0
    avg_engagement_rate = (filtered_df['Engagements'].sum() / total_impressions * 100) if total_impressions > 0 else 0
    weighted_roas = calculate_weighted_avg(filtered_df, 'ROAS', 'Spend($)')
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg CTR", f"{avg_ctr:.2f}%")
    col2.metric("Avg Conversion Rate", f"{avg_conv_rate:.2f}%")
    col3.metric("Avg Engagement Rate", f"{avg_engagement_rate:.2f}%")
    col4.metric("Weighted ROAS", f"{weighted_roas:.2f}x")
    
    st.markdown("---")
    
    # Conversion Funnel
    st.subheader("📉 Marketing Funnel Analysis")
    
    col_funnel, col_device = st.columns([2, 1])
    
    with col_funnel:
        funnel_metrics = {
            'Stage': ['Impressions', 'Clicks', 'Conversions'],
            'Volume': [
                total_impressions,
                total_clicks,
                filtered_df['Conversions'].sum()
            ],
            'Conversion Rate': [
                100,
                (total_clicks / total_impressions * 100) if total_impressions > 0 else 0,
                (filtered_df['Conversions'].sum() / total_clicks * 100) if total_clicks > 0 else 0
            ]
        }
        
        fig_funnel = go.Figure()
        
        fig_funnel.add_trace(go.Funnel(
            y=funnel_metrics['Stage'],
            x=funnel_metrics['Volume'],
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=["#636EFA", "#EF553B", "#00CC96"])
        ))
        
        fig_funnel.update_layout(
            title="Conversion Funnel",
            height=400
        )
        
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col_device:
        st.markdown("**Device Performance**")
        device_performance = filtered_df.groupby('Device_Type').agg({
            'Conversions': 'sum',
            'CTR(%)': 'mean',
            'Conversion_Rate(%)': 'mean'
        }).round(2)
        
        st.dataframe(
            device_performance.style.format({
                'Conversions': '{:,.0f}',
                'CTR(%)': '{:.2f}%',
                'Conversion_Rate(%)': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        # Device pie chart
        device_conv = filtered_df.groupby('Device_Type')['Conversions'].sum()
        fig_device = px.pie(
            values=device_conv.values,
            names=device_conv.index,
            title="Conversions by Device"
        )
        st.plotly_chart(fig_device, use_container_width=True)
    
    st.markdown("---")
    
    # Performance trends
    col_trend1, col_trend2 = st.columns(2)
    
    with col_trend1:
        st.subheader("📈 CTR & Conversion Rate Trends")
        trend_data = filtered_df.groupby('Date').agg({
            'Impressions': 'sum',
            'Clicks': 'sum',
            'Conversions': 'sum'
        })
        trend_data['CTR(%)'] = (trend_data['Clicks'] / trend_data['Impressions'] * 100)
        trend_data['Conv_Rate(%)'] = (trend_data['Conversions'] / trend_data['Clicks'] * 100)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend_data.index,
            y=trend_data['CTR(%)'],
            name='CTR (%)',
            line=dict(color='#636EFA')
        ))
        fig_trend.add_trace(go.Scatter(
            x=trend_data.index,
            y=trend_data['Conv_Rate(%)'],
            name='Conv Rate (%)',
            line=dict(color='#EF553B'),
            yaxis='y2'
        ))
        
        fig_trend.update_layout(
            yaxis=dict(title='CTR (%)'),
            yaxis2=dict(title='Conversion Rate (%)', overlaying='y', side='right'),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_trend2:
        st.subheader("💰 Spend vs Conversions by Platform")
        platform_perf = filtered_df.groupby('Platform').agg({
            'Spend($)': 'sum',
            'Conversions': 'sum'
        })
        
        fig_platform = go.Figure()
        fig_platform.add_trace(go.Bar(
            x=platform_perf.index,
            y=platform_perf['Spend($)'],
            name='Spend ($)',
            marker_color='#FF6B6B'
        ))
        fig_platform.add_trace(go.Bar(
            x=platform_perf.index,
            y=platform_perf['Conversions'] * 10,  # Scale for visualization
            name='Conversions (×10)',
            marker_color='#4ECDC4'
        ))
        
        fig_platform.update_layout(
            barmode='group',
            height=400,
            yaxis_title='Value'
        )
        
        st.plotly_chart(fig_platform, use_container_width=True)
    
    st.markdown("---")
    
    # Audience segment analysis
    st.subheader("👥 Audience Segment Performance")
    
    segment_perf = filtered_df.groupby('Audience_Segment').agg({
        'Conversions': 'sum',
        'Revenue($)': 'sum',
        'Spend($)': 'sum',
        'CTR(%)': 'mean',
        'Conversion_Rate(%)': 'mean'
    })
    segment_perf['ROAS'] = segment_perf['Revenue($)'] / segment_perf['Spend($)']
    segment_perf['CPA($)'] = segment_perf['Spend($)'] / segment_perf['Conversions']
    
    st.dataframe(
        segment_perf.sort_values('Revenue($)', ascending=False).style.format({
            'Conversions': '{:,.0f}',
            'Revenue($)': '${:,.0f}',
            'Spend($)': '${:,.0f}',
            'CTR(%)': '{:.2f}%',
            'Conversion_Rate(%)': '{:.2f}%',
            'ROAS': '{:.2f}x',
            'CPA($)': '${:.2f}'
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Campaign breakdown
    st.subheader("🔍 Top 10 Campaigns by Performance")
    
    campaign_detail = filtered_df.groupby(['Campaign_Name', 'Platform']).agg({
        'CTR(%)': 'mean',
        'Conversion_Rate(%)': 'mean',
        'CPC($)': 'mean',
        'Conversions': 'sum',
        'Spend($)': 'sum',
        'Revenue($)': 'sum'
    })
    campaign_detail['ROAS'] = campaign_detail['Revenue($)'] / campaign_detail['Spend($)']
    
    top_campaigns = campaign_detail.sort_values('Revenue($)', ascending=False).head(10)
    
    st.dataframe(
        top_campaigns.style.format({
            'CTR(%)': '{:.2f}%',
            'Conversion_Rate(%)': '{:.2f}%',
            'CPC($)': '${:.2f}',
            'Conversions': '{:,.0f}',
            'Spend($)': '${:,.0f}',
            'Revenue($)': '${:,.0f}',
            'ROAS': '{:.2f}x'
        }),
        use_container_width=True
    )
    
    st.info("💡 **CMO Insight:** Focus on campaigns with high CTR but low conversion rates - this indicates strong creative but potential landing page or offer issues. Conversely, low CTR with high conversion rates suggests targeting improvements needed.")

# ===============================================================
# TAB 2 — CFO DASHBOARD
# ===============================================================
with tab_cfo:
    st.header("💰 CFO Dashboard — Financial Efficiency View")
    st.write("**Focus: Spend Optimization, Profitability, ROI, and Unit Economics**")
    
    # Financial KPIs Row 1
    total_spend = filtered_df['Spend($)'].sum()
    total_revenue = filtered_df['Revenue($)'].sum()
    total_profit = filtered_df['Profit($)'].sum()
    profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", format_large_number(total_revenue))
    col2.metric("Total Spend", format_large_number(total_spend))
    col3.metric("Total Profit", format_large_number(total_profit))
    col4.metric("Profit Margin", f"{profit_margin:.1f}%")
    
    # Financial KPIs Row 2
    weighted_roi = ((total_revenue - total_spend) / total_spend * 100) if total_spend > 0 else 0
    avg_cac = total_spend / filtered_df['Conversions'].sum() if filtered_df['Conversions'].sum() > 0 else 0
    profit_per_conv = total_profit / filtered_df['Conversions'].sum() if filtered_df['Conversions'].sum() > 0 else 0
    spend_to_revenue = (total_spend / total_revenue) if total_revenue > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROI", f"{weighted_roi:.1f}%")
    col2.metric("Avg CAC", f"${avg_cac:.2f}")
    col3.metric("Profit per Conversion", f"${profit_per_conv:.2f}")
    col4.metric("Spend/Revenue Ratio", f"{spend_to_revenue:.2f}")
    
    st.markdown("---")
    
    # Financial trends
    st.subheader("📈 Financial Performance Trends")
    
    financial_trend = filtered_df.groupby('Date').agg({
        'Spend($)': 'sum',
        'Revenue($)': 'sum',
        'Profit($)': 'sum'
    })
    
    fig_financial = go.Figure()
    fig_financial.add_trace(go.Scatter(
        x=financial_trend.index,
        y=financial_trend['Revenue($)'],
        name='Revenue',
        fill='tonexty',
        line=dict(color='#00CC96')
    ))
    fig_financial.add_trace(go.Scatter(
        x=financial_trend.index,
        y=financial_trend['Spend($)'],
        name='Spend',
        line=dict(color='#EF553B')
    ))
    fig_financial.add_trace(go.Scatter(
        x=financial_trend.index,
        y=financial_trend['Profit($)'],
        name='Profit',
        line=dict(color='#636EFA', width=3)
    ))
    
    fig_financial.update_layout(
        hovermode='x unified',
        height=400,
        yaxis_title='Amount ($)'
    )
    
    st.plotly_chart(fig_financial, use_container_width=True)
    
    st.markdown("---")
    
    # Platform financial performance
    col_platform, col_region = st.columns(2)
    
    with col_platform:
        st.subheader("💹 Platform Unit Economics")
        
        platform_econ = filtered_df.groupby('Platform').agg({
            'Spend($)': 'sum',
            'Revenue($)': 'sum',
            'Profit($)': 'sum',
            'Conversions': 'sum'
        })
        
        platform_econ['CPA($)'] = platform_econ['Spend($)'] / platform_econ['Conversions']
        platform_econ['Revenue_per_Conv($)'] = platform_econ['Revenue($)'] / platform_econ['Conversions']
        platform_econ['Profit_per_Conv($)'] = platform_econ['Profit($)'] / platform_econ['Conversions']
        platform_econ['ROAS'] = platform_econ['Revenue($)'] / platform_econ['Spend($)']
        platform_econ['ROI(%)'] = (platform_econ['Revenue($)'] - platform_econ['Spend($)']) / platform_econ['Spend($)'] * 100
        
        st.dataframe(
            platform_econ[['CPA($)', 'Profit_per_Conv($)', 'ROAS', 'ROI(%)']].sort_values('Profit_per_Conv($)', ascending=False).style.format({
                'CPA($)': '${:.2f}',
                'Profit_per_Conv($)': '${:.2f}',
                'ROAS': '{:.2f}x',
                'ROI(%)': '{:.1f}%'
            }),
            use_container_width=True
        )
    
    with col_region:
        st.subheader("🌍 Regional ROI Performance")
        
        region_roi = filtered_df.groupby('Region').agg({
            'Spend($)': 'sum',
            'Revenue($)': 'sum',
            'Profit($)': 'sum'
        })
        region_roi['ROI(%)'] = (region_roi['Revenue($)'] - region_roi['Spend($)']) / region_roi['Spend($)'] * 100
        
        fig_region_roi = px.bar(
            region_roi.sort_values('ROI(%)', ascending=False),
            y=region_roi.sort_values('ROI(%)', ascending=False).index,
            x='ROI(%)',
            orientation='h',
            color='ROI(%)',
            color_continuous_scale='RdYlGn',
            text='ROI(%)'
        )
        
        fig_region_roi.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_region_roi.update_layout(height=350, showlegend=False, xaxis_title="ROI (%)")
        
        st.plotly_chart(fig_region_roi, use_container_width=True)
    
    st.markdown("---")
    
    # Profitability heatmap
    st.subheader("🔥 Platform × Region Profitability Heatmap")
    
    heatmap_data = filtered_df.groupby(['Platform', 'Region'])['Profit($)'].sum().unstack(fill_value=0)
    
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Region", y="Platform", color="Profit ($)"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='RdYlGn',
        aspect='auto',
        text_auto='.0f'
    )
    
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # Budget efficiency analysis
    st.subheader("💸 Budget Efficiency Analysis")
    
    col_efficiency1, col_efficiency2 = st.columns(2)
    
    with col_efficiency1:
        st.markdown("**Campaign Objective ROI**")
        
        objective_performance = filtered_df.groupby('Campaign_Objective').agg({
            'Spend($)': 'sum',
            'Revenue($)': 'sum',
            'Profit($)': 'sum',
            'Conversions': 'sum'
        })
        objective_performance['ROI(%)'] = (objective_performance['Revenue($)'] - objective_performance['Spend($)']) / objective_performance['Spend($)'] * 100
        objective_performance['Spend_Share(%)'] = objective_performance['Spend($)'] / objective_performance['Spend($)'].sum() * 100
        
        fig_objective = px.scatter(
            objective_performance.reset_index(),
            x='Spend_Share(%)',
            y='ROI(%)',
            size='Profit($)',
            color='Campaign_Objective',
            hover_data=['Spend($)', 'Revenue($)', 'Conversions'],
            title='Budget Allocation vs ROI by Objective'
        )
        
        fig_objective.update_layout(height=400)
        st.plotly_chart(fig_objective, use_container_width=True)
    
    with col_efficiency2:
        st.markdown("**Cost Efficiency Metrics**")
        
        efficiency_metrics = filtered_df.groupby('Platform').agg({
            'CPC($)': 'mean',
            'CPM($)': 'mean',
            'Cost_per_Engagement($)': 'mean',
            'CAC($)': 'mean'
        }).round(2)
        
        st.dataframe(
            efficiency_metrics.style.format({
                'CPC($)': '${:.2f}',
                'CPM($)': '${:.2f}',
                'Cost_per_Engagement($)': '${:.2f}',
                'CAC($)': '${:.2f}'
            }).background_gradient(cmap='RdYlGn_r', axis=0),
            use_container_width=True
        )
        
        st.markdown("**Profit Margin by Platform**")
        margin_data = filtered_df.groupby('Platform').agg({
            'Profit_Margin': 'mean',
            'Profit($)': 'sum'
        }).sort_values('Profit_Margin', ascending=False)
        
        fig_margin = px.bar(
            margin_data.reset_index(),
            x='Platform',
            y='Profit_Margin',
            color='Profit_Margin',
            color_continuous_scale='Greens',
            text='Profit_Margin'
        )
        fig_margin.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_margin.update_layout(height=300, showlegend=False, yaxis_title="Profit Margin")
        st.plotly_chart(fig_margin, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed platform summary
    st.subheader("🔍 Comprehensive Platform Financial Summary")
    
    platform_summary = filtered_df.groupby('Platform').agg({
        'Spend($)': 'sum',
        'Revenue($)': 'sum',
        'Profit($)': 'sum',
        'Conversions': 'sum',
        'Impressions': 'sum',
        'Clicks': 'sum'
    })
    
    platform_summary['ROI(%)'] = (platform_summary['Revenue($)'] - platform_summary['Spend($)']) / platform_summary['Spend($)'] * 100
    platform_summary['ROAS'] = platform_summary['Revenue($)'] / platform_summary['Spend($)']
    platform_summary['CAC($)'] = platform_summary['Spend($)'] / platform_summary['Conversions']
    platform_summary['CPA($)'] = platform_summary['Spend($)'] / platform_summary['Conversions']
    platform_summary['Profit_Margin(%)'] = platform_summary['Profit($)'] / platform_summary['Revenue($)'] * 100
    
    st.dataframe(
        platform_summary[['Spend($)', 'Revenue($)', 'Profit($)', 'ROI(%)', 'ROAS', 'CAC($)', 'Profit_Margin(%)']].sort_values('Profit($)', ascending=False).style.format({
            'Spend($)': '${:,.0f}',
            'Revenue($)': '${:,.0f}',
            'Profit($)': '${:,.0f}',
            'ROI(%)': '{:.1f}%',
            'ROAS': '{:.2f}x',
            'CAC($)': '${:.2f}',
            'Profit_Margin(%)': '{:.1f}%'
        }).background_gradient(subset=['Profit($)', 'ROI(%)', 'ROAS'], cmap='Greens').background_gradient(subset=['CAC($)'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Budget allocation recommendations
    st.subheader("📊 Budget Allocation Insights")
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("**Current Spend Distribution**")
        spend_dist = filtered_df.groupby('Platform')['Spend($)'].sum()
        
        fig_spend_pie = px.pie(
            values=spend_dist.values,
            names=spend_dist.index,
            title='Spend Share by Platform',
            hole=0.4
        )
        fig_spend_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_spend_pie, use_container_width=True)
    
    with col_rec2:
        st.markdown("**Profit Contribution**")
        profit_dist = filtered_df.groupby('Platform')['Profit($)'].sum()
        
        fig_profit_pie = px.pie(
            values=profit_dist.values,
            names=profit_dist.index,
            title='Profit Share by Platform',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Greens_r
        )
        fig_profit_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_profit_pie, use_container_width=True)
    
    # Financial insights
    best_roi_platform = platform_summary.sort_values('ROI(%)', ascending=False).index[0]
    best_roi_value = platform_summary.loc[best_roi_platform, 'ROI(%)']
    
    worst_roi_platform = platform_summary.sort_values('ROI(%)', ascending=True).index[0]
    worst_roi_value = platform_summary.loc[worst_roi_platform, 'ROI(%)']
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>✅ Highest ROI:</strong> {best_roi_platform} delivers <strong>{best_roi_value:.1f}%</strong> ROI. 
    Consider increasing budget allocation to maximize returns.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="warning-box">
    <strong>⚠️ Budget Optimization Needed:</strong> {worst_roi_platform} shows only <strong>{worst_roi_value:.1f}%</strong> ROI. 
    Review campaign strategies or consider reallocating budget to higher-performing platforms.
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced metrics table
    st.subheader("📈 Advanced Financial Metrics")
    
    advanced_metrics = filtered_df.groupby(['Platform', 'Region']).agg({
        'Spend($)': 'sum',
        'Revenue($)': 'sum',
        'Profit($)': 'sum',
        'Conversions': 'sum'
    })
    
    advanced_metrics['Profit_per_Conv($)'] = advanced_metrics['Profit($)'] / advanced_metrics['Conversions']
    advanced_metrics['Revenue_per_Conv($)'] = advanced_metrics['Revenue($)'] / advanced_metrics['Conversions']
    advanced_metrics['LTV_CAC_Ratio'] = advanced_metrics['Revenue_per_Conv($)'] / (advanced_metrics['Spend($)'] / advanced_metrics['Conversions'])
    advanced_metrics['Payback_Period'] = advanced_metrics['Spend($)'] / advanced_metrics['Profit($)']
    
    top_performers = advanced_metrics.sort_values('Profit($)', ascending=False).head(10)
    
    st.dataframe(
        top_performers[['Profit($)', 'Profit_per_Conv($)', 'LTV_CAC_Ratio', 'Payback_Period']].style.format({
            'Profit($)': '${:,.0f}',
            'Profit_per_Conv($)': '${:.2f}',
            'LTV_CAC_Ratio': '{:.2f}x',
            'Payback_Period': '{:.2f}'
        }).background_gradient(subset=['Profit($)', 'LTV_CAC_Ratio'], cmap='Greens').background_gradient(subset=['Payback_Period'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    st.info("💡 **CFO Insight:** Focus on Platform-Region combinations with high Profit per Conversion and LTV:CAC ratios above 3:1. Lower payback periods indicate faster return on investment. Consider reducing spend on segments with payback periods exceeding your target threshold.")

# ===============================================================
# FOOTER
# ===============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <strong>Marketing Performance Dashboard</strong> | Designed for CMO & CFO Collaboration<br>
    Data-driven insights for strategic decision-making
</div>
""", unsafe_allow_html=True)

# Fixed footer: ensure only one top-level markdown + caption (removed duplicated indented lines)
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
