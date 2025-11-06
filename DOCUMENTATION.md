***

# Marketing Performance Dashboard — Technical Documentation

## 1. Project Overview

This documentation details the design and visual logic of the Marketing Performance Dashboard, which synthesizes actionable reporting for executive, CMO, and CFO stakeholders in an e-commerce context. The dashboard fulfills assignment requirements by delivering both business and technical clarity, and every major visual is justified with its rationale.

***

## 2. Assignment Context

- **CMO:** Needs campaign performance, funnel, segment, and creative insights for growth.
- **CFO:** Requires thorough visibility of spend, profit, cost efficiency, and ROI to guide investment.
- **Executive:** Demands high-level summaries to prioritize action and investment.
- Data is synthetically created to mimic true omnichannel marketing reality (across Meta, Google, Amazon, TikTok).

***

## 3. Data Generation (`data_gen.py`)

- **Purpose:** To mirror real ad ecosystems in the absence of public multi-platform e-commerce datasets.
- **Scope:** 120 days × 4 platforms × 5 campaigns × 4 regions × 3 devices × 3 audience segments × 4 objectives (~2,400 rows).
- **Metrics:** All visualized KPIs (impressions, revenue, ROAS, CAC, conversions, profit, engagement, etc.) are computed to allow for multi-dimensional slicing.

***

## 4. Dashboard Structure, Visuals, and Visual Rationale

### Shared Features

- **Universal Filters:** Platform, region, campaign objective, device, segment, and date range.  
  *Rationale*: Ensures every stakeholder can target analytics to their exact context, maximizing decision relevance.

***

### Executive Summary Tab

#### Visuals & Rationale

- **Period-over-Period Comparison (Metric Cards):** Shows recent revenue, spend, and conversions vs. previous period.
  - *Rationale*: Trend cards make shifts in business growth/costs instantly visible for strategic interventions.

- **Weighted ROAS Metric:** Combines campaign ROAS values weighted by spend.
  - *Rationale*: Provides a true picture of return on investment, accounting for actual resource allocation.

- **Top & Bottom Performing Campaign Tables:** DataFrames ranking campaigns by profit, revenue, ROI, and ROAS.
  - *Rationale*: Directs executive and CMO/CFO attention to success stories and problem areas—enables rapid resource reallocation.

- **Platform Performance Radar Chart (Normalized):** Radar/Spider chart mapping major platforms across ROAS, ROI, CTR.
  - *Rationale*: Delivers at-a-glance multivariate comparison; radar’s structure surfaces both strengths and gaps for each channel efficiently.

- **Key Insights (Highlight/Warning Boxes):** Automated call-outs for best/worst platforms (profit, ROAS).
  - *Rationale*: Draws managerial focus to the highest-leverage action points without requiring data exploration.

***

### CMO Dashboard

#### Visuals & Rationale

- **Total Impressions, Reach, Clicks, CPA (Metric Cards):** Summarizes the scope and outcomes of all marketing exposure.
  - *Rationale*: Puts scale and engagement front and center; CPA allows instant efficiency checks.

- **CTR, Conversion Rate, Engagement Rate, Weighted ROAS (Metric Cards):** KPIs summarizing audience interaction and performance per spend.
  - *Rationale*: These percentages help marketers identify where their campaigns are excelling or underperforming in the funnel, driving rapid creative/targeting pivots.

- **Marketing Funnel Visualization (Funnel Chart):** Visualizes the drop-off between impressions → clicks → conversions.
  - *Rationale*: Makes obvious where in the marketing journey attention or testing is most needed (creative/targeting vs. website/offer vs. checkout).

- **Device Performance (Table & Pie Chart):** Detailed conversion KPIs by device; pie chart for share-of-total.
  - *Rationale*: Supports budget and creative optimization by showing which device types actually produce results, not just reach.

- **CTR & Conversion Rate Trend (Over Time):** Line plots of user engagement and conversion efficiency.
  - *Rationale*: Quickly surfaces seasonality, campaign learning effects, or emergent successes/challenges over time.

- **Spend vs. Conversions by Platform (Bar/Scatter Chart):** Contrasts spend (bars) and conversions (bars/scatter) across channels.
  - *Rationale*: Visually quantifies ROI of each channel, guiding immediate budget reallocations from a marketing perspective.

- **Audience Segment Performance (Table):** KPIs and ROI per customer segment.
  - *Rationale*: Allows CMO to shift targeting/creative spend toward most valuable segments.

- **Top 10 Campaigns (Table):** Multi-KPI table enabling quick comparison.
  - *Rationale*: Supports micro-optimization of specific campaigns and creative variants.

***

### CFO Dashboard

#### Visuals & Rationale

- **Core Financial KPIs (Metric Cards):** Revenue, spend, profit, margin, ROI, CAC, profit per conversion.
  - *Rationale*: These top-level stats let the CFO instantly judge effectiveness, efficiency, and the health of marketing investment.

- **Financial Performance Trends (Line Area Chart):** Revenue, spend, profit trends over time.
  - *Rationale*: Enables monitoring of financial cycles, tests correlation between spend increases and revenue/profit changes.

- **Platform Unit Economics (Table):** CPA, ROAS, margin at channel level.
  - *Rationale*: Provides financial efficiency insights for each channel, empowering deliberate, data-driven scaling or cutbacks.

- **Regional ROI Performance (Horizontal Bar):** ROI by region.
  - *Rationale*: Illuminates geographic strengths/weaknesses, guiding expansion or contraction of local marketing.

- **Platform × Region Profit Heatmap:** Matrix of profit by channel/region pair.
  - *Rationale*: Visual heatmaps highlight synergy or drag between market and channel—impossible to see in flat tables.

- **Budget Efficiency Analysis**
  - **Scatter (Budget Allocation vs ROI by Objective):** Plots spend share and ROI per campaign objective.
    - *Rationale*: Surfaces which objectives deliver both scale and efficiency, driving overall strategy.
  - **Table (Cost Efficiency):** Aggregates CPC, CPM, CAC, and other ratios by platform.
    - *Rationale*: Pinpoints where financial inefficiencies arise and where cost control should be instituted.
  - **Profit Margin by Platform (Bar):** Margin for each channel.
    - *Rationale*: Detects high (or negative) margin contributors rapidly.

- **Comprehensive Platform Financial Summary (Table):** Detailed table with all major KPIs and gradients.
  - *Rationale*: Facilitates comparison of overall health across platforms in a single glance for the CFO.

- **Spend and Profit Share by Platform (Pie Charts):** Visual share-of-spend and profit.
  - *Rationale*: Highlights alignment (or mismatch) between where money is spent vs. where profit accrues. Drives optimization.

- **Advanced Financial Metrics (Table):** Profit per Conversion, LTV/CAC, Payback Period, highlighted by gradient and with insight callout.
  - *Rationale*: Moves CFO’s attention beyond averages to nuanced, segment-by-segment and market-by-market efficiency (e.g., identifying “hidden” pockets of ROI).

***

## 5. Design Rationale Summary

Every chart and table was purposefully chosen:
- To map directly to a stakeholder’s core business objective (awareness, engagement, conversion, profit).
- To allow information comparison, trend detection, or easy identification of actionable outliers.
- To maximize interpretability and drive clear, data-backed next steps for each stakeholder independently.

***

## 6. Technical Highlights

- **Data caching for speed**, advanced aggregation functions (e.g., weighted ROAS), and sectioned layouts for readability.
- **Responsive layout and cohesive visual styling** for professional delivery.
- **Error handling and context-aware insights** ensure usable feedback with any filter state.

***

## 7. Limitations & Opportunities

- Real data integration, mobile-first optimization, and forecast/anomaly modules could further increase business value in the future.

***
