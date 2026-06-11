import streamlit as st
import pandas as pd
import numpy as np
import os, json, pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Farmer Decision Intelligence — AP",
    page_icon="🌾",
    layout="wide"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean font + spacing */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* KPI card */
    .kpi-card {
        background: #1a1f2e;
        border: 1px solid #2a3044;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
    }
    .kpi-label { font-size: 12px; color: #8892a4; text-transform: uppercase; letter-spacing: 0.8px; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #ffffff; margin-top: 4px; }
    .kpi-sub   { font-size: 11px; color: #5a6478; margin-top: 2px; }

    /* Section header */
    .section-title {
        font-size: 13px; font-weight: 600;
        color: #8892a4; text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 12px;
        border-bottom: 1px solid #2a3044; padding-bottom: 6px;
    }

    /* Risk badge */
    .badge-high   { background:#3d1515; color:#ff6b6b; border:1px solid #ff6b6b;
                    padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-medium { background:#3d2a10; color:#f9a825; border:1px solid #f9a825;
                    padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-low    { background:#0d2e1a; color:#4caf50; border:1px solid #4caf50;
                    padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

    /* Simulator result card */
    .result-card {
        border-radius: 12px; padding: 24px;
        border-width: 1px; border-style: solid;
    }
    .result-prob { font-size: 52px; font-weight: 800; line-height: 1; }
    .result-label { font-size: 13px; color: #8892a4; margin-bottom: 4px; }
    .result-rec {
        background: rgba(255,255,255,0.04);
        border-radius: 8px; padding: 14px;
        font-size: 14px; color: #c8d0de; line-height: 1.6;
        margin-top: 16px;
    }

    /* Hide streamlit default header junk */
    #MainMenu, footer { visibility: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1f2e; border-radius: 8px 8px 0 0;
        padding: 8px 20px; color: #8892a4; font-size: 13px;
    }
    .stTabs [aria-selected="true"] { background: #252c3f; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path   = "dashboard_output.csv"
    model_path = "model_assets.pkl"
    json_path  = "model_performance.json"
    if not all(os.path.exists(p) for p in [csv_path, model_path, json_path]):
        with st.spinner("Running ML pipeline for the first time..."):
            try:
                import AP_Crop_Switching
                AP_Crop_Switching.run_pipeline()
            except Exception as e:
                st.error(f"Pipeline error: {e}"); st.stop()
    return pd.read_csv(csv_path)

df = load_data()

metrics      = json.load(open("model_performance.json")) if os.path.exists("model_performance.json") else None
model_assets = pickle.load(open("model_assets.pkl","rb")) if os.path.exists("model_assets.pkl") else None

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c8d0de", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor="#2a3044", linecolor="#2a3044"),
    yaxis=dict(gridcolor="#2a3044", linecolor="#2a3044"),
)
COLORS = ["#4c8eff","#ff6b6b","#4caf50","#f9a825","#ab47bc"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌾 Farmer Decision Intelligence")
    st.markdown("---")
    st.markdown('<p class="section-title">Filter Data</p>', unsafe_allow_html=True)

    sel_district = st.selectbox("District",  ["All"] + sorted(df["District_Name"].unique()))
    sel_year     = st.selectbox("Year",      ["All"] + sorted(df["Crop_Year"].unique()))
    sel_season   = st.selectbox("Season",    ["All"] + sorted(df["Season"].unique()))
    sel_risk     = st.selectbox("Risk Level",["All","High","Medium","Low"])

    st.markdown("---")
    if st.button("⚡ Re-run ML Pipeline", width="stretch"):
        with st.spinner("Re-running..."):
            try:
                import importlib, AP_Crop_Switching
                importlib.reload(AP_Crop_Switching)
                AP_Crop_Switching.run_pipeline()
                st.cache_data.clear()
                st.success("Done! Refresh page.")
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.caption("Data: GoI Agriculture & Rainfall\nCoverage: AP districts, 1997–2014")

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df.copy()
if sel_district != "All": filtered = filtered[filtered["District_Name"] == sel_district]
if sel_year     != "All": filtered = filtered[filtered["Crop_Year"]     == int(sel_year)]
if sel_season   != "All": filtered = filtered[filtered["Season"]        == sel_season]
if sel_risk     != "All": filtered = filtered[filtered["Risk_Level"]    == sel_risk]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## Farmer Decision Intelligence")
st.markdown('<p style="color:#8892a4;margin-top:-12px;margin-bottom:20px;">Andhra Pradesh Crop Switching Risk Platform &nbsp;·&nbsp; 1997–2014</p>', unsafe_allow_html=True)

if filtered.empty:
    st.warning("No records match the current filters."); st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────────
switching_rate = filtered["Switched"].mean() * 100
avg_prob       = filtered["Switch_Prob"].mean()
high_risk_n    = (filtered["Risk_Level"] == "High").sum()
total_n        = len(filtered)

k1, k2, k3, k4 = st.columns(4)
for col, label, val, sub in [
    (k1, "Total Records",     f"{total_n:,}",          "filtered view"),
    (k2, "Switching Rate",    f"{switching_rate:.1f}%", "of records switched"),
    (k3, "Avg Switch Prob",   f"{avg_prob:.2f}",        "model probability"),
    (k4, "High Risk Records", f"{high_risk_n:,}",       "prob > 0.70"),
]:
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮  Simulator", "📊  Overview", "🗺  Districts", "🤖  Model", "📋  Data"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DECISION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not model_assets:
        st.warning("Model not found. Run the pipeline first via the sidebar button.")
    else:
        model   = model_assets["model"]
        le_dist = model_assets["le_district"]
        le_seas = model_assets["le_season"]
        features= model_assets["features"]

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<p class="section-title">Input Parameters</p>', unsafe_allow_html=True)

            sim_district = st.selectbox("District", sorted(le_dist.classes_), key="sd")
            sim_season   = st.selectbox("Season",   sorted(le_seas.classes_), key="ss")

            subset = df[(df["District_Name"] == sim_district) & (df["Season"] == sim_season)]
            if subset.empty: subset = df[df["District_Name"] == sim_district]
            d_area = float(subset["Area"].mean())       if not subset.empty else 15000.0
            d_prod = float(subset["Production"].mean()) if not subset.empty else 30000.0
            if np.isnan(d_area) or d_area <= 0: d_area = 1000.0
            if np.isnan(d_prod) or d_prod <= 0: d_prod = 2000.0

            sim_area = st.number_input("Cultivated Area (ha)",       min_value=1.0, value=d_area, step=500.0)
            sim_prod = st.number_input("Expected Production (MT)",   min_value=1.0, value=d_prod, step=1000.0)

            dist_rows   = df[df["District_Name"] == sim_district]
            normal_rain = float(dist_rows.iloc[0]["annual_rainfall"] - dist_rows.iloc[0]["rainfall_deviation"]) \
                          if not dist_rows.empty else 950.0
            st.caption(f"Historical avg rainfall for {sim_district}: **{normal_rain:.0f} mm**")
            sim_rain = st.slider("Expected Rainfall (mm)", 100.0, 2500.0, normal_rain, 10.0)

        with right:
            st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

            # Build feature vector
            dist_enc  = le_dist.transform([sim_district])[0]
            seas_enc  = le_seas.transform([sim_season])[0]
            yield_val = sim_prod / sim_area
            rain_dev  = sim_rain - normal_rain

            # Build all 16 features
            area_log   = np.log1p(sim_area)
            prod_log   = np.log1p(sim_prod)
            yield_log  = np.log1p(yield_val)
            axr        = sim_area  * sim_rain
            yxr        = yield_val * rain_dev
            # For lag features use district historical averages as proxy
            prev_area  = float(subset["Area"].mean())  if not subset.empty else sim_area
            prev_yield = float(subset["Yield"].mean()) if not subset.empty else yield_val
            area_chg   = sim_area  - prev_area
            yield_chg  = yield_val - prev_yield
            if np.isnan(prev_area):  prev_area  = sim_area
            if np.isnan(prev_yield): prev_yield = yield_val
            if np.isnan(area_chg):   area_chg   = 0.0
            if np.isnan(yield_chg):  yield_chg  = 0.0

            feat_map = {
                "Area": sim_area, "Production": sim_prod, "Yield": yield_val,
                "annual_rainfall": sim_rain, "rainfall_deviation": rain_dev,
                "District_enc": dist_enc, "Season_enc": seas_enc,
                "Area_log": area_log, "Production_log": prod_log, "Yield_log": yield_log,
                "Area_x_Rain": axr, "Yield_x_Rain": yxr,
                "Prev_Area": prev_area, "Prev_Yield": prev_yield,
                "Area_change": area_chg, "Yield_change": yield_chg,
            }
            sim_df = pd.DataFrame([[feat_map.get(f, 0) for f in features]], columns=features)
            prob   = float(model.predict_proba(sim_df)[0][1])

            # Expert guardrails
            if   sim_rain < 0.70 * normal_rain: prob = max(prob, 0.75)
            elif sim_rain < 0.85 * normal_rain: prob = max(prob, 0.50)
            elif sim_rain > 1.40 * normal_rain: prob = max(prob, 0.55)

            # Result card
            if prob > 0.7:
                bg, border, tc = "#1f0a0a", "#ff6b6b", "#ff6b6b"
                badge  = '<span class="badge-high">HIGH RISK</span>'
                rec    = "⚠️ High switching probability. Consider drought-resistant crop varieties or water conservation measures immediately."
            elif prob > 0.4:
                bg, border, tc = "#1f1505", "#f9a825", "#f9a825"
                badge  = '<span class="badge-medium">MEDIUM RISK</span>'
                rec    = "📋 Moderate risk. Monitor rainfall deviation trends. Keep backup irrigation plan ready."
            else:
                bg, border, tc = "#051a0a", "#4caf50", "#4caf50"
                badge  = '<span class="badge-low">LOW RISK</span>'
                rec    = "✅ Conditions stable. Crop continuity is highly probable for this district-season combination."

            st.markdown(f"""
            <div class="result-card" style="background:{bg};border-color:{border};">
                <div style="margin-bottom:12px">{badge}</div>
                <div class="result-label">Switching Probability</div>
                <div class="result-prob" style="color:{tc}">{prob*100:.1f}%</div>
                <div class="result-rec">{rec}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge chart using Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 28, "color": tc}},
                gauge={
                    "axis":      {"range": [0, 100], "tickcolor": "#8892a4"},
                    "bar":       {"color": tc},
                    "bgcolor":   "#1a1f2e",
                    "bordercolor": "#2a3044",
                    "steps": [
                        {"range": [0,  40], "color": "#0d2e1a"},
                        {"range": [40, 70], "color": "#3d2a10"},
                        {"range": [70,100], "color": "#3d1515"},
                    ],
                    "threshold": {"line": {"color": tc, "width": 3}, "value": prob * 100},
                },
            ))
            fig_gauge.update_layout(
                height=180, margin=dict(l=20,r=20,t=20,b=10),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8d0de")
            )
            st.plotly_chart(fig_gauge, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    r1c1, r1c2 = st.columns(2, gap="medium")

    with r1c1:
        st.markdown('<p class="section-title">Switching Events by Year</p>', unsafe_allow_html=True)
        yearly = filtered.groupby("Crop_Year")["Switched"].sum().reset_index()
        fig = px.line(yearly, x="Crop_Year", y="Switched",
                      markers=True, color_discrete_sequence=["#4c8eff"])
        fig.update_traces(line_width=2.5, marker_size=7)
        fig.update_layout(**PLOT_THEME, height=280,
                          xaxis_title="Year", yaxis_title="Switches")
        st.plotly_chart(fig, width="stretch")

    with r1c2:
        st.markdown('<p class="section-title">Risk Level Distribution</p>', unsafe_allow_html=True)
        risk_counts = filtered["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        color_map   = {"High": "#ff6b6b", "Medium": "#f9a825", "Low": "#4caf50"}
        fig = px.pie(risk_counts, names="Risk", values="Count",
                     color="Risk", color_discrete_map=color_map, hole=0.55)
        fig.update_traces(textposition="outside", textfont_size=12)
        fig.update_layout(**PLOT_THEME, height=280, showlegend=True,
                          legend=dict(orientation="v", x=1.05))
        st.plotly_chart(fig, width="stretch")

    st.markdown('<p class="section-title">Most Switched-Away Crops (Top 10)</p>', unsafe_allow_html=True)
    switched = filtered[filtered["Switched"] == 1]["Crop"].value_counts().head(10).reset_index()
    switched.columns = ["Crop", "Count"]
    if switched.empty:
        st.info("No switching events in the current filter.")
    else:
        fig = px.bar(switched, x="Count", y="Crop", orientation="h",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(**PLOT_THEME, height=320, coloraxis_showscale=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRICT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Switching Frequency by District</p>', unsafe_allow_html=True)
    dist_data = filtered.groupby("District_Name").agg(
        Switches       = ("Switched",     "sum"),
        Switching_Rate = ("Switched",     "mean"),
        Avg_Prob       = ("Switch_Prob",  "mean"),
    ).reset_index().sort_values("Switches", ascending=False)

    fig = px.bar(dist_data, x="District_Name", y="Switches",
                 color="Switching_Rate", color_continuous_scale="RdYlGn_r",
                 hover_data={"Switching_Rate": ":.1%", "Avg_Prob": ":.2f"})
    fig.update_layout(**PLOT_THEME, height=300,
                      xaxis_title="", yaxis_title="Total Switches",
                      coloraxis_colorbar=dict(title="Switch Rate", tickformat=".0%"))
    st.plotly_chart(fig, width="stretch")

    d1, d2 = st.columns(2, gap="medium")

    with d1:
        st.markdown('<p class="section-title">Rainfall vs Switching Behaviour</p>', unsafe_allow_html=True)
        rain_df = filtered.copy()
        rain_df["Switched_Label"] = rain_df["Switched"].map({0:"No Switch", 1:"Switched"})
        fig = px.box(rain_df, x="Switched_Label", y="annual_rainfall",
                     color="Switched_Label",
                     color_discrete_map={"No Switch":"#4caf50","Switched":"#ff6b6b"})
        fig.update_layout(**PLOT_THEME, height=280,
                          xaxis_title="", yaxis_title="Annual Rainfall (mm)",
                          showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with d2:
        st.markdown('<p class="section-title">Top 10 Highest-Risk Records</p>', unsafe_allow_html=True)
        top_risk = filtered.sort_values("Switch_Prob", ascending=False).head(10)
        fig = px.bar(top_risk, x="Switch_Prob", y="District_Name",
                     orientation="h", color="Switch_Prob",
                     color_continuous_scale="Reds",
                     hover_data={"Crop_Year": True, "Crop": True})
        fig.update_layout(**PLOT_THEME, height=280,
                          xaxis_title="Switch Probability", yaxis_title="",
                          coloraxis_showscale=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Model Comparison</p>', unsafe_allow_html=True)
    st.caption("All models trained with enhanced features (log transforms, lag features, interaction terms). CV metric is F1 — appropriate for imbalanced dataset (~20% switching rate).")

    if metrics and "models" in metrics:
        model_df_display = pd.DataFrame(metrics["models"])
    else:
        model_df_display = pd.DataFrame({
            "Model":          ["Gradient Boosting","Random Forest","Decision Tree","Logistic Regression","KNN"],
            "Test Accuracy":  [0.9262, 0.8770, 0.8525, 0.6393, 0.7623],
            "F1 Score":       [0.8235, 0.6939, 0.6786, 0.3889, 0.2564],
            "CV F1 (5-fold)": [0.7274, 0.6918, 0.5187, 0.4693, 0.3314],
        })

    m1, m2 = st.columns(2, gap="medium")
    with m1:
        fig = go.Figure()
        metrics_cols = [c for c in model_df_display.columns if c != "Model"]
        bar_colors   = ["#4c8eff","#ff6b6b","#4caf50"]
        for i, col in enumerate(metrics_cols):
            fig.add_trace(go.Bar(
                name=col, x=model_df_display["Model"],
                y=model_df_display[col],
                marker_color=bar_colors[i % len(bar_colors)]
            ))
        fig.update_layout(**PLOT_THEME, height=300, barmode="group",
                          legend=dict(orientation="h", y=1.12))
        fig.update_yaxes(range=[0,1])
        st.plotly_chart(fig, width="stretch")

    with m2:
        st.markdown('<p class="section-title">Switch Probability Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(filtered, x="Switch_Prob", nbins=30,
                           color_discrete_sequence=["#ab47bc"])
        fig.add_vline(x=0.4, line_dash="dash", line_color="#f9a825",
                      annotation_text="Medium (0.4)", annotation_font_color="#f9a825")
        fig.add_vline(x=0.7, line_dash="dash", line_color="#ff6b6b",
                      annotation_text="High (0.7)", annotation_font_color="#ff6b6b")
        fig.update_layout(**PLOT_THEME, height=300,
                          xaxis_title="Probability", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")

    st.markdown('<p class="section-title">Feature Importance — Gradient Boosting</p>', unsafe_allow_html=True)
    if metrics and "importances" in metrics:
        feat_df = pd.DataFrame(metrics["importances"]).sort_values("Importance", ascending=True)
    else:
        feat_df = pd.DataFrame({
            "Feature":    ["Area_change","Prev_Yield","Yield_log","Area_log","Production_log",
                           "Yield_x_Rain","Area_x_Rain","Area","Yield","Production",
                           "annual_rainfall","rainfall_deviation","District_enc","Season_enc","Prev_Area"],
            "Importance": [0.18,0.14,0.12,0.11,0.10,0.08,0.07,0.06,0.05,0.05,
                           0.05,0.04,0.03,0.03,0.02]
        }).sort_values("Importance", ascending=True)

    fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Viridis")
    fig.update_layout(
        **PLOT_THEME,
        height=400,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, width="stretch")

    # Key insights banner if available
    if metrics and "insights" in metrics:
        ins = metrics["insights"]
        st.markdown("---")
        st.markdown('<p class="section-title">Key Findings</p>', unsafe_allow_html=True)
        i1,i2,i3,i4 = st.columns(4)
        for col, label, val in [
            (i1, "Most Volatile District", ins["most_volatile_district"]),
            (i2, "Most Stable District",   ins["most_stable_district"]),
            (i3, "Most Switched Crop",     ins["most_switched_away_crop"]),
            (i4, "Best Model",             f"{ins['best_model']} (CV F1: {ins['best_model_cv']:.2f})"),
        ]:
            col.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="font-size:18px">{val}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Filtered Records</p>', unsafe_allow_html=True)
    display_cols = ["Crop_Year","District_Name","Crop","Season",
                    "annual_rainfall","Switch_Prob","Risk_Level","Recommendation"]

    # Colour-coded Risk_Level
    def colour_risk(val):
        colours = {"High":"background-color:#3d1515;color:#ff6b6b",
                   "Medium":"background-color:#3d2a10;color:#f9a825",
                   "Low":"background-color:#0d2e1a;color:#4caf50"}
        return colours.get(val,"")

    styled = (filtered[display_cols]
              .sort_values("Switch_Prob", ascending=False)
              .style.map(colour_risk, subset=["Risk_Level"])
              .format({"Switch_Prob": "{:.3f}"}))
    st.dataframe(styled, width="stretch", height=500)

    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", csv_bytes, "filtered_data.csv", "text/csv")

st.markdown("---")
st.caption("Farmer Decision Intelligence · VINS Internship — IIT Ropar · Data: Government of India (1997–2014)")
