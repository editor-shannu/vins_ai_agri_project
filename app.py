import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle

st.set_page_config(
    page_title="AP Farmer Decision Intelligence",
    page_icon="🌾",
    layout="wide"
)

@st.cache_data
def load_data():
    csv_path = "dashboard_output.csv"
    model_path = "model_assets.pkl"
    json_path = "model_performance.json"
    
    # Auto-run pipeline if any core asset is missing
    if not os.path.exists(csv_path) or not os.path.exists(model_path) or not os.path.exists(json_path):
        with st.spinner("Initializing: Processing datasets and training models in the background..."):
            try:
                import AP_Crop_Switching
                AP_Crop_Switching.run_pipeline()
            except Exception as e:
                st.error(f"Error executing backend pipeline on startup: {e}")
                st.stop()
    return pd.read_csv(csv_path)

df = load_data()

# Load dynamic model performance metrics if available
metrics = None
if os.path.exists("model_performance.json"):
    try:
        with open("model_performance.json", "r") as f:
            metrics = json.load(f)
    except Exception:
        pass

# Load model assets for simulator
model_assets = None
if os.path.exists("model_assets.pkl"):
    try:
        with open("model_assets.pkl", "rb") as f:
            model_assets = pickle.load(f)
    except Exception:
        pass

# 2. Main Filters
st.sidebar.title("🌾 Filters")
districts = ["All"] + sorted(df["District_Name"].unique().tolist())
sel_district = st.sidebar.selectbox("District", districts)
years = ["All"] + sorted(df["Crop_Year"].unique().tolist())
sel_year = st.sidebar.selectbox("Year", years)
seasons = ["All"] + sorted(df["Season"].unique().tolist())
sel_season = st.sidebar.selectbox("Season", seasons)
sel_risk = st.sidebar.selectbox("Risk Level", ["All", "High", "Medium", "Low"])

filtered = df.copy()
if sel_district != "All":
    filtered = filtered[filtered["District_Name"] == sel_district]
if sel_year != "All":
    filtered = filtered[filtered["Crop_Year"] == int(sel_year)]
if sel_season != "All":
    filtered = filtered[filtered["Season"] == sel_season]
if sel_risk != "All":
    filtered = filtered[filtered["Risk_Level"] == sel_risk]

st.title("🌾 Farmer Decision Intelligence Platform")
st.caption("Predicting Crop Switching Behaviour & Minimizing Risk in Andhra Pradesh")
st.markdown("---")

# 3. Premium Agricultural Intelligence Briefing Banner
if metrics and "insights" in metrics:
    insights = metrics["insights"]
    st.markdown(
        f"""
        <div style="background-color: #112233; padding: 22px; border-radius: 12px; border-left: 6px solid #2ecc71; margin-bottom: 25px; color: #ffffff; font-family: sans-serif;">
            <h4 style="margin: 0 0 12px 0; color: #2ecc71; font-weight: bold; letter-spacing: 0.5px;">📋 STATEWIDE AGRICULTURAL INTELLIGENCE BRIEFING</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div>
                    <span style="font-size: 13px; color: #88aadd;">Overall Switching Rate</span><br/>
                    <strong style="font-size: 22px; color: #ffffff;">{insights['overall_switching_rate']:.1f}%</strong>
                </div>
                <div>
                    <span style="font-size: 13px; color: #88aadd;">Most Volatile District</span><br/>
                    <strong style="font-size: 22px; color: #e74c3c;">{insights['most_volatile_district']}</strong>
                </div>
                <div>
                    <span style="font-size: 13px; color: #88aadd;">Most Stable District</span><br/>
                    <strong style="font-size: 22px; color: #2ecc71;">{insights['most_stable_district']}</strong>
                </div>
                <div>
                    <span style="font-size: 13px; color: #88aadd;">Most Switched-Away Crop</span><br/>
                    <strong style="font-size: 22px; color: #f39c12;">{insights['most_switched_away_crop']}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if filtered.empty:
    st.warning("No records match the selected filters. Try a different combination.")
    st.stop()

# Key KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records",     f"{len(filtered):,}")
c2.metric("Switching Rate",    f"{filtered['Switched'].mean()*100:.1f}%")
c3.metric("Avg Switch Prob",   f"{filtered['Switch_Prob'].mean():.2f}")
c4.metric("High Risk Records", f"{(filtered['Risk_Level']=='High').sum():,}")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Decision Simulator", "📊 Overview", "🗺 District Analysis", "🤖 Model Insights", "📋 Data Table"
])

# Tab 1: Live Decision Simulator (Farmer Tool)
with tab1:
    st.subheader("🔮 Interactive Decision Intelligence Simulator")
    st.markdown("Select your district, crop, and expected environmental metrics below. The platform will predict the crop switching risk probability and provide customized recommendations.")
    
    if not model_assets:
        st.warning("Prediction model not found. Please click '⚡ Run ML Pipeline' in the sidebar first to train and export the model assets.")
    else:
        # Load encoders & model
        model = model_assets['model']
        le_dist = model_assets['le_district']
        le_seas = model_assets['le_season']
        features = model_assets['features']
        
        sim_col1, sim_col2 = st.columns([1, 1])
        
        with sim_col1:
            st.markdown("#### **Farmer Input Parameters**")
            
            # Select inputs based on encoder values
            sim_district = st.selectbox("Select District", sorted(le_dist.classes_), key="sim_dist")
            sim_season = st.selectbox("Select Season", sorted(le_seas.classes_), key="sim_seas")
            
            # Calculate dynamic defaults from actual historical data
            subset = df[(df["District_Name"] == sim_district) & (df["Season"] == sim_season)]
            if subset.empty:
                subset = df[df["District_Name"] == sim_district]
                
            if not subset.empty:
                default_area = float(round(subset["Area"].mean(), 2))
                default_prod = float(round(subset["Production"].mean(), 2))
            else:
                default_area = 15000.0
                default_prod = 30000.0
                
            if np.isnan(default_area) or default_area <= 0:
                default_area = 1000.0
            if np.isnan(default_prod) or default_prod <= 0:
                default_prod = 2000.0
                
            # Area and production inputs
            sim_area = st.number_input("Cultivated Area (Hectares)", min_value=1.0, value=default_area, step=500.0)
            sim_production = st.number_input("Expected Production (Metric Tons)", min_value=1.0, value=default_prod, step=1000.0)
            
            # Normal rainfall indicator
            dist_rows = df[df["District_Name"] == sim_district]
            if not dist_rows.empty:
                normal_rain = float(dist_rows.iloc[0]["annual_rainfall"] - dist_rows.iloc[0]["rainfall_deviation"])
            else:
                normal_rain = 950.0
                
            st.info(f"Normal Average Rainfall for {sim_district}: **{normal_rain:.1f} mm**")
            sim_rainfall = st.slider("Expected Rainfall (mm)", min_value=100.0, max_value=2500.0, value=normal_rain, step=10.0)
            
        with sim_col2:
            st.markdown("#### **Switching Prediction & Recommendation**")
            
            # 1. Transform inputs
            dist_enc = le_dist.transform([sim_district])[0]
            seas_enc = le_seas.transform([sim_season])[0]
            yield_val = sim_production / sim_area
            rain_dev = sim_rainfall - normal_rain
            
            # 2. Build input dataframe matching model features
            sim_df = pd.DataFrame([{
                'Area': sim_area,
                'Production': sim_production,
                'Yield': yield_val,
                'annual_rainfall': sim_rainfall,
                'rainfall_deviation': rain_dev,
                'District_enc': dist_enc,
                'Season_enc': seas_enc
            }])
            
            # 3. Run prediction
            prob = float(model.predict_proba(sim_df[features])[0][1])
            
            # Apply Expert Heuristic Guardrails (Decision Intelligence Rules)
            # A) Severe drought override (Deficit > 30% from normal average)
            if sim_rainfall < 0.7 * normal_rain:
                prob = max(prob, 0.75)
            # B) Moderate drought override (Deficit > 15% from normal average)
            elif sim_rainfall < 0.85 * normal_rain:
                prob = max(prob, 0.50)
            # C) Flood / Excess rainfall override (Excess > 40% from normal average)
            elif sim_rainfall > 1.4 * normal_rain:
                prob = max(prob, 0.55)
            
            # 4. Display card based on risk level
            if prob > 0.7:
                risk_title = "HIGH SWITCHING RISK"
                bg_color = "#331111"
                text_color = "#ff4c4c"
                border_color = "#ff4c4c"
                rec = "High switching probability detected. Ensure local water conservation or switch to heat/drought resistant crops immediately."
            elif prob > 0.4:
                risk_title = "MEDIUM SWITCHING RISK"
                bg_color = "#332211"
                text_color = "#f39c12"
                border_color = "#f39c12"
                rec = "Moderate risk. Monitor rainfall deviations. Ensure backup irrigation facilities are active."
            else:
                risk_title = "LOW SWITCHING RISK"
                bg_color = "#113311"
                text_color = "#2ecc71"
                border_color = "#2ecc71"
                rec = "Agricultural conditions are stable. Farmer crop continuity is highly probable."

            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 25px; border-radius: 12px; border: 2px solid {border_color}; margin-top: 15px; font-family: sans-serif;">
                    <span style="font-size: 12px; letter-spacing: 1.5px; color: #a0a0a0; font-weight: bold; text-transform: uppercase;">Prediction Result</span>
                    <h2 style="margin: 5px 0 15px 0; color: {text_color}; font-weight: bold;">{risk_title}</h2>
                    <div style="margin-bottom: 20px;">
                        <span style="font-size: 14px; color: #d0d0d0;">Switching Probability:</span>
                        <span style="font-size: 20px; font-weight: bold; color: {text_color}; float: right;">{prob*100:.1f}%</span>
                    </div>
                    <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;">
                        <strong style="font-size: 13px; color: #ffffff; text-transform: uppercase;">Tailored Advisor Recommendation:</strong><br/>
                        <p style="margin: 8px 0 0 0; font-size: 15px; color: #e0e0e0; line-height: 1.4;">{rec}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Visual probability gauge using progress bar
            st.markdown(" ")
            st.write("Switching Probability Gauge:")
            st.progress(prob)

# Tab 2: Overview
with tab1:
    pass # Managed via streamlit tabs below

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Year-wise Switching Trend")
        yearly = filtered.groupby("Crop_Year")["Switched"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(yearly["Crop_Year"], yearly["Switched"], marker="o", color="#3498db", linewidth=2)
        ax.set_xlabel("Year"); ax.set_ylabel("Total Switches")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Risk Level Distribution")
        risk_counts = filtered["Risk_Level"].value_counts()
        color_map  = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
        pie_colors = [color_map.get(r, "grey") for r in risk_counts.index]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%",
               colors=pie_colors, startangle=140)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Most Switched-Away Crops")
    switched_crops = filtered[filtered["Switched"] == 1]["Crop"].value_counts().head(10)
    if switched_crops.empty:
        st.info("No switching events in the current filter.")
    else:
        fig, ax = plt.subplots(figsize=(11, 3.5))
        switched_crops.plot(kind="bar", ax=ax, color="coral")
        ax.set_xlabel("Crop"); ax.set_ylabel("Times Switched Away")
        ax.set_xticklabels(switched_crops.index, rotation=35, ha="right")
        fig.tight_layout(); st.pyplot(fig); plt.close()

with tab3:
    st.subheader("District-wise Crop Switching Frequency")
    dist_switch = filtered.groupby("District_Name")["Switched"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    dist_switch.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("District"); ax.set_ylabel("Switches")
    ax.set_xticklabels(dist_switch.index, rotation=45, ha="right")
    fig.tight_layout(); st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rainfall vs Switching")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="Switched", y="annual_rainfall", data=filtered,
                    palette=["#2ecc71", "#e74c3c"], ax=ax)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No Switch", "Switched"])
        ax.set_xlabel(""); ax.set_ylabel("Annual Rainfall (mm)")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Top 10 High-Risk Combinations")
        top_risk = filtered.sort_values("Switch_Prob", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=top_risk, x="Switch_Prob", y="District_Name",
                    palette="Reds_r", ax=ax)
        ax.set_xlabel("Switching Probability"); ax.set_ylabel("")
        fig.tight_layout(); st.pyplot(fig); plt.close()

with tab4:
    st.subheader("Model Performance Summary")
    if metrics and "models" in metrics:
        st.caption("Actual model performance metrics loaded from current run:")
        model_results = pd.DataFrame(metrics["models"])
        model_results.columns = [c.replace("_", " ").title() for c in model_results.columns]
        st.dataframe(model_results.set_index("Model"), use_container_width=True)
    else:
        st.caption("Reference values:")
        model_results = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "KNN", "Decision Tree"],
            "Test Accuracy":        [0.8000, 0.8000, 0.7615, 0.7923],
            "F1 Score":             [0.1875, 0.3810, 0.3111, 0.5263],
            "CV Accuracy (5-fold)": [0.7874, 0.7535, 0.7196, 0.6103],
        })
        st.dataframe(model_results.set_index("Model"), use_container_width=True)

    st.subheader("Switching Probability Distribution")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(filtered["Switch_Prob"], bins=30, color="#9b59b6", edgecolor="white")
    ax.axvline(0.4, color="orange", linestyle="--", label="Medium threshold (0.4)")
    ax.axvline(0.7, color="red",    linestyle="--", label="High threshold (0.7)")
    ax.set_xlabel("Switch Probability"); ax.set_ylabel("Count")
    ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Feature Importance (Random Forest)")
    if metrics and "importances" in metrics:
        st.caption("Actual feature importances loaded from current run:")
        feat_df = pd.DataFrame(metrics["importances"]).sort_values("Importance", ascending=False)
    else:
        st.caption("Reference values:")
        feat_df = pd.DataFrame({
            "Feature":    ["Area", "Production", "Yield", "rainfall_deviation", "annual_rainfall", "District_enc", "Season_enc"],
            "Importance": [0.2291, 0.1776, 0.1715, 0.1261, 0.1107, 0.0960, 0.0886]
        }).sort_values("Importance", ascending=False)
        
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Random Forest — Feature Importance")
    fig.tight_layout(); st.pyplot(fig); plt.close()

with tab5:
    st.subheader("Filtered Dashboard Records")
    display_cols = ["Crop_Year", "District_Name", "Crop", "Season",
                    "annual_rainfall", "Switch_Prob", "Risk_Level", "Recommendation"]
    st.dataframe(
        filtered[display_cols].sort_values("Switch_Prob", ascending=False),
        use_container_width=True, height=500
    )
    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Filtered Data as CSV", csv_bytes,
                       "filtered_dashboard.csv", "text/csv")

st.markdown("---")
st.caption("Project: Farmer Decision Intelligence | IIT Ropar Agri Internship | Data: Government of India (1997–2014)")
