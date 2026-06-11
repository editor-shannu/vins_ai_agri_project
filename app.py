import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

st.set_page_config(
    page_title="Farmer Decision Intelligence — AP",
    page_icon="🌾",
    layout="wide"
)

# Helper function to run the ML pipeline dynamically
def trigger_pipeline_run():
    with st.spinner("Running machine learning pipeline... This might take 10-15 seconds."):
        try:
            # Dynamically import and run the pipeline
            import AP_Crop_Switching
            # Force reload the module in case it changed or needs re-execution
            import importlib
            importlib.reload(AP_Crop_Switching)
            AP_Crop_Switching.run_pipeline()
            st.success("Pipeline executed successfully! Data and models updated.")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error executing pipeline: {e}")

@st.cache_data
def load_data():
    path = "dashboard_output.csv"
    if not os.path.exists(path):
        with st.spinner("Initializing: Processing datasets and training models in the background..."):
            try:
                import AP_Crop_Switching
                AP_Crop_Switching.run_pipeline()
            except Exception as e:
                st.error(f"Error executing backend pipeline on startup: {e}")
                st.stop()
    return pd.read_csv(path)

# 1. Dataset Management / Upload in Sidebar
st.sidebar.title("⚙️ Dataset Management")
with st.sidebar.expander("Update / Upload Datasets"):
    st.write("Upload new GoI csv files to recalculate predictions and retrain models.")
    
    # File uploaders
    crop_file = st.file_uploader("Upload Crop Production CSV", type=["csv"], key="crop")
    rain_annual_file = st.file_uploader("Upload Annual Rainfall CSV", type=["csv"], key="annual")
    rain_dist_file = st.file_uploader("Upload District Normal Rainfall CSV", type=["csv"], key="dist")
    
    # Save files if uploaded
    if st.button("⚡ Run ML Pipeline"):
        saved_any = False
        os.makedirs("datasets", exist_ok=True)
        
        if crop_file is not None:
            with open("datasets/crop_production.csv", "wb") as f:
                f.write(crop_file.getbuffer())
            st.info("Saved crop_production.csv")
            saved_any = True
            
        if rain_annual_file is not None:
            with open("datasets/rainfall_in_india_1901-2015.csv", "wb") as f:
                f.write(rain_annual_file.getbuffer())
            st.info("Saved rainfall_in_india_1901-2015.csv")
            saved_any = True
            
        if rain_dist_file is not None:
            with open("datasets/districtwise_rainfall_normal.csv", "wb") as f:
                f.write(rain_dist_file.getbuffer())
            st.info("Saved districtwise_rainfall_normal.csv")
            saved_any = True
            
        # Run the pipeline
        trigger_pipeline_run()

df = load_data()

# Load dynamic model performance metrics if available
metrics = None
if os.path.exists("model_performance.json"):
    try:
        with open("model_performance.json", "r") as f:
            metrics = json.load(f)
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

st.title("🌾 Farmer Decision Intelligence Dashboard")
st.caption("Predicting Crop Switching Behaviour in Andhra Pradesh (1997–2014) — Dynamic ML Dashboard")
st.markdown("---")

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

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🗺 District Analysis", "🤖 Model Insights", "📋 Data Table"
])

with tab1:
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

with tab2:
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

with tab3:
    st.subheader("Model Performance Summary")
    if metrics and "models" in metrics:
        st.caption("Actual model performance metrics loaded from current run:")
        model_results = pd.DataFrame(metrics["models"])
        # Format column names for clarity
        model_results.columns = [c.replace("_", " ").title() for c in model_results.columns]
        st.dataframe(model_results.set_index("Model"), use_container_width=True)
    else:
        st.caption("Reference values (run the pipeline locally to update):")
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
        st.caption("Reference values (run the pipeline locally to update):")
        feat_df = pd.DataFrame({
            "Feature":    ["Area", "Production", "Yield", "rainfall_deviation", "annual_rainfall", "District_enc", "Season_enc"],
            "Importance": [0.2291, 0.1776, 0.1715, 0.1261, 0.1107, 0.0960, 0.0886]
        }).sort_values("Importance", ascending=False)
        
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Random Forest — Feature Importance")
    fig.tight_layout(); st.pyplot(fig); plt.close()

with tab4:
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
