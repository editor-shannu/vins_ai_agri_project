import os
import warnings
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for matplotlib so it runs without GUI windows popping up
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

def run_pipeline():
    print("All libraries imported successfully")

    # ==========================================
    # Section 1 — Load Data
    # ==========================================
    print("\n--- Section 1: Loading Data ---")
    crop        = pd.read_csv('datasets/crop_production.csv')
    rain_annual = pd.read_csv('datasets/rainfall_in_india_1901-2015.csv')
    rain_dist   = pd.read_csv('datasets/districtwise_rainfall_normal.csv')

    print("Crop Production shape :", crop.shape)
    print("Annual Rainfall shape  :", rain_annual.shape)
    print("District Rainfall shape:", rain_dist.shape)

    # ==========================================
    # Section 2 — Data Cleaning & AP Filtering
    # ==========================================
    print("\n--- Section 2: Data Cleaning & AP Filtering ---")
    crop = crop[crop['State_Name'] == 'Andhra Pradesh'].copy()
    crop = crop[(crop['Crop_Year'] >= 1997) & (crop['Crop_Year'] <= 2014)]

    # Fix district names in crop to match rainfall data (prevents NaNs during merge)
    district_mapping = {
        'VISAKHAPATANAM': 'VISAKHAPATNAM',
        'KADAPA': 'KUDDAPAH',
        'SPSR NELLORE': 'NELLORE'
    }
    crop['District_Name'] = crop['District_Name'].replace(district_mapping)

    crop['Yield'] = crop['Production'] / crop['Area']
    crop.replace([np.inf, -np.inf], np.nan, inplace=True)
    crop.dropna(inplace=True)

    print("AP rows after cleaning:", len(crop))
    print("Districts:", crop['District_Name'].nunique())
    print("Crops    :", crop['Crop'].nunique())
    print("Years    :", sorted(crop['Crop_Year'].unique()))

    # ==========================================
    # Section 3 — Feature Engineering & Switching Label Creation
    # ==========================================
    print("\n--- Section 3: Feature Engineering & Switching Label ---")
    dominant = (
        crop.sort_values('Area', ascending=False)
        .groupby(['District_Name', 'Season', 'Crop_Year'])
        .first()
        .reset_index()
    )[['District_Name', 'Season', 'Crop_Year', 'Crop', 'Area', 'Production', 'Yield']]

    dominant = dominant.sort_values(['District_Name', 'Season', 'Crop_Year'])

    dominant['Prev_Crop'] = dominant.groupby(['District_Name', 'Season'])['Crop'].shift(1)
    dominant['Switched']  = (dominant['Crop'] != dominant['Prev_Crop']).astype(int)
    dominant.loc[dominant['Prev_Crop'].isna(), 'Switched'] = 0

    # Merge annual rainfall (Coastal AP subdivision)
    coastal = rain_annual[rain_annual['SUBDIVISION'] == 'COASTAL ANDHRA PRADESH']
    coastal = coastal[['YEAR', 'ANNUAL']].rename(columns={'ANNUAL': 'annual_rainfall'})
    dominant = dominant.merge(coastal, left_on='Crop_Year', right_on='YEAR', how='left')

    # Merge district average rainfall
    rain_dist.columns = rain_dist.columns.str.strip()
    dist_col   = [c for c in rain_dist.columns if 'DISTRICT' in c.upper()][0]
    annual_col = [c for c in rain_dist.columns if 'ANNUAL'   in c.upper()][0]
    rain_avg   = rain_dist[[dist_col, annual_col]].copy()
    rain_avg.columns = ['District_Name', 'district_avg_rainfall']

    dominant = dominant.merge(rain_avg, on='District_Name', how='left')
    dominant['rainfall_deviation'] = dominant['annual_rainfall'] - dominant['district_avg_rainfall']

    print("Final dataset shape:", dominant.shape)
    print("Switching rate: {:.1f}%".format(dominant['Switched'].mean() * 100))

    # ==========================================
    # Section 4 — Exploratory Data Analysis (EDA)
    # ==========================================
    print("\n--- Section 4: Performing EDA & Saving Plots ---")
    os.makedirs('plots', exist_ok=True)

    # Plot 1: District-wise switching frequency
    switch_freq = dominant.groupby('District_Name')['Switched'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    switch_freq.plot(kind='bar', color='steelblue')
    plt.title('District-wise Crop Switching Frequency (1997–2014)', fontsize=14)
    plt.xlabel('District'); plt.ylabel('Number of Switches')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/1_district_switching_frequency.png')
    plt.close()

    # Plot 2: Rainfall vs Switching
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Switched', y='annual_rainfall', data=dominant, palette=['#2ecc71', '#e74c3c'])
    plt.xticks([0, 1], ['No Switch', 'Switched'])
    plt.title('Annual Rainfall vs Crop Switching Decision', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/2_rainfall_vs_switching.png')
    plt.close()

    # Plot 3: Most switched-away crops
    crop_switch = dominant[dominant['Switched'] == 1]['Prev_Crop'].value_counts().head(10)
    plt.figure(figsize=(12, 5))
    crop_switch.plot(kind='bar', color='coral')
    plt.title('Top 10 Most Switched-Away Crops in AP', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/3_most_switched_away_crops.png')
    plt.close()

    # Plot 4: Year-wise switching trend
    yearly = dominant.groupby('Crop_Year')['Switched'].sum()
    plt.figure(figsize=(12, 5))
    plt.plot(yearly.index, yearly.values, marker='o', color='purple', linewidth=2)
    plt.title('Year-wise Crop Switching Trend in Andhra Pradesh', fontsize=14)
    plt.xlabel('Year'); plt.ylabel('Total Switches')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/4_yearly_switching_trend.png')
    plt.close()

    # Plot 5: Correlation heatmap
    corr_cols = ['Area', 'Production', 'Yield',
                 'annual_rainfall', 'district_avg_rainfall',
                 'rainfall_deviation', 'Switched']
    plt.figure(figsize=(9, 7))
    sns.heatmap(dominant[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/5_feature_correlation.png')
    plt.close()
    print("EDA plots saved in 'plots/' directory.")

    # ==========================================
    # Section 5 — Unsupervised Learning: K-Means Clustering
    # ==========================================
    print("\n--- Section 5: Unsupervised Learning (K-Means) ---")
    cluster_df = dominant.groupby('District_Name').agg(
        Switched        = ('Switched',        'mean'),
        Area            = ('Area',            'mean'),
        Production      = ('Production',      'mean'),
        annual_rainfall = ('annual_rainfall', 'mean')
    ).reset_index()

    X_cluster = cluster_df[['Switched', 'Area', 'Production', 'annual_rainfall']]
    from sklearn.preprocessing import StandardScaler
    scaler_c  = StandardScaler()
    X_scaled  = scaler_c.fit_transform(X_cluster)

    # Elbow method
    inertia = []
    for k in range(1, 11):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method — Optimal K', fontsize=14)
    plt.xlabel('K'); plt.ylabel('Inertia')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/6_clustering_elbow.png')
    plt.close()

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_df['Cluster'] = kmeans.fit_predict(X_scaled)

    from sklearn.decomposition import PCA
    pca      = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for c in range(3):
        mask = cluster_df['Cluster'] == c
        plt.scatter(pca_data[mask, 0], pca_data[mask, 1], label=f'Cluster {c}', color=colors[c], s=120)
        for i, row in cluster_df[mask].iterrows():
            idx = cluster_df.index.get_loc(i)
            plt.annotate(row['District_Name'], (pca_data[idx, 0], pca_data[idx, 1]), fontsize=7, ha='right')
    plt.title('District Clusters — PCA View', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/7_district_clusters_pca.png')
    plt.close()

    cluster_summary = cluster_df.groupby('Cluster').agg(
        Avg_Switching_Rate = ('Switched',        'mean'),
        Avg_Area           = ('Area',            'mean'),
        Avg_Production     = ('Production',      'mean'),
        Avg_Rainfall       = ('annual_rainfall', 'mean'),
        Num_Districts      = ('District_Name',   'count')
    ).round(2)

    print("\nCluster Interpretation Summary:")
    print(cluster_summary)

    # ==========================================
    # Section 6 — Supervised Learning
    # ==========================================
    print("\n--- Section 6: Supervised Learning (Model Comparison) ---")
    # Drop rows where Prev_Crop is NaN (the first crop year for each district-season group)
    model_df = dominant.dropna(subset=['Prev_Crop']).copy()

    from sklearn.preprocessing import LabelEncoder
    le1, le2, le3 = LabelEncoder(), LabelEncoder(), LabelEncoder()
    model_df['District_enc'] = le1.fit_transform(model_df['District_Name'])
    model_df['Season_enc']   = le2.fit_transform(model_df['Season'])
    model_df['Crop_enc']     = le3.fit_transform(model_df['Crop'])

    features = ['Area', 'Production', 'Yield',
                'annual_rainfall', 'rainfall_deviation',
                'District_enc', 'Season_enc']

    X_model = model_df[features]
    y_model = model_df['Switched']

    from sklearn.model_selection import train_test_split, cross_val_score
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_model, test_size=0.2, random_state=42, stratify=y_model
    )
    print("Train size:", len(X_train), "| Test size:", len(X_test))

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, f1_score

    models = {
        'Decision Tree'      : DecisionTreeClassifier(random_state=42),
        'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN'                : KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred   = model.predict(X_test)
        acc    = accuracy_score(y_test, pred)
        f1     = f1_score(y_test, pred, zero_division=0)
        cv_acc = cross_val_score(model, X_model, y_model, cv=5, scoring='accuracy').mean()
        results.append({
            'Model': name,
            'Test Accuracy'       : float(round(acc, 4)),
            'F1 Score'            : float(round(f1,  4)),
            'CV Accuracy (5-fold)': float(round(cv_acc, 4))
        })
        print(f"{name:22s} | Acc: {acc:.4f} | F1: {f1:.4f} | CV: {cv_acc:.4f}")

    results_df = pd.DataFrame(results).sort_values('CV Accuracy (5-fold)', ascending=False)
    print("\nModel performance rankings:")
    print(results_df.to_string(index=False))

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(results_df))
    w = 0.25
    ax.bar(x - w, results_df['Test Accuracy'],        w, label='Test Accuracy',        color='#3498db')
    ax.bar(x,     results_df['F1 Score'],              w, label='F1 Score',             color='#e74c3c')
    ax.bar(x + w, results_df['CV Accuracy (5-fold)'], w, label='CV Accuracy (5-fold)', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Comparison — Accuracy, F1, Cross-Validation', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/8_model_comparison_bar.png')
    plt.close()

    # Confusion matrix — Random Forest
    rf_model = models['Random Forest']
    pred_rf  = rf_model.predict(X_test)

    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred_rf, display_labels=['No Switch', 'Switched'], cmap='Blues', ax=ax
    )
    ax.set_title('Random Forest — Confusion Matrix', fontsize=13)
    plt.tight_layout()
    plt.savefig('plots/9_rf_confusion_matrix.png')
    plt.close()

    # Feature importance
    importance_df = pd.DataFrame({
        'Feature'   : features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Random Forest — Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/10_rf_feature_importance.png')
    plt.close()
    print("\nFeature importances:")
    print(importance_df.to_string(index=False))

    # ==========================================
    # Section 7 — Farmer Decision Intelligence Dashboard Data Preparation
    # ==========================================
    print("\n--- Section 7: Dashboard Data Preparation ---")
    dashboard_df = dominant.copy()
    # Fill in Prev_Crop NaNs to prevent enc errors, then drop them
    dashboard_df['Prev_Crop'] = dashboard_df['Prev_Crop'].fillna('None')

    dashboard_df['District_enc'] = le1.transform(dashboard_df['District_Name'])
    dashboard_df['Season_enc']   = le2.transform(dashboard_df['Season'])
    dashboard_df['Crop_enc']     = le3.transform(dashboard_df['Crop'])

    dashboard_df['Switch_Prob'] = rf_model.predict_proba(dashboard_df[features])[:, 1]

    def risk_label(p):
        if p > 0.7:   return 'High'
        elif p > 0.4: return 'Medium'
        return 'Low'

    def recommendation(p):
        if p > 0.7:   return 'High switching risk — plan alternate crop immediately'
        elif p > 0.4: return 'Moderate risk — monitor rainfall and yield trends'
        return 'Stable — current crop likely to continue'

    dashboard_df['Risk_Level']     = dashboard_df['Switch_Prob'].apply(risk_label)
    dashboard_df['Recommendation'] = dashboard_df['Switch_Prob'].apply(recommendation)

    dashboard = dashboard_df[[
        'Crop_Year', 'District_Name', 'Crop', 'Season',
        'annual_rainfall', 'Switch_Prob', 'Risk_Level', 'Recommendation'
    ]]

    # Top 10 high-risk combinations plot
    top_risk = dashboard.sort_values('Switch_Prob', ascending=False).head(10)
    plt.figure(figsize=(11, 5))
    sns.barplot(data=top_risk, x='Switch_Prob', y='District_Name', hue='Crop_Year', dodge=False, palette='Reds_r')
    plt.title('Top 10 High-Risk District-Year Combinations', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/11_top_high_risk.png')
    plt.close()

    # Risk distribution pie
    risk_counts = dashboard_df['Risk_Level'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
            colors=['#e74c3c', '#f39c12', '#2ecc71'], startangle=140)
    plt.title('Crop Switching Risk Distribution Across AP', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/12_risk_distribution_pie.png')
    plt.close()

    # ==========================================
    # Section 8 — Key Insights & Conclusion
    # ==========================================
    print("\n--- Section 8: Key Insights ---")
    most_volatile  = dominant.groupby('District_Name')['Switched'].mean().idxmax()
    most_stable    = dominant.groupby('District_Name')['Switched'].mean().idxmin()
    most_switched  = dominant[dominant['Switched']==1]['Prev_Crop'].value_counts().idxmax()
    best_model_row = results_df.iloc[0]
    top_feature    = importance_df.iloc[0]['Feature']
    overall_switch = dominant['Switched'].mean() * 100
    high_risk_count= (dashboard_df['Risk_Level'] == 'High').sum()

    print("=" * 55)
    print("     FARMER DECISION INTELLIGENCE — KEY INSIGHTS")
    print("=" * 55)
    print(f"  Overall switching rate         : {overall_switch:.1f}%")
    print(f"  Most volatile district         : {most_volatile}")
    print(f"  Most stable district           : {most_stable}")
    print(f"  Crop switched away from most   : {most_switched}")
    print(f"  Best ML model (CV accuracy)    : {best_model_row['Model']}")
    print(f"  Top predictive feature         : {top_feature}")
    print(f"  High-risk records              : {high_risk_count}")
    print("=" * 55)

    # Save performance metrics to JSON for dynamic rendering on Dashboard
    performance_data = {
        "models": results,
        "importances": importance_df.to_dict(orient="records"),
        "insights": {
            "overall_switching_rate": float(overall_switch),
            "most_volatile_district": str(most_volatile),
            "most_stable_district": str(most_stable),
            "most_switched_away_crop": str(most_switched),
            "best_model": str(best_model_row['Model']),
            "best_model_cv": float(best_model_row['CV Accuracy (5-fold)']),
            "top_feature": str(top_feature),
            "high_risk_records_count": int(high_risk_count)
        }
    }
    with open("model_performance.json", "w") as f:
        json.dump(performance_data, f, indent=4)

    # Save model assets for real-time interactive predictions
    import pickle
    model_assets = {
        'model': rf_model,
        'le_district': le1,
        'le_season': le2,
        'le_crop': le3,
        'features': features
    }
    with open("model_assets.pkl", "wb") as f:
        pickle.dump(model_assets, f)

    # ==========================================
    # Section 9 — Export Dashboard CSV
    # ==========================================
    print("\n--- Section 9: Exporting CSV ---")
    output_cols = [
        'Crop_Year', 'District_Name', 'Crop', 'Season',
        'Area', 'Production', 'Yield',
        'annual_rainfall', 'rainfall_deviation',
        'Switch_Prob', 'Risk_Level', 'Recommendation', 'Switched'
    ]
    dashboard_df[output_cols].to_csv('dashboard_output.csv', index=False)
    print("Saved: dashboard_output.csv")
    print("Saved: model_performance.json")
    print("Now run: streamlit run app.py")

if __name__ == '__main__':
    run_pipeline()
