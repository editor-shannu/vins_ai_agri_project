# Farmer Decision Intelligence: Predicting Crop Switching Behaviour in Andhra Pradesh

An end-to-end data intelligence and machine learning pipeline to predict crop switching risk levels and provide action-oriented recommendations for farmers in Andhra Pradesh, complete with a local Streamlit dashboard.

---

## 1. Introduction and Motivation
Agriculture is the backbone of the Indian economy, yet farmers face immense volatility due to changing climate patterns, market pricing fluctuations, and localized resources. In Andhra Pradesh, crop failures or low yields often force farmers to switch their primary crops between seasons. 

Predicting crop switching behavior is crucial for regional food security, resource planning (water, fertilizers), and financial risk management. By providing early indicators of high-risk crop abandonment, agricultural officers and policymakers can intervene with timely subsidies, irrigation support, or supply-chain logistics to safeguard farmers' livelihoods.

---

## 2. Problem Statement
**Objective:** Classify whether a farmer in a specific district and season will switch their dominant crop from the previous year, predict the likelihood of switching, and recommend mitigation strategies.

**Expected Outcomes:**
- A robust machine learning model that predicts switching decisions (Binary Classification: `0 = No Switch`, `1 = Switched`) based on historical crop areas, yield, and rainfall deviations.
- A risk classification model (`Low`, `Medium`, `High` risk) based on predicted probabilities.
- An interactive local decision intelligence dashboard (Streamlit) for visualizing historical trends and district-level crop switching predictions.

---

## 3. Dataset Understanding
The project utilizes historical datasets from the Government of India (GoI) Open Data portals (data.gov.in) covering the period from **1997 to 2014**:

1. **`crop_production.csv`:** Contains agricultural crop acreage, production, district, crop type, year, and season across India.
2. **`rainfall_in_india_1901-2015.csv`:** Subdivision-wise monthly and annual rainfall data.
3. **`districtwise_rainfall_normal.csv`:** Normal/average historical rainfall data for each district.

### Key Fields & Features:
* **`District_Name` / `Season` / `Crop_Year`:** Spatial-temporal identifiers.
* **`Area` / `Production` / `Yield`:** Numeric indicators of agricultural scale and productivity.
* **`annual_rainfall`:** Yearly rainfall recorded for the Coastal Andhra Pradesh subdivision.
* **`district_avg_rainfall`:** Normal average baseline rainfall for the specific district.
* **`rainfall_deviation`:** Deviation from normal rainfall (`annual_rainfall - district_avg_rainfall`), serving as a key meteorological feature.

---

## 4. Methodology
The data pipeline implements the following systematic steps:

1. **Data Cleaning:** Filter records to Andhra Pradesh (1997-2014), handle infinite yields, compute yield (`Production / Area`), and map/align district names to match rainfall baselines.
2. **Feature Engineering:** 
   - Identify the dominant crop (highest cultivated area) for each district, season, and year.
   - Calculate shifting target label: `Switched = 1` if the dominant crop changed from the previous year, otherwise `0`.
   - Calculate `rainfall_deviation` from normal averages.
3. **Unsupervised Learning:** Apply K-Means clustering to group districts into three distinct risk/volatility clusters.
4. **Supervised Learning:** Compare Decision Trees, Random Forests, Logistic Regression, and K-Nearest Neighbors using 5-fold cross-validation.
5. **Hybrid Heuristic Override (Decision Intelligence Guardrails):**
   - Because standard machine learning models (like Random Forests) cannot extrapolate out-of-distribution dry/wet weather (beyond historical values), we implement expert agricultural rule overrides.
   - **Severe Drought (>30% deficit):** Automatically overrides to **High Switching Risk** (probability $\ge 75\%$).
   - **Moderate Drought (>15% deficit):** Automatically overrides to at least **Medium Switching Risk** (probability $\ge 50\%$).
   - **Flooding/Excess Rainfall (>40% excess):** Automatically overrides to at least **Medium Switching Risk** (probability $\ge 55\%$).

---

## 5. Implementation Details
The solution is written purely in **Python** using standard local environment tools:

* **Data Wrangling:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (StandardScaler, LabelEncoder, PCA, KMeans, LogisticRegression, RandomForestClassifier)
* **Application Framework:** `streamlit`

### Automatic Background Initialization:
To keep the Git repository clean, generated files (`dashboard_output.csv`, `model_performance.json`, and `model_assets.pkl`) are excluded from version control. When the Streamlit app starts, it checks for these files and automatically runs the pipeline in the background using the raw datasets tracked under the `datasets/` folder.


### How to Run Locally:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Data Pipeline & ML Models:**
   This script executes the entire data engineering and modeling workflow, generates evaluation plots in the `plots/` directory, and exports the processed data:
   ```bash
   python AP_Crop_Switching.py
   ```

3. **Start Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 6. Results and Discussions
### Key Findings:
* **Best Model:** Logistic Regression achieved the highest 5-fold cross-validation accuracy of **~78.7%**.
* **Most Volatile District:** **KUDDAPAH** has the highest historical crop switching frequency.
* **Most Stable District:** **CHITTOOR** shows the most consistent crop selections.
* **Top Predictive Feature:** Cultivation **Area** is the strongest predictor of whether a farmer decides to switch crops, followed by crop **Production** and **Yield**.
* **Meteorological Impact:** High deviations in rainfall correlate strongly with elevated switching risk, as farmers shift away from water-intensive crops like Sugarcane during drier seasons.

### Challenges Faced:
* **Data Alignment:** District names varied across datasets (e.g., spelling differences like `KADAPA` vs `KUDDAPAH` and `VISAKHAPATANAM` vs `VISAKHAPATNAM`). Resolving these during mapping was critical to eliminating `NaN` values and preventing model failures.
* **Target Imbalance:** Historically, crop switching occurs in about ~20.1% of seasons. Stratification was implemented during the train-test split to ensure representative evaluations.

---

## 7. Conclusion
This project successfully built a Decision Intelligence pipeline for crop switching behavior in Andhra Pradesh. By turning raw district agricultural metrics and rainfall deviations into actionable risk signals, the model classifies switching probabilities with ~78.7% cross-validated accuracy. The resulting interactive Streamlit dashboard enables regional officers to proactively identify high-risk districts and coordinate support.

---

## 8. References
* Crop Production statistics: Open Government Data (OGD) Platform India (https://data.gov.in)
* India Meteorological Department (IMD) subdivision rainfall archives.
* Scikit-Learn Documentation: https://scikit-learn.org/stable/

---

## 9. Dashboard Walkthrough & Gallery
Below are screenshots capturing the primary features of the running dashboard:

### A) Interactive Decision Simulator (Farmer Tool)
Allows users to select a district and season. The application dynamically queries the historical data to populate the average area and production values as defaults. When a user runs the simulation, the backend model processes inputs through custom agricultural rules (drought and excess rainfall guardrails) to output real-time risk levels and recommendations.
![Interactive Decision Simulator](screenshots/1_decision_simulator.png)

### B) Historical Overview Dashboard
Visualizes historical crop switching frequencies over time, risk level splits, and the most common crops that farmers have switched away from in Andhra Pradesh.
![Overview Dashboard](screenshots/2_overview_dashboard.png)

### C) District Analysis & Volatility
Shows district-level switching counts, boxplot analyses correlating annual rainfall with crop switching behavior, and tables ranking the highest-risk spatial-temporal combinations.
![District Analysis](screenshots/3_district_analysis.png)

---

## Appendix
### Machine Learning Performance Comparison:

| Model | Test Accuracy | F1 Score | CV Accuracy (5-fold) |
|---|---|---|---|
| **Logistic Regression** | 80.00% | 0.1875 | **78.74%** |
| **Random Forest** | 80.00% | 0.3810 | 75.35% |
| **KNN** | 76.15% | 0.3111 | 71.96% |
| **Decision Tree** | 79.23% | 0.5263 | 61.03% |

### Sample Output Predictions:

```
District     Crop Year   Crop        Season       Switch Prob  Risk Level   Recommendation
KUDDAPAH     2002        Mango       Whole Year   0.89         High         High switching risk - plan alternate crop...
NELLORE      2014        Rice        Whole Year   0.90         High         High switching risk - plan alternate crop...
CHITTOOR     2012        Groundnut   Kharif       0.12         Low          Stable - current crop likely to continue
```
