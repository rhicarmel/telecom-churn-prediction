# 📡 Telecom Customer Churn Prediction

## Overview
A complete machine learning pipeline predicting customer churn for Interconnect using demographic, contract, internet, and phone service data.  
The project identifies customers likely to leave so retention teams can intervene with targeted offers.

**Goal:** Build an accurate churn prediction model with strong AUC-ROC.  
**Best Model:** **CatBoost**  
- **AUC-ROC:** 0.845  
- **Accuracy:** 0.803  

🔗 **Run Notebook:**  
[`TelecomChurn.ipynb`](https://github.com/rhicarmel/telecom-churn-prediction/blob/main/notebooks/TelecomChurn.ipynb)

🔗 **Run the interactive app on Streamlit:**  
https://telecom-churn-prediction-rhi-222.streamlit.app/

---

## Functionality
- Processes and merges **7,000+** customer records from four datasets.  
- Cleans contract, internet, and billing fields; converts targets; handles missing values.  
- Encodes categorical variables and scales numeric features.  
- Trains and compares multiple models:
  - Logistic Regression  
  - Random Forest  
  - Gradient Boosting  
  - CatBoost (final model)  
- Evaluates performance using **AUC-ROC** and **accuracy**.  
- Visualizes churn patterns and feature importance.

---

## Key Insights
- **Month-to-month contracts** show the highest churn rate.  
- Customers on **one-year or two-year contracts** have significantly better retention.  
- **Electronic check payments** are strongly associated with higher churn.  
- Long-term customers with multiple bundled services are more likely to stay.

---

## Results

| Model | AUC-ROC | Accuracy |
|-------|---------|----------|
| Logistic Regression | 0.826 | 0.788 |
| Gradient Boosting | 0.845 | 0.794 |
| **CatBoost (Final)** | **0.845** | **0.803** |

CatBoost delivered the best balance of performance and stability, and is used in the Streamlit application.

---

## Tech Stack
- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**, **CatBoost**, **XGBoost**
- **Matplotlib**, **Seaborn**
- **Streamlit** (interactive app)
- Developed in **Jupyter Notebook**

---

## Running the Project
1. Clone  
   `git clone https://github.com/rhicarmel/telecom-churn-prediction.git`
2. Install  
   `pip install -r requirements.txt`
3. Launch notebook  
   `jupyter notebook "TelecomChurn.ipynb"`

---

## Future Improvements
- Add interpretability with SHAP/LIME.
- Build a Streamlit dashboard for churn predictions.
- Automate model retraining with new data.
---

### Reproducibility
- Train and test split is stratified by `Churn`  
- Random seeds set for NumPy, Python, CatBoost  
- Metrics reported on held-out test set

---

## Author
**Rhiannon Fillingham**  

• 📎 [LinkedIn](http://linkedin.com/rhiannonfilli)
