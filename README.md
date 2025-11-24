# 📡 Telecom Customer Churn Prediction

## Overview
A complete machine learning pipeline predicting customer churn for Interconnect using demographic, contract, internet, and phone service data.  
The project identifies customers likely to leave so retention teams can intervene with targeted offers.

**Goal:** Build an accurate churn prediction model with strong AUC-ROC.  
**Best Model:** **CatBoost**  
- **AUC-ROC:** 0.845  
- **Accuracy:** 0.803  

### Run Notebook: 
[![Run Notebook](https://img.shields.io/badge/📓_Open_Notebook-orange?style=for-the-badge)](https://github.com/rhicarmel/telecom-churn-prediction/blob/main/notebooks/TelecomChurn.ipynb)

### Run the interactive app on Streamlit:
[![Streamlit App](https://img.shields.io/badge/🚀_Open_Streamlit_App-ff4b4b?style=for-the-badge)](https://telecom-churn-prediction-rhi-222.streamlit.app/)

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
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FF6F00?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-0044cc?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-013243?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C9?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
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

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/rhiannonfilli)
