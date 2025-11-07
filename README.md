# 📡 TELECOM CUSTOMER CHURN PREDICTION

## Overview
A machine learning project predicting telecom customer churn for **Interconnect** using demographic, contract, and service data.  
Built to identify customers likely to leave so marketing can offer retention incentives.

**Goal:** Predict churn with high accuracy and AUC-ROC.  
**Best Model:** 🏆 CatBoost (AUC-ROC = 0.8453 | Accuracy = 0.8034)

🔗 [View the full notebook here](./[updated]TelecomChurn.ipynb)

▶️ [Interactive App](https://deepnote.com/app/projects-8f6f/Predicting-Telecom-Customer-Churn-Insights-and-Strategies-23880e3f-a5be-4f31-87df-79348fce5c4c?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=data-app&utm_content=23880e3f-a5be-4f31-87df-79348fce5c4c)  

---

## Functionality
- Analyzes 7,000+ customer records across contract, internet, and phone data.  
- Preprocesses, encodes, and merges multiple datasets.  
- Tests multiple models (Logistic Regression, Random Forest, Gradient Boosting, CatBoost).  
- Evaluates performance using AUC-ROC and accuracy metrics.  
- Highlights top churn indicators (contract type, payment method, tenure, service usage).

---

## Key Insights
- Month-to-month contracts → highest churn.  
- Long-term contracts & multiple services → higher retention.  
- Electronic check payments → strong churn predictor.

---

## Results
| Model | AUC-ROC | Accuracy |
|--------|----------|-----------|
| Logistic Regression | 0.826 | 0.788 |
| Gradient Boosting | 0.845 | 0.794 |
| **CatBoost (Final)** | **0.845** | **0.803** |

---

## Tech Stack
**Python**, Pandas, NumPy, Scikit-learn, CatBoost, XGBoost, Matplotlib, Seaborn  
*Developed in Jupyter Notebook*

---

### Running the Project
1. Clone  
   `git clone https://github.com/rhicarmel/telecom-churn-prediction.git`
2. Install  
   `pip install -r requirements.txt`
3. Launch notebook  
   `jupyter notebook "[updated]TelecomChurn.ipynb"`

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
