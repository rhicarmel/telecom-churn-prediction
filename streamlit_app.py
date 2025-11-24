import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from catboost import CatBoostClassifier

# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📞",
    layout="wide",
)

st.title("📞 Telecom Customer Churn Prediction")
st.write(
    "Interactive app built from the **Telecom Customer Churn** project. "
    "It trains a CatBoost model on the underlying telecom data and lets you "
    "explore churn patterns and view model predictions for individual customers."
)

st.caption(
    "Trains a CatBoost model on telecom customer data to estimate churn risk "
    "and explore behavior patterns across the customer base."
)

# ---------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------

@st.cache_data
def load_raw_data():
    contract_df = pd.read_csv("data/contract.csv")
    personal_df = pd.read_csv("data/personal.csv")
    internet_df = pd.read_csv("data/internet.csv")
    phone_df = pd.read_csv("data/phone.csv")

    # Create target from EndDate
    contract_df["EndDate"] = contract_df["EndDate"].replace("No", np.nan)
    contract_df["EndDate"] = pd.to_datetime(contract_df["EndDate"])
    contract_df["Churn"] = contract_df["EndDate"].notna().astype(int)

    # Merge all datasets
    df = (
        contract_df.merge(personal_df, on="customerID", how="left")
        .merge(internet_df, on="customerID", how="left")
        .merge(phone_df, on="customerID", how="left")
    )

    # Fix numeric columns exactly like the notebook
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["MonthlyCharges"].fillna(df["MonthlyCharges"].median(), inplace=True)

    return df


@st.cache_resource
def train_model():
    """Recreate the notebook pipeline and train the final CatBoost model."""
    df = load_raw_data().copy()

    # Prepare features as in notebook
    model_df = df.drop(columns=["customerID", "BeginDate", "EndDate"])
    model_df = pd.get_dummies(model_df, drop_first=True)

    X = model_df.drop(columns=["Churn"])
    y = model_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Scale numerical columns (same as notebook)
    num_cols = ["MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # Final CatBoost model (same hyperparameters as notebook)
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=4,
        verbose=0,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "num_cols": num_cols,
        "feature_cols": X_train_scaled.columns.tolist(),
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "auc": auc,
        "accuracy": acc,
    }

    return artifacts, df


def prepare_single_customer_features(
    raw_df, customer_id, feature_cols, scaler, num_cols
):
    """Build a single-row feature matrix for a selected customer."""

    row = raw_df.loc[raw_df["customerID"] == customer_id].copy()
    if row.empty:
        return None

    # Recreate the preprocessing pipeline on this row
    row["EndDate"] = row["EndDate"].replace("No", np.nan)
    row["EndDate"] = pd.to_datetime(row["EndDate"])
    row["Churn"] = row["EndDate"].notna().astype(int)

    model_row = row.drop(columns=["customerID", "BeginDate", "EndDate"])
    model_row = pd.get_dummies(model_row, drop_first=True)

    # Align columns with training feature set
    model_row = model_row.reindex(columns=feature_cols, fill_value=0)

    # Scale numeric columns using training scaler
    model_row[num_cols] = scaler.transform(model_row[num_cols])

    return model_row


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------

with st.spinner("Training CatBoost model from project data..."):
    artifacts, df_raw = train_model()

model = artifacts["model"]
scaler = artifacts["scaler"]
num_cols = artifacts["num_cols"]
feature_cols = artifacts["feature_cols"]
auc = artifacts["auc"]
accuracy = artifacts["accuracy"]

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("Overview", "Customer Explorer"),
)

# ---------------------------------------------------------
# Overview page
# ---------------------------------------------------------
if page == "Overview":
    st.subheader("Model Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("AUC-ROC", f"{auc:.3f}")
    with col2:
        st.metric("Accuracy", f"{accuracy:.3f}")

    st.markdown("---")
    st.subheader("Churn Distribution")

    churn_counts = df_raw["Churn"].value_counts().rename(
        {0: "Active", 1: "Churned"})
    fig, ax = plt.subplots()
    ax.bar(churn_counts.index, churn_counts.values)
    ax.set_xlabel("Status")
    ax.set_ylabel("Number of customers")
    ax.set_title("Active vs Churned Customers")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Sample of Data")
    st.dataframe(df_raw.head(20))

# ---------------------------------------------------------
# Customer Explorer page
# ---------------------------------------------------------
elif page == "Customer Explorer":

    st.subheader("Customer Churn Prediction")
    st.caption(
        "Select a customer to view their profile and the model's estimated probability "
        "that they will churn. The prediction is based on contract details, services, "
        "and billing information."
    )

    customer_ids = df_raw["customerID"].sort_values().unique()
    selected_id = st.selectbox("Select a customer ID", customer_ids)

    if selected_id:
        row_raw = df_raw.loc[df_raw["customerID"] == selected_id]

        st.markdown("**Selected customer details:**")
        st.dataframe(row_raw)

        features_row = prepare_single_customer_features(
            df_raw, selected_id, feature_cols, scaler, num_cols
        )

        if features_row is not None:
            prob = model.predict_proba(features_row)[:, 1][0]
            pred_class = "Churn" if prob >= 0.5 else "Stay"

            st.markdown("### Prediction")
            st.write(f"**Estimated probability of churn:** `{prob:.2%}`")
            st.write(f"**Model decision:** **{pred_class}**")
        else:
            st.error("Could not build features for this customer.")
