
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, r2_score

st.set_page_config(page_title="Launch Vehicle ML Dashboard", layout="wide")
st.title("SkyWeight : AI Powered Payload Prediction for Launch Vehicle")

# Upload Excel file
uploaded_file = st.file_uploader("📂 Upload your Excel dataset", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Encode categorical columns
    df_encoded = df.copy()
    le_engine = LabelEncoder()
    le_orbit = LabelEncoder()
    df_encoded['engine_type'] = le_engine.fit_transform(df_encoded['engine_type'])
    df_encoded['orbit_target'] = le_orbit.fit_transform(df_encoded['orbit_target'])

    # Feature Engineering
    df_encoded['total_mass'] = df_encoded['fuel_mass'] + df_encoded['dry_mass']
    df_encoded['thrust_to_weight'] = df_encoded['thrust'] / df_encoded['total_mass']
    df_encoded['burn_efficiency'] = df_encoded['thrust'] / df_encoded['burn_time']
    df_encoded['fuel_burn_ratio'] = df_encoded['fuel_mass'] / df_encoded['burn_time']
    df_encoded['stage_efficiency'] = df_encoded['thrust'] / df_encoded['stage_count']

    # Simulated realistic payload estimation (new target)
    df_encoded['payload_est'] = 0.4 * df_encoded['thrust'] * (df_encoded['burn_time'] / df_encoded['total_mass'])

    # Additional classification labels
    df_encoded['structure_safe'] = ((df_encoded['dry_mass'] / df_encoded['thrust']) < 10).astype(int)
    df_encoded['delta_v_per_kg'] = df_encoded['thrust'] / df_encoded['fuel_mass']
    df_encoded['fault_detected'] = ((df_encoded['thrust'] < 2000) | (df_encoded['burn_time'] > 900)).astype(int)

    # Final input features
    feature_cols = [
        'engine_type', 'orbit_target', 'stage_count', 'perigee_km',
        'total_mass', 'thrust_to_weight', 'burn_efficiency',
        'fuel_burn_ratio', 'stage_efficiency'
    ]

    def run_payload_regression():
        X = df_encoded[feature_cols]
        y = df_encoded['payload_est']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("1. Payload Estimation")
        st.write(f"📉 Mean Squared Error: `{mse:.2f}`")
        st.write(f"📊 R² Score (Accuracy): `{r2:.2f}`")

        fig, ax = plt.subplots()
        ax.plot(y_test.values[:50], label='Actual')
        ax.plot(y_pred[:50], label='Predicted')
        ax.set_title("Payload Estimation - First 50 Samples")
        ax.legend()
        st.pyplot(fig)

        st.write("📌 Feature Importance:")
        importances = model.feature_importances_
        st.dataframe(pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False))

    def run_classification_task(name, y_col):
        X = df_encoded[feature_cols]
        y = df_encoded[y_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.subheader(f"{name} (Classification)")
        st.write(f"✅ Accuracy: `{acc:.2f}`")
        st.dataframe(pd.DataFrame(report).transpose())

    def run_regression_task(name, y_col):
        X = df_encoded[feature_cols]
        y = df_encoded[y_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader(f"{name} (Regression)")
        st.write(f"📉 Mean Squared Error: `{mse:.2f}`")
        st.write(f"📊 R² Score: `{r2:.2f}`")

        fig, ax = plt.subplots()
        ax.plot(y_test.values[:50], label='Actual')
        ax.plot(y_pred[:50], label='Predicted')
        ax.set_title(f"{name} - First 50 Predictions")
        ax.legend()
        st.pyplot(fig)

    with st.spinner("🧠 Training all models..."):
        run_payload_regression()
        run_classification_task("2. Structural Integrity Classification", "structure_safe")
        run_classification_task("3. Launch Success Prediction", "success")
        run_regression_task("4. Fuel Efficiency Estimation", "delta_v_per_kg")
        run_classification_task("5. Fault Detection", "fault_detected")
        st.success("✅ All models trained and results displayed!")

else:
    st.info("📁 Upload your dataset (Excel format) to begin.")

