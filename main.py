import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# ------------------------------
# Load model and scaler
# ------------------------------
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(page_title="MedPredict", page_icon="🩺", layout="centered")
st.title("🩺 MedPredict - Insurance Cost Estimator")
st.write(
    "Enter your details in the sidebar to estimate your medical insurance cost. "
    "Predictions are based on a public dataset (USD values) and are for educational purposes."
)

# ------------------------------
# Function to get user inputs from sidebar
# ------------------------------
st.sidebar.header("User Profile")


def get_profile(suffix="1"):
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, key=f"age_{suffix}")
    sex = st.sidebar.selectbox("Sex", ["male", "female"], key=f"sex_{suffix}")
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, key=f"bmi_{suffix}")
    children = st.sidebar.number_input("Number of Children", min_value=0, max_value=5, value=0,
                                       key=f"children_{suffix}")
    smoker = st.sidebar.selectbox("Smoker", ["yes", "no"], key=f"smoker_{suffix}")
    region = st.sidebar.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"],
                                  key=f"region_{suffix}")
    return age, sex, bmi, children, smoker, region


# ------------------------------
# Get profile inputs
# ------------------------------
age, sex, bmi, children, smoker, region = get_profile(suffix="1")


# ------------------------------
# Encode and scale features
# ------------------------------
def encode_features(age, sex, bmi, children, smoker, region):
    sex_enc = 0 if sex == "male" else 1
    smoker_enc = 0 if smoker == "yes" else 1
    region_map = {"southeast": 0, "southwest": 1, "northeast": 2, "northwest": 3}
    region_enc = region_map[region]

    numeric_features = np.array([[age, bmi, children]])
    numeric_scaled = scaler.transform(numeric_features)

    features = np.array(
        [[numeric_scaled[0, 0], sex_enc, numeric_scaled[0, 1], numeric_scaled[0, 2], smoker_enc, region_enc]])
    return features


features = encode_features(age, sex, bmi, children, smoker, region)

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔮 Predict Insurance Cost"):
    prediction = model.predict(features)[0]

    # Risk Category
    if prediction < 10000:
        risk = "🟢 Low Risk"
    elif prediction < 25000:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.subheader(f"Estimated Cost: 💰 ${prediction:,.2f} USD")
    st.write(f"Risk Category: {risk}")

    # ------------------------------
    # SHAP Explanation (Custom Bar Plot)
    # ------------------------------
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For classification models, take first array (regression case)
            shap_values = shap_values[0]

        # Get absolute SHAP values for this prediction
        abs_shap_values = np.abs(shap_values[0])

        # Feature names
        feature_names = ["Age", "Sex", "BMI", "Children", "Smoker", "Region"]

        # Sort features by importance
        sorted_idx = np.argsort(abs_shap_values)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = abs_shap_values[sorted_idx]

        # Create custom styled bar plot
        st.write("### 🔎 Feature Importance for This Prediction")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create horizontal bars
        bars = ax.barh(sorted_features, sorted_values, color='#4c72b0')

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}',
                    ha='left', va='center')

        # Style the plot
        ax.set_xlabel('Absolute SHAP Value (Impact on Prediction)')
        ax.set_title('How Each Feature Affected Your Prediction')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Optional: Add detailed SHAP values table
        st.write("### 📊 Detailed Feature Impacts")
        shap_table = []
        for name, value in zip(feature_names, shap_values[0]):
            shap_table.append({
                "Feature": name,
                "SHAP Value": f"{value:.2f}",
                "Impact": "Increases cost" if value > 0 else "Decreases cost"
            })

        st.table(shap_table)

    except Exception as e:
        st.warning(f"SHAP explanation not available: {str(e)}")
