# 🏥 MedPredict

**MedPredict** is a machine learning web application that predicts medical insurance costs based on personal and lifestyle attributes. It uses regression models and SHAP explainability to deliver accurate and interpretable results.

---

## 🌐 Live Demo

Check out the deployed app: [MedPredict by Pranshu Garg](https://medpredict-pranshugarg.streamlit.app/)

---

## 📌 Features

- Predict insurance charges using:
  - Age, Sex, BMI, Children, Smoker, Region
- Provides risk categorization:
  - 🟢 Low Risk | 🟡 Medium Risk | 🔴 High Risk
- SHAP-based model interpretability:
  - Visualizes feature impact for each prediction
- Clean, sidebar-based UI using Streamlit

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Streamlit, NumPy, scikit-learn, matplotlib, SHAP
- **Modeling:** Random Forest Regressor, Linear Regression
- **Visualization:** Matplotlib, SHAP plots

---

## 📊 Model Performance

| Model                 | MAE    | R² Score |
|----------------------|--------|----------|
| Linear Regression     | ~4190  | 0.78     |
| Random Forest         | ~2508  | 0.86     |
| Tuned Random Forest   | ~2450  | 0.87     |

---

## 📂 Dataset

- Source: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Features: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`

---

## 📈 SHAP Interpretability

- SHAP summary plots show how each feature affects predictions
- `smoker`, `age`, and `bmi` are the most impactful features
- Provides a detailed table showing each feature’s contribution to a specific prediction

---

## 🚀 How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/pranshugarg123/MedPredict.git
   cd MedPredict
