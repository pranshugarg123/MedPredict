# 🏥 MedPredict

**MedPredict** is a machine learning project that predicts medical insurance costs based on personal and lifestyle attributes. It uses regression models and SHAP explainability to deliver accurate and interpretable results.

---

## 📌 Features

- Predicts insurance charges using:
  - Age, Sex, BMI, Children, Smoker, Region
- Trained using:
  - Linear Regression
  - Random Forest Regressor (tuned with GridSearchCV)
- SHAP-based model interpretability
- Visual data analysis using Matplotlib and Seaborn

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn, SHAP
- **Notebook Environment:** Jupyter / Google Colab

---

## 📊 Model Performance

| Model                 | MAE   | R² Score |
|----------------------|-------|----------|
| Linear Regression     | ~4190 | 0.78     |
| Random Forest         | ~2508 | 0.86     |
| Tuned Random Forest   | ~2450 | 0.87     |

---

## 📂 Dataset

- Source: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Features: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`

---

## 📈 SHAP Interpretability

- SHAP summary plots show how each feature affects predictions
- `smoker`, `age`, and `bmi` are the most impactful features

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/pranshugarg123/MedPredict.git
   cd MedPredict
