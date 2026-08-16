# ❤️ Heart Disease Prediction

A Streamlit web app that predicts a person's 10-year risk of coronary heart disease (CHD) using the Framingham Heart Study dataset. It combines multiple ML models, hyperparameter tuning via the Enhanced Whale Optimization Algorithm (EWOA), an interactive risk dashboard, and downloadable PDF reports.

## ✨ Features

- **Risk Prediction** — Enter patient health parameters and get a CHD risk score (Low / High Risk) averaged across three trained models.
- **Multiple ML Models** — Logistic Regression, SVM, and Random Forest, with an ensemble-style averaged prediction.
- **EWOA Hyperparameter Tuning** — Model hyperparameters are optimized using the Enhanced Whale Optimization Algorithm (`train_models_with_ewoa.py`), with saved best parameters and optimization history.
- **BMI Calculator** — Built-in calculator that can auto-fill the BMI field in the prediction form.
- **Interactive Dashboard** — Feature-wise risk contribution charts, a confusion matrix, and model accuracy comparison, all built with Plotly.
- **PDF Report Export** — Download a generated PDF summary of each prediction, including input values and risk score.
- **Simple Login Screen** — Basic username/password gate before accessing the app.

## 🗂️ Project Structure

```
Heart_Disease_Prediction/
├── app.py                      # Main Streamlit application
├── train_models_with_ewoa.py   # Model training + EWOA hyperparameter optimization
├── prediction_utils.py         # Helper functions used for predictions
├── scaled_data.py               # Data scaling utilities
├── scaler.py                    # Scaler creation script
├── test_model.py                # Model testing script
├── framingham.csv               # Framingham Heart Study dataset
├── heart_model.pkl              # Trained Random Forest model
├── lr_model.pkl                 # Trained Logistic Regression model
├── svm_model.pkl                # Trained SVM model
├── scaler.pkl                   # Fitted feature scaler
├── ewoa_best_params.pkl         # Best hyperparameters found by EWOA
├── ewoa_history.pkl             # EWOA optimization history
├── low risk.pdf                 # Sample generated report
└── requirements.txt              # Python dependencies
```

## 🧠 Dataset

The app uses the **Framingham Heart Study** dataset (`framingham.csv`), with the following input features:

`male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose`

Target variable: `TenYearCHD` (whether the patient developed CHD within 10 years).

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/manohari268/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Login

The app has a simple built-in login screen for demo purposes:
- **Username:** `admin`
- **Password:** `heart123`

> ⚠️ These credentials are hardcoded in `app.py` for demonstration purposes only. Do not use this authentication approach in a production/public deployment — replace it with a proper auth system before exposing the app publicly.

## 🏗️ Model Training

To retrain the models and re-run EWOA hyperparameter optimization:

```bash
python train_models_with_ewoa.py
```

This regenerates the `.pkl` model files, the scaler, and the EWOA parameter/history files used by `app.py`.

## 🛠️ Tech Stack

- **Frontend/App:** Streamlit
- **ML:** scikit-learn (Logistic Regression, SVM, Random Forest)
- **Optimization:** Enhanced Whale Optimization Algorithm (EWOA)
- **Data Handling:** pandas, numpy
- **Visualization:** Plotly, Matplotlib
- **Reports:** fpdf / Matplotlib PDF export

## 📌 Notes

- This project is for educational/academic purposes and is **not a substitute for professional medical advice**.
- Predictions are based on a single historical dataset and should not be used for real clinical decision-making.

## 📄 License

No license specified yet — consider adding one (e.g., MIT) if you'd like others to reuse this project.
