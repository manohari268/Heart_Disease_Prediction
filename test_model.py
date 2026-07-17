# test_models.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# ----------------------------
# Load Dataset
# ----------------------------
def load_dataset(path="framingham.csv"):
    df = pd.read_csv(path)
    df = df.fillna(df.mean(numeric_only=True))
    return df
# ----------------------------
# Load Pickle File Safely
# ----------------------------
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        print(f"❌ ERROR: Unable to load {path}")
        return None
# ----------------------------
# MAIN TEST FUNCTION
# ----------------------------
def test_models():
    print("\n📌 Loading dataset...")
    df = load_dataset()
    
    FEATURES = [
        'male','age','education','currentSmoker','cigsPerDay',
        'BPMeds','prevalentStroke','prevalentHyp','diabetes',
        'totChol','sysBP','diaBP','BMI','heartRate','glucose'
    ] 
    X = df[FEATURES]
    y = df["TenYearCHD"]
    # Load scaler
    scaler = load_pickle("scaler.pkl")
    if scaler is None:
        print("❌ Scaler not found!")
        return
    X_scaled = scaler.transform(X)
    # Train-test split
    print("\n📌 Splitting dataset (80% training, 20% testing)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    # Load ML models
    models = {
        "Logistic Regression": load_pickle("lr_model.pkl"),
        "SVM": load_pickle("svm_model.pkl"),
        "Random Forest": load_pickle("heart_model.pkl"),
    }
    print("\n============================")
    print("🔍 MODEL PERFORMANCE TESTING")
    print("============================")
    # Evaluate each model
    for name, model in models.items():
        if model is None:
            print(f"❌ {name} model not found!")
            continue
        print(f"\n------------------------")
        print(f"📌 Testing: {name}")
        print("------------------------")
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
    print("\n🎉 Testing Completed Successfully.")
# Run tests
if __name__ == "__main__":
    test_models()