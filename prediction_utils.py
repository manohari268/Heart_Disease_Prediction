import numpy as np
import pickle

# Load scaler & models once
scaler = pickle.load(open("scaler.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))
rf_model = pickle.load(open("heart_model.pkl", "rb"))
rf_ewoa = pickle.load(open("rf_ewoa_model.pkl", "rb"))

# Final Feature Order
FEATURES = [
    'male','age','education','currentSmoker','cigsPerDay',
    'BPMeds','prevalentStroke','prevalentHyp','diabetes',
    'totChol','sysBP','diaBP','BMI','heartRate','glucose'
]


def preprocess_input(data_dict):
    """Convert user inputs to scaled numpy array."""
    arr = np.array([data_dict[f] for f in FEATURES]).reshape(1, -1)
    return scaler.transform(arr)


def predict_all(models_input):
    """Run predictions using all models."""
    scaled = preprocess_input(models_input)

    return {
        "lr_pred": lr_model.predict_proba(scaled)[0][1],
        "svm_pred": svm_model.predict_proba(scaled)[0][1],
        "rf_pred": rf_model.predict_proba(scaled)[0][1],
        "rf_ewoa_pred": rf_ewoa.predict_proba(scaled)[0][1]
    }


# ------------------------------------------------------
# 🔥 Add this block at the bottom for testing manually
# ------------------------------------------------------
if __name__ == "__main__":
    sample = {
        'male':1,'age':45,'education':2,'currentSmoker':0,'cigsPerDay':0,
        'BPMeds':0,'prevalentStroke':0,'prevalentHyp':1,'diabetes':0,
        'totChol':200,'sysBP':130,'diaBP':85,'BMI':25,'heartRate':78,'glucose':90
    }

    print(predict_all(sample))