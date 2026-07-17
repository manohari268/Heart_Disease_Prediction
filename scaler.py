# show_scaler_info.py
import pickle
import pandas as pd

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Extract values
means = scaler.mean_
stds = scaler.scale_

# If you want to show with feature names:
FEATURES = [
    'male','age','education','currentSmoker','cigsPerDay',
    'BPMeds','prevalentStroke','prevalentHyp','diabetes',
    'totChol','sysBP','diaBP','BMI','heartRate','glucose'
]

# Create a table
df = pd.DataFrame({
    "feature": FEATURES,
    "mean": means,
    "std": stds,
})

print("\n📌 SCALER DETAILS — STANDARDIZATION PARAMETERS\n")
print(df.to_string(index=False))