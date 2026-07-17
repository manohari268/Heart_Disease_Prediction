import pickle
import pandas as pd

# Load dataset
df = pd.read_csv("framingham.csv")
df=df.fillna(df.mean(numeric_only=True))

FEATURES = [
    'male','age','education','currentSmoker','cigsPerDay',
    'BPMeds','prevalentStroke','prevalentHyp','diabetes',
    'totChol','sysBP','diaBP','BMI','heartRate','glucose'
]

X = df[FEATURES]

# Load scaler.pkl
scaler = pickle.load(open("scaler.pkl", "rb"))

# Scale dataset
scaled = scaler.transform(X)

# Convert to DataFrame
scaled_df = pd.DataFrame(scaled, columns=FEATURES)

print("\nScaled Dataset (first 10 rows):\n")
print(scaled_df.head())
print("\nMeans after scaling:\n")
print(scaled_df.mean())
print("\nStandard deviations after scaling:\n")
print(scaled_df.std())
