# train_models_with_ewoa.py
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

print("📌 Loading dataset...")
df = pd.read_csv("framingham.csv")
df = df.fillna(df.mean(numeric_only=True))

# FIXED FEATURE LIST (15 features)
FEATURES = [
    'male','age','education','currentSmoker','cigsPerDay',
    'BPMeds','prevalentStroke','prevalentHyp','diabetes',
    'totChol','sysBP','diaBP','BMI','heartRate','glucose'
]

print(f"Using features ({len(FEATURES)}): {FEATURES}")

X = df[FEATURES]
y = df["TenYearCHD"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pickle.dump(scaler, open("scaler.pkl", "wb"))
print("Saved scaler.pkl")

# ---- EWOA helpers ----
def decode_solution(sol):
    return {
        "lr_C": float(sol[0]),
        "svm_C": float(sol[1]),
        "svm_gamma": float(sol[2]),
        "rf_n": int(max(10, round(sol[3]))),
        "rf_depth": int(max(1, round(sol[4])))
    }

def evaluate(sol):
    params = decode_solution(sol)

    # Logistic Regression
    lr = LogisticRegression(C=params["lr_C"], max_iter=500, solver="lbfgs")
    lr.fit(X_train_scaled, y_train)
    a1 = accuracy_score(y_test, lr.predict(X_test_scaled))

    # SVM (probability=True to support SHAP later)
    svm = SVC(C=params["svm_C"], gamma=params["svm_gamma"], probability=True)
    svm.fit(X_train_scaled, y_train)
    a2 = accuracy_score(y_test, svm.predict(X_test_scaled))

    # RF
    rf = RandomForestClassifier(n_estimators=params["rf_n"], max_depth=params["rf_depth"], random_state=42)
    rf.fit(X_train_scaled, y_train)
    a3 = accuracy_score(y_test, rf.predict(X_test_scaled))

    return (a1 + a2 + a3) / 3.0
# ---- EWOA hyperparams ----
pop_size = 12         # lower to speed up; increase if you want more search
max_iter = 20         # reduce for faster runs
dim = 5
lb = [0.001, 0.01, 1e-5, 50, 3]
ub = [10, 20, 1.0, 400, 30]
# initialize population
pop = np.random.uniform(lb, ub, (pop_size, dim))
scores = np.array([evaluate(sol) for sol in pop])
best_idx = np.argmax(scores)
best_score = scores[best_idx]
best_sol = pop[best_idx].copy()
history = []
print(f"\nRunning EWOA: pop_size={pop_size}, max_iter={max_iter}\n")
for it in range(max_iter):
    a = 2 * (1 - it / max_iter)
    for i in range(pop_size):
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * a * r1 - a
        C = 2 * r2

        b = 1
        l = np.random.uniform(-1, 1)
        p = np.random.rand()

        if p < 0.5:
            if abs(A) < 1:
                D = abs(C * best_sol - pop[i])
                pop[i] = best_sol - A * D
            else:
                rand_idx = np.random.randint(pop_size)
                D = abs(C * pop[rand_idx] - pop[i])
                pop[i] = pop[rand_idx] - A * D
        else:
            D = abs(best_sol - pop[i])
            pop[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_sol

        pop[i] = np.clip(pop[i], lb, ub)

    scores = np.array([evaluate(sol) for sol in pop])
    if scores.max() > best_score:
        best_score = scores.max()
        best_sol = pop[np.argmax(scores)].copy()

    history.append(best_score)
    print(f"Iteration {it+1}/{max_iter} — Best Score: {best_score:.4f}")

best_params = decode_solution(best_sol)
print("\nEWOA Best Score:", best_score)
print("Best params:", best_params)

# save EWOA artifacts
pickle.dump(best_params, open("ewoa_best_params.pkl", "wb"))
pickle.dump(history, open("ewoa_history.pkl", "wb"))
print("Saved ewoa_best_params.pkl and ewoa_history.pkl")

# ---- Train final models on full training set (scaled) ----
print("\nTraining final models on training data...")
lr = LogisticRegression(C=best_params["lr_C"], max_iter=1000, solver="lbfgs")
svm = SVC(C=best_params["svm_C"], gamma=best_params["svm_gamma"], probability=True)
rf = RandomForestClassifier(n_estimators=best_params["rf_n"], max_depth=best_params["rf_depth"], random_state=42)

lr.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)

pickle.dump(lr, open("lr_model.pkl", "wb"))
pickle.dump(svm, open("svm_model.pkl", "wb"))
pickle.dump(rf, open("heart_model.pkl", "wb"))
print("Saved lr_model.pkl, svm_model.pkl, heart_model.pkl")
print("\n🎉 TRAINING + EWOA complete.")