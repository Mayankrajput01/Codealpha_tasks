import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

print("=" * 60)
print("   Credit Score Prediction — Training Started")
print("=" * 60)


np.random.seed(0)
N = 5000


age                  = np.random.randint(21, 65, N)
annual_income        = np.random.randint(120000, 2500000, N)
num_credit_cards     = np.random.randint(0, 10, N)
num_loans            = np.random.randint(0, 7, N)
num_delayed_payments = np.random.randint(0, 25, N)
outstanding_debt     = np.random.randint(0, 800000, N)
credit_utilization   = np.random.uniform(1, 99, N).round(1)
credit_history_yrs   = np.random.randint(0, 30, N)
monthly_balance      = np.random.randint(0, 150000, N)
emi_monthly          = np.random.randint(0, 60000, N)
interest_rate        = np.random.randint(5, 36, N)
missed_payments      = np.random.randint(0, 15, N)
debt_to_income       = (outstanding_debt / (annual_income + 1)).round(3)
savings_rate         = (monthly_balance / ((annual_income / 12) + 1)).round(3)


score = np.zeros(N, dtype=float)

# Income impact
score += np.where(annual_income > 1200000, 3.0,
         np.where(annual_income > 600000,  2.0,
         np.where(annual_income > 300000,  1.0, 0.0)))

# Credit history
score += np.where(credit_history_yrs > 15, 3.0,
         np.where(credit_history_yrs > 7,  2.0,
         np.where(credit_history_yrs > 3,  1.0, 0.0)))

# Payment behaviour
score += np.where(num_delayed_payments == 0, 3.0,
         np.where(num_delayed_payments <= 2,  2.0,
         np.where(num_delayed_payments <= 5,  0.5, -2.0)))

score += np.where(missed_payments == 0,  2.0,
         np.where(missed_payments <= 2, 0.5, -2.5))

# Debt & utilization
score += np.where(credit_utilization < 20,  2.5,
         np.where(credit_utilization < 40,  1.5,
         np.where(credit_utilization < 60,  0.0, -2.0)))

score += np.where(debt_to_income < 0.2,  2.0,
         np.where(debt_to_income < 0.4,  1.0,
         np.where(debt_to_income < 0.6,  0.0, -2.0)))

# Savings
score += np.where(savings_rate > 0.3, 2.0,
         np.where(savings_rate > 0.1, 1.0, 0.0))

# Cards & loans
score += np.where(num_credit_cards <= 3, 1.0,
         np.where(num_credit_cards <= 6, 0.0, -1.0))
score += np.where(num_loans <= 2, 1.0,
         np.where(num_loans <= 4,  0.0, -1.5))

# Interest rate
score += np.where(interest_rate < 12, 1.5,
         np.where(interest_rate < 20, 0.5, -1.0))

# Age factor
score += np.where(age > 45, 1.0,
         np.where(age > 30, 0.5, 0.0))

# Add realistic noise
score += np.random.normal(0, 0.8, N)



labels = np.where(score >= 9,  "Good",
         np.where(score >= 4,  "Standard", "Poor"))

df = pd.DataFrame({
    "age":                  age,
    "annual_income":        annual_income,
    "num_credit_cards":     num_credit_cards,
    "num_loans":            num_loans,
    "num_delayed_payments": num_delayed_payments,
    "outstanding_debt":     outstanding_debt,
    "credit_utilization":   credit_utilization,
    "credit_history_yrs":   credit_history_yrs,
    "monthly_balance":      monthly_balance,
    "emi_monthly":          emi_monthly,
    "interest_rate":        interest_rate,
    "missed_payments":      missed_payments,
    "debt_to_income":       debt_to_income,
    "savings_rate":         savings_rate,
    "label":                labels,
})

print(f"\nDataset: {N} samples")
print(f"Class distribution:\n{df['label'].value_counts().to_string()}\n")

label_map     = {"Poor": 0, "Standard": 1, "Good": 2}
reverse_map   = {v: k for k, v in label_map.items()}
df["encoded"] = df["label"].map(label_map)

FEATURES = [
    "age", "annual_income", "num_credit_cards", "num_loans",
    "num_delayed_payments", "outstanding_debt", "credit_utilization",
    "credit_history_yrs", "monthly_balance", "emi_monthly",
    "interest_rate", "missed_payments", "debt_to_income", "savings_rate"
]

X = df[FEATURES].values
y = df["encoded"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


print("Training model (Gradient Boosting)...")
model = GradientBoostingClassifier(
    n_estimators     = 300,
    learning_rate    = 0.08,
    max_depth        = 5,
    min_samples_leaf = 15,
    subsample        = 0.85,
    random_state     = 42
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nTest Accuracy   : {acc*100:.2f}%")
print(f"CV Accuracy     : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Poor","Standard","Good"]))


with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

# Save feature list and label map for app.py
meta_path = os.path.join(BASE_DIR, "meta.pkl")
with open(meta_path, "wb") as f:
    pickle.dump({"features": FEATURES, "label_map": reverse_map}, f)

print(f"\nModel  saved -> {MODEL_PATH}")
print(f"Scaler saved -> {SCALER_PATH}")
print(f"Meta   saved -> {meta_path}")
print("\n✅ Done! Run: streamlit run app.py")
