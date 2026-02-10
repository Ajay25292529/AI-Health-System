import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# DIABETES MODEL
# -------------------------

diabetes_df = pd.read_csv("data/diabetes.csv")

X_d = diabetes_df.drop("Outcome", axis=1)
y_d = diabetes_df["Outcome"]

scaler_d = StandardScaler()
X_d_scaled = scaler_d.fit_transform(X_d)

X_train, X_test, y_train, y_test = train_test_split(X_d_scaled, y_d, test_size=0.2, random_state=42)

model_d = RandomForestClassifier()
model_d.fit(X_train, y_train)

acc_d = accuracy_score(y_test, model_d.predict(X_test))
print("Diabetes Accuracy:", acc_d)

joblib.dump(model_d, "models/diabetes_model.pkl")
joblib.dump(scaler_d, "models/diabetes_scaler.pkl")

# -------------------------
# HEART MODEL
# -------------------------

heart_df = pd.read_csv("data/heart.csv")

X_h = heart_df.drop("target", axis=1)
y_h = heart_df["target"]

scaler_h = StandardScaler()
X_h_scaled = scaler_h.fit_transform(X_h)

X_train, X_test, y_train, y_test = train_test_split(X_h_scaled, y_h, test_size=0.2, random_state=42)

model_h = RandomForestClassifier()
model_h.fit(X_train, y_train)

acc_h = accuracy_score(y_test, model_h.predict(X_test))
print("Heart Accuracy:", acc_h)

joblib.dump(model_h, "models/heart_model.pkl")
joblib.dump(scaler_h, "models/heart_scaler.pkl")

print("Models trained & saved successfully!")
