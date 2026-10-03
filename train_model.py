import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("data.csv")

# Drop unnecessary ID columns
df = df.drop(["nameOrig", "nameDest"], axis=1)

# Convert transaction type into numerical columns
df = pd.get_dummies(df, columns=["type"], drop_first=True)

# Separate features and target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "model/fraud_model.pkl")

print("Model training completed successfully!")
print("Model saved at model/fraud_model.pkl")