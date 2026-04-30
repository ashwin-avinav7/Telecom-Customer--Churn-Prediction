import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load saved model
model = pickle.load(open("models/gb_model.pkl", "rb"))

# Load dataset
df = pd.read_csv("data/telco_churn.csv")

# Convert target column into 0/1
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Remove customerID
df.drop(columns=["customerID"], inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing rows
df.dropna(inplace=True)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# One hot encoding
X = pd.get_dummies(X, drop_first=True)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Align columns if needed
saved_cols = pickle.load(open("models/feature_cols.pkl", "rb"))
X_test = X_test.reindex(columns=saved_cols, fill_value=0)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc*100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))