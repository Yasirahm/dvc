
import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("data/students.csv")

# 2. Separate input and output
X = data.drop(columns=["student_id", "result"])
y = data["result"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Create ML pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 5. Train model
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 8. Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

# 9. Save metrics
os.makedirs("reports", exist_ok=True)

metrics = {
    "accuracy": float(accuracy),
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

with open("reports/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved successfully!")
print("Metrics saved successfully!")
