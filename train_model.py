import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data/gameplay_data.csv")

# Input features
X = data[
    [
        "accuracy",
        "reaction_time",
        "score",
        "mistakes",
        "games_played"
    ]
]

# Target
y = data["skill_level"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("Random Forest model trained successfully!")
from sklearn.metrics import accuracy_score

# Predict using test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nActual Skill Levels:")
print(y_test.values)

print("\nPredicted Skill Levels:")
print(y_pred)
import joblib

joblib.dump(model, "model/random_forest_model.pkl")

print("Model saved successfully!")