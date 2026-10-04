import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/gameplay_data.csv")

X = data[
    [
        "accuracy",
        "reaction_time",
        "score",
        "mistakes",
        "games_played"
    ]
]

y = data["skill_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))
