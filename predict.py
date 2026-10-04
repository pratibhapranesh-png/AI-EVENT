import joblib

# Load trained model
model = joblib.load("model/random_forest_model.pkl")

# New player's gameplay data
player = [[
    85,     # accuracy
    0.7,    # reaction time
    850,    # score
    4,      # mistakes
    19      # games played
]]

# Predict skill level
prediction = model.predict(player)

print("Predicted Skill Level:", prediction[0])