from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/random_forest_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    player = [[
        data["accuracy"],
        data["reaction_time"],
        data["score"],
        data["mistakes"],
        data["games_played"]
    ]]

    prediction = model.predict(player)

    return jsonify({
        "skill_level": prediction[0]
    })

if __name__ == "__main__":
    app.run(debug=True)