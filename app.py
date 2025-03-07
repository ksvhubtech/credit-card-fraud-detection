from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model/fraud_detection_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array([data["features"]]).reshape(1, -1)
        prediction = model.predict(features)[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
