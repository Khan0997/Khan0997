
from flask_cors import Flask, request, jsonify
from flask_cors import CORS
from joblib import load

model = load("model.joblib")
vectorizer = load("vectorizer.joblib")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return jsonify({"result": "fake" if prediction == 1 else "real"})

if __name__ == "__main__":
    app.run()
