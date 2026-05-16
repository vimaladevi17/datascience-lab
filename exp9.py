'''exp-9 Model Serialization and Deployment using Flask as a REST API
Aim

To serialize a trained machine learning model and deploy it as a RESTful API using Flask.

📝 Description

In this experiment, a machine learning model is first trained and then serialized using joblib. The serialized model is then integrated into the Flask web framework to create a REST API.

The experiment includes the following steps:

Training a machine learning model
Saving (serializing) the trained model into a file
Creating a REST API using Flask
Loading the serialized model into the API
Creating endpoints such as /predict for prediction requests
Testing the API using Thunder Client/Postman

This experiment demonstrates how machine learning models can be integrated into production-level applications for web and enterprise use.'''
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]   # input list

    features = np.array(data).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
