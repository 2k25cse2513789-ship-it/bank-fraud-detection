from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model/fraud_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    step = float(request.form["step"])
    transaction_type = request.form["type"]
    amount = float(request.form["amount"])
    oldbalanceOrg = float(request.form["oldbalanceOrg"])
    newbalanceOrig = float(request.form["newbalanceOrig"])
    oldbalanceDest = float(request.form["oldbalanceDest"])
    newbalanceDest = float(request.form["newbalanceDest"])
    isFlaggedFraud = int(request.form["isFlaggedFraud"])

    type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
    type_DEBIT = 1 if transaction_type == "DEBIT" else 0
    type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
    type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0

    data = pd.DataFrame([[
        step,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        isFlaggedFraud,
        type_CASH_OUT,
        type_DEBIT,
        type_PAYMENT,
        type_TRANSFER
    ]], columns=[
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER"
    ])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "⚠️ Fraudulent Transaction Detected"
    else:
        result = "✅ Normal Transaction"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)