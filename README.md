# 🛡️ Bank Fraud Detection using Machine Learning

## 📌 Project Overview

This project is a Machine Learning based Bank Fraud Detection System. It analyzes transaction details and predicts whether a transaction is **Normal** or **Fraudulent**.

The project uses a **Random Forest Classifier** trained on the PaySim financial transaction dataset.

## 🎯 Objective

The main objective of this project is to detect potentially fraudulent financial transactions using Machine Learning.

## 🧠 Machine Learning Model

The project uses:

* Random Forest Classifier
* Scikit-learn
* Pandas
* NumPy
* Joblib

The dataset contains transaction information such as:

* Transaction type
* Transaction amount
* Origin account balance
* Destination account balance
* Transaction step
* Fraud indicator

## ⚙️ Project Workflow

```text
Transaction Input
       ↓
Data Preprocessing
       ↓
Feature Encoding
       ↓
Random Forest Model
       ↓
Fraud Prediction
       ↓
Normal / Fraudulent Result
```

## 🌐 Web Application

The project uses **Flask** to create a web interface where users can enter transaction details.

The application provides two possible outputs:

### ✅ Normal Transaction

The model predicts that the given transaction does not belong to the fraudulent class.

### ⚠️ Fraudulent Transaction Detected

The model predicts that the given transaction belongs to the fraudulent class.

## 📊 Dataset

The project uses the PaySim synthetic financial transaction dataset.

The dataset contains more than **6.3 million transactions** and includes both normal and fraudulent transactions.

## 📈 Model Performance

The Random Forest model achieved approximately:

* ROC-AUC: **0.9935**
* Fraud Recall: **0.78**
* Fraud Precision: **0.98**

These results are based on the test data used during model evaluation.

## 📁 Project Structure

```text
projectmini/
│
├── data.csv
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── model/
│   └── fraud_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── notebooks/
    └── Analysis.ipynb
```

## ▶️ How to Run

### 1. Activate virtual environment

```bash
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask application

```bash
python app.py
```

### 4. Open in browser

```text
http://127.0.0.1:5000
```

## ⚠️ Important Note

This project is an academic Machine Learning demonstration using a synthetic dataset. A prediction from this model should not be treated as proof that a real-world transaction is fraudulent.

## 👨‍💻 Project Purpose

This project demonstrates the practical use of:

* Machine Learning
* Classification
* Random Forest
* Class imbalance handling
* Feature preprocessing
* Flask web development
* Model deployment
