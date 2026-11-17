import os
import secrets
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import tensorflow as tf

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    session
)

from src.inference import load_inference_components, predict_churn


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
DATASET_PATH = PROJECT_DIR / "bank dataset.csv"
ADDITIONS_PATH = PROJECT_DIR / "customer_additions.csv"

# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static"
)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    secrets.token_hex(32)
)

DEMO_EMAIL = os.environ.get("DEMO_EMAIL")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD")

# --------------------------------------------------
# LOAD BANK DATASET
# --------------------------------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Bank dataset not found: {DATASET_PATH}"
    )
bank_df = pd.read_csv(DATASET_PATH)
bank_df.columns = bank_df.columns.str.strip()

# Load separately stored new customers, if available
if ADDITIONS_PATH.exists():
    additions_df = pd.read_csv(ADDITIONS_PATH)
    additions_df = additions_df.reindex(columns=bank_df.columns)
    bank_df = pd.concat(
        [bank_df, additions_df],
        ignore_index=True
    )

print("Bank dataset loaded:", bank_df.shape)
# --------------------------------------------------
# FRONTEND ROUTE
# --------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")
# --------------------------------------------------
# LOGIN API
# --------------------------------------------------

@app.route("/api/auth/login", methods=["POST"])
def login():

    if not DEMO_EMAIL or not DEMO_PASSWORD:
        return jsonify({
            "status": "error",
            "message": "Demo login credentials are not configured."
        }), 500

    data = request.get_json(silent=True) or {}

    email = data.get("email", "").strip()
    password = data.get("password", "")

    if email != DEMO_EMAIL or password != DEMO_PASSWORD:
        return jsonify({
            "status": "error",
            "message": "Invalid email or password."
        }), 401

    user_data = {
        "name": "Demo Administrator",
        "bankId": "DEMO"
    }

    session["user"] = user_data

    return jsonify({
        "status": "success",
        "data": user_data
    }), 200 
# --------------------------------------------------
# LOGOUT API
# --------------------------------------------------

@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({
        "status": "success",
        "message": "Logged out successfully."
    }), 200
# --------------------------------------------------
# CUSTOMER DATA API
# --------------------------------------------------

@app.route("/api/bank/<bank_id>", methods=["GET"])
def get_bank_customers(bank_id):

    # Check login session
    if "user" not in session:
        return jsonify({
            "status": "error",
            "message": "Please log in first."
        }), 401

    # This project currently uses one demo portfolio
    if bank_id != "DEMO":
        return jsonify({
            "status": "error",
            "message": "Portfolio not found."
        }), 404

    required_columns = [
        "CustomerId",
        "Surname",
        "Tenure",
        "Balance"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in bank_df.columns
    ]

    if missing_columns:
        return jsonify({
            "status": "error",
            "message": f"Missing dataset columns: {missing_columns}"
        }), 500

    customers = []

    for _, row in bank_df.iterrows():
        customers.append({
            "customerId": int(row["CustomerId"]),
            "name": str(row["Surname"]),
            "tenure": int(row["Tenure"]),
            "balance": round(float(row["Balance"]), 2)
        })

    return jsonify({
        "status": "success",
        "bank_name": "Demo Portfolio",
        "customers": customers
    }), 200
# --------------------------------------------------
# MODEL PREDICTION API
# --------------------------------------------------

PREDICTIONS_CACHE = None


def get_all_predictions():

    global PREDICTIONS_CACHE

    # Reuse predictions after the first calculation
    if PREDICTIONS_CACHE is not None:
        return PREDICTIONS_CACHE

    # Load the trained model and bank preprocessor
    model, preprocessor, config = load_inference_components()

    input_features = config["input_features"]

    # Verify required columns
    missing = [
        col for col in input_features
        if col not in bank_df.columns
    ]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    # Prepare all customers using the trained preprocessing pipeline
    customer_data = bank_df[input_features].copy()

    numeric_features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Satisfaction Score",
        "Point Earned"
    ]

    for feature in numeric_features:
        customer_data[feature] = pd.to_numeric(
            customer_data[feature],
            errors="raise"
        )

    if customer_data.isnull().any().any():
        raise ValueError("Customer dataset contains missing values.")

    numeric_values = customer_data[numeric_features].to_numpy(
        dtype=float
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError("Customer dataset contains invalid numeric values.")

    # Apply the same preprocessing used during model training
    processed_data = preprocessor.transform(customer_data)

    # Batch prediction
    probabilities = model.predict(
        np.asarray(processed_data, dtype=np.float32),
        batch_size=256,
        verbose=0
    ).reshape(-1)

    threshold = float(config["threshold"])

    predictions = []

    for customer_id, probability in zip(
        bank_df["CustomerId"],
        probabilities
    ):

        probability = float(probability)

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability.")

        predicted_class = int(probability >= threshold)

        if probability < 0.30:
            risk_level = "Low"
        elif probability < threshold:
            risk_level = "Medium"
        else:
            risk_level = "High"

        predictions.append({
            "customerId": int(customer_id),
            "prob": round(probability * 100, 2),
            "risk_level": risk_level,
            "prediction": (
                "Churn" if predicted_class == 1 else "No Churn"
            )
        })

    PREDICTIONS_CACHE = predictions

    return predictions


@app.route("/api/predict/all/<bank_id>", methods=["GET"])
def predict_all_customers(bank_id):

    if "user" not in session:
        return jsonify({
            "status": "error",
            "message": "Please log in first."
        }), 401

    if bank_id != "DEMO":
        return jsonify({
            "status": "error",
            "message": "Portfolio not found."
        }), 404

    try:
        predictions = get_all_predictions()
        return jsonify(predictions), 200

    except Exception:
        app.logger.exception("Customer prediction failed")
        return jsonify({
            "status": "error",
            "message": "Unable to generate customer predictions."
        }), 500

# --------------------------------------------------
# CREATE CUSTOMER + INDIVIDUAL CHURN PREDICTION
# --------------------------------------------------

@app.route("/api/customers/<bank_id>", methods=["POST"])
def create_customer(bank_id):

    global bank_df, PREDICTIONS_CACHE

    if "user" not in session:
        return jsonify({"message": "Please log in first."}), 401

    if bank_id != "DEMO":
        return jsonify({"message": "Portfolio not found."}), 404

    try:
        data = request.get_json(silent=True) or {}
        name = str(data.get("name", "")).strip()

        if not name or len(name) > 100:
            return jsonify({"message": "Enter a valid customer name."}), 400

        required = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary", "Satisfaction Score", "Card Type",
            "Point Earned"
        ]

        missing = [
            field for field in required
            if field not in data or data[field] in (None, "")
        ]

        if missing:
            return jsonify({
                "message": f"Missing fields: {', '.join(missing)}"
            }), 400

        customer = {field: data[field] for field in required}

        # Validate supported categorical values
        if customer["Geography"] not in ["France", "Germany", "Spain"]:
            return jsonify({"message": "Invalid geography."}), 400

        if customer["Gender"] not in ["Female", "Male"]:
            return jsonify({"message": "Invalid gender."}), 400

        if customer["Card Type"] not in [
            "DIAMOND", "GOLD", "PLATINUM", "SILVER"
        ]:
            return jsonify({"message": "Invalid card type."}), 400

        # Validate numeric values and ranges
        numeric_ranges = {
            "CreditScore": (350, 850),
            "Age": (18, 92),
            "Tenure": (0, 10),
            "Balance": (0, 250898.09),
            "NumOfProducts": (1, 4),
            "HasCrCard": (0, 1),
            "IsActiveMember": (0, 1),
            "EstimatedSalary": (11.58, 199992.48),
            "Satisfaction Score": (1, 5),
            "Point Earned": (119, 1000)
        }

        for field, (minimum, maximum) in numeric_ranges.items():
            value = float(customer[field])

            if not np.isfinite(value) or not minimum <= value <= maximum:
                return jsonify({
                    "message": f"{field} must be between {minimum} and {maximum}."
                }), 400

            customer[field] = value

        for field in ["CreditScore", "Age", "Tenure",
                      "NumOfProducts", "HasCrCard",
                      "IsActiveMember", "Satisfaction Score",
                      "Point Earned"]:
            if not customer[field].is_integer():
                return jsonify({"message": f"{field} must be a whole number."}), 400
            customer[field] = int(customer[field])

        # Generate a unique customer ID
        customer_id = int(pd.to_numeric(
            bank_df["CustomerId"], errors="raise"
        ).max()) + 1

        customer["CustomerId"] = customer_id
        customer["Surname"] = name

        # Predict before saving the customer
        prediction_input = {
            field: customer[field] for field in required
        }
        prediction_input["CustomerId"] = customer_id

        result = predict_churn(prediction_input)

        # Preserve all original dataset columns
        new_row = {column: np.nan for column in bank_df.columns}

        for field, value in customer.items():
            new_row[field] = value

        new_customer_df = pd.DataFrame(
            [new_row], columns=bank_df.columns
        )

        # Save additions separately from the original dataset
        if ADDITIONS_PATH.exists():
            additions_df = pd.read_csv(ADDITIONS_PATH)
            additions_df = additions_df.reindex(columns=bank_df.columns)
        else:
            additions_df = pd.DataFrame(columns=bank_df.columns)

        additions_df = pd.concat(
            [additions_df, new_customer_df],
            ignore_index=True
        )

        additions_df.to_csv(ADDITIONS_PATH, index=False)

        # Update in-memory customer records and invalidate cache
        bank_df = pd.concat(
            [bank_df, new_customer_df],
            ignore_index=True
        )
        PREDICTIONS_CACHE = None

        return jsonify({
            "status": "success",
            "message": "Customer created and churn risk assessed.",
            "customer": {
                "customerId": customer_id,
                "name": name,
                "tenure": customer["Tenure"],
                "balance": customer["Balance"]
            },
            "prediction": result
        }), 201

    except (ValueError, TypeError, OverflowError) as error:
        return jsonify({"message": str(error)}), 400

    except Exception:
        app.logger.exception("Customer creation failed")
        return jsonify({
            "message": "Unable to create customer. Please try again."
        }), 500

# --------------------------------------------------
# NEXA AI CHAT API
# --------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def nexa_chat():

    if "user" not in session:
        return jsonify({
            "reply": "Please log in to access Nexa AI."
        }), 401

    data = request.get_json(silent=True) or {}

    message = str(data.get("message", "")).strip()
    bank_id = data.get("bankId")

    if not message:
        return jsonify({
            "reply": "Please enter a question."
        }), 400

    if bank_id != "DEMO":
        return jsonify({
            "reply": "I couldn't identify the requested portfolio."
        }), 404

    try:
        predictions = get_all_predictions()

        total = len(predictions)
        high = sum(p["risk_level"] == "High" for p in predictions)
        medium = sum(p["risk_level"] == "Medium" for p in predictions)
        low = sum(p["risk_level"] == "Low" for p in predictions)

        high_percent = round((high / total) * 100, 2) if total else 0

        question = message.lower()

        if any(word in question for word in ["high risk", "high-risk", "critical"]):
            top_customers = sorted(
                [p for p in predictions if p["risk_level"] == "High"],
                key=lambda p: p["prob"],
                reverse=True
            )[:5]

            customer_list = "\n".join(
                f"Customer {p['customerId']}: {p['prob']}% churn probability"
                for p in top_customers
            )

            reply = (
                f"Your portfolio has {high} high-risk customers.\n\n"
                f"Here are the five highest-risk customers:\n"
                f"{customer_list}"
            )

        elif any(word in question for word in ["medium risk", "medium-risk"]):
            reply = (
                f"There are {medium} medium-risk customers "
                f"in your portfolio."
            )

        elif any(word in question for word in ["low risk", "low-risk", "safe"]):
            reply = (
                f"There are {low} low-risk customers "
                f"in your portfolio."
            )

        elif any(word in question for word in ["accuracy", "performance", "model"]):
            reply = (
                "The selected model is Random Transfer Fine-tuned. "
                "Its held-out test accuracy was 86.73%, F1-score was "
                "68.06%, and ROC-AUC was 89.65%. "
                "These are evaluation results, not live dashboard accuracy."
            )

        elif any(word in question for word in ["total", "customers", "portfolio", "summary"]):
            reply = (
                f"Portfolio summary:\n"
                f"Total customers: {total}\n"
                f"High risk: {high} ({high_percent}%)\n"
                f"Medium risk: {medium}\n"
                f"Low risk: {low}"
            )

        else:
            reply = (
                "I can help you analyze your portfolio. Try asking about "
                "the portfolio summary, high-risk customers, medium-risk "
                "customers, low-risk customers, or model performance."
            )

        return jsonify({"reply": reply}), 200

    except Exception:
        app.logger.exception("Nexa AI request failed")
        return jsonify({
            "reply": "I couldn't retrieve the analytics right now. Please try again."
        }), 500
# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    print("\nChurn Predictor Dashboard")
    print("Frontend: http://127.0.0.1:5000")
    print("Portfolio: Demo Portfolio")
    print("Press CTRL+C to stop the server.\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )