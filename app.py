from flask import Flask, request, jsonify
from flask import render_template
from flask import send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import shap
import traceback

app = Flask(__name__, template_folder="templates")
CORS(app, origins="*", supports_credentials=True)

# --- Custom focal loss used during model training ---
def sigmoid_focal_crossentropy(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred)
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
    ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    w = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(w * ce)

# --- Load model and preprocessing tools ---
model = tf.keras.models.load_model("models/best_model.h5", custom_objects={"sigmoid_focal_crossentropy": sigmoid_focal_crossentropy})
scaler = joblib.load("models/scaler.pkl")
le_home = joblib.load("models/encoder_person_home_ownership.pkl")
le_intent = joblib.load("models/encoder_loan_intent.pkl")
le_grade = joblib.load("models/encoder_loan_grade.pkl")
le_default = joblib.load("models/encoder_cb_person_default_on_file.pkl")

# --- SHAP Setup ---
sample_input = np.zeros((1, 9))
try:
    explainer = shap.DeepExplainer(model, sample_input)
except:
    explainer = shap.KernelExplainer(model.predict, sample_input)

# --- Hidden Charges ---
def calculate_hidden_charges(loan_amount: float) -> dict:
    processing_fee = max(1500, 0.02 * loan_amount)
    dst = (loan_amount / 200) * 1.50 if loan_amount >= 250000 else 0
    disbursement_fee = 1500 if loan_amount >= 50000 else 1000
    total = processing_fee + dst + disbursement_fee
    return {
        "processing_fee": round(processing_fee, 2),
        "documentary_stamp_tax": round(dst, 2),-+
        "disbursement_fee": disbursement_fee,
        "total_hidden_charges": round(total, 2)
    }

# --- Fairness & Bias Logic ---
def apply_bias_overrides(prob, data_dict):
    overrides = []
    adjusted = prob

    if data_dict["person_home_ownership"] in [2, 3]:
        adjusted *= 0.9
        overrides.append("⚠️ Applicant does not fully own a home (OWN/RENT)")
    if data_dict["loan_intent"] in [4, 5]:
        adjusted *= 0.85
        overrides.append("⚠️ Loan intent is PERSONAL or SMALL BUSINESS, considered high-risk")
    if data_dict["person_age"] < 25:
        adjusted *= 0.92
        overrides.append("⚠️ Applicant is under 25 years old")
    if data_dict["person_income"] < 15000:
        adjusted *= 0.88
        overrides.append("⚠️ Applicant income is below PHP 15,000")
    if data_dict["person_emp_length"] < 2:
        adjusted *= 0.9
        overrides.append("⚠️ Employment length is under 2 years")
    if data_dict["loan_grade"] >= 4:
        adjusted *= 0.8
        overrides.append("⚠️ Loan grade is D or worse")

    return adjusted, overrides

# --- Friendly SHAP Labels ---
friendly_map = {
    "person_income": ("💰 Monthly Income", "Your income increased your approval chances."),
    "person_home_ownership": ("🏠 Home Ownership", "Owning or renting contributed positively to your profile."),
    "loan_amnt": ("💳 Loan Amount", "The amount you applied for fits well with your financial profile."),
    "loan_intent": ("🎯 Loan Purpose", "Loans for personal or small business use are slightly riskier."),
    "person_age": ("📅 Age", "Your age had a small impact on approval."),
    "cb_person_cred_hist_length": ("📈 Credit History Length", "Your credit history had little effect."),
    "loan_grade": ("📂 Loan Grade", "Your loan grade was acceptable."),
    "cb_person_default_on_file": ("🧾 Default History", "No recent defaults helped your application."),
    "person_emp_length": ("💼 Employment Length", "Stable employment helped increase your chances.")
}

# --- Prediction Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {key: request.form[key] for key in request.form}
        input_data["person_age"] = float(input_data["person_age"])
        input_data["person_income"] = float(input_data["person_income"])
        input_data["loan_amnt"] = float(input_data["loan_amnt"])
        input_data["cb_person_cred_hist_length"] = float(input_data["cb_person_cred_hist_length"])
        input_data["person_emp_length"] = float(input_data["person_emp_length"])

        home_ownership_encoded = le_home.transform([input_data["person_home_ownership"]])[0]
        loan_intent_encoded = le_intent.transform([input_data["loan_intent"]])[0]
        loan_grade_encoded = le_grade.transform([input_data["loan_grade"]])[0]
        default_encoded = le_default.transform([input_data["cb_person_default_on_file"]])[0]

        X = pd.DataFrame([[
            input_data["person_age"],
            input_data["person_income"],
            input_data["person_emp_length"],
            home_ownership_encoded,
            loan_intent_encoded,
            loan_grade_encoded,
            input_data["cb_person_cred_hist_length"],
            default_encoded,
            input_data["loan_amnt"]
        ]], columns=[
            "person_age", "person_income", "person_emp_length", "person_home_ownership",
            "loan_intent", "loan_grade", "cb_person_cred_hist_length", "cb_person_default_on_file",
            "loan_amnt"
        ])

        X_scaled = scaler.transform(X)
        raw_proba = float(model.predict(X_scaled)[0][0])
        prediction = "Approved" if raw_proba >= 0.5 else "Denied"

        bias_data = {
            "person_age": input_data["person_age"],
            "person_income": input_data["person_income"],
            "person_emp_length": input_data["person_emp_length"],
            "person_home_ownership": home_ownership_encoded,
            "loan_intent": loan_intent_encoded,
            "loan_grade": loan_grade_encoded
        }
        adjusted_proba, bias_flags = apply_bias_overrides(raw_proba, bias_data)

        # --- SHAP Feature Impact Explanation ---
        # --- SHAP Feature Impact Explanation ---
        try:
            shap_vals = explainer.shap_values(X_scaled)  # still use scaled for model compatibility
            friendly_explanations = []
            for col, val in zip(X.columns, shap_vals[0]):
                impact = "👍 Positive" if val > 0.01 else "👎 Negative" if val < -0.01 else "➖ Neutral"
                name, desc = friendly_map.get(col, (col, "No description available."))
                friendly_explanations.append({
                    "feature": name,
                    "impact": impact,
                    "value": round(float(val), 4),
                    "explanation": desc
                })
            print("SHAP output:", friendly_explanations)
        except Exception as e:
            print("[!] SHAP Error:", e)
            friendly_explanations = []
        approval_reasons = []
        denial_reasons = []
        if prediction == "Approved":
            if input_data["person_income"] > 5000: approval_reasons.append("High income")
            if input_data["cb_person_cred_hist_length"] > 5: approval_reasons.append("Strong credit history")
        else:
            if input_data["person_income"] < 2000: denial_reasons.append("Low income")
            if input_data["cb_person_cred_hist_length"] < 2: denial_reasons.append("Short credit history")

        hidden_charges_flag = input_data["loan_amnt"] > (input_data["person_income"] * 0.8)
        hidden_charges = calculate_hidden_charges(input_data["loan_amnt"])

        print("\n=== MODEL DEBUG ===")
        print("Scaled Input:", X_scaled)
        print(f"Raw Probability: {raw_proba:.4f}")
        print(f"Adjusted (Fairness) Probability: {adjusted_proba:.4f}")
        print("====================\n")

        return jsonify({
            "prediction": prediction,
            "probability": round(raw_proba, 4),
            "adjusted_probability": round(adjusted_proba, 4),
            "approval_reasons": approval_reasons,
            "denial_reasons": denial_reasons,
            "explanations": friendly_explanations,
            "hidden_charges_flag": hidden_charges_flag,
            "hidden_charges": hidden_charges,
            "potential_biases": bias_flags
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render sets the PORT env variable
    app.run(host="0.0.0.0", port=port)
    
@app.route("/")
def index():
    return render_template("index.html")


