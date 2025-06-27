import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# Load model
try:
    model = tf.keras.models.load_model("models/best_model.h5", compile=False)
except Exception as e:
    print("Error loading model:", e)

# Load preprocessors
scaler = joblib.load("models/scaler.pkl")
le_home = joblib.load("models/encoder_person_home_ownership.pkl")
le_intent = joblib.load("models/encoder_loan_intent.pkl")

# Sanity check
print("Home ownership classes:", le_home.classes_)
print("Loan intent classes:", le_intent.classes_)

# Construct input
sample_data = {
    "person_age": 34,                      # typical working adult
    "person_income": 72000,                # solid middle-class income
    "person_emp_length": 7,                # decent job stability
    "person_home_ownership": 2,            # 'OWN', from encoder
    "loan_intent": 0,                      # 'DEBTCONSOLIDATION', from encoder
    "loan_grade": 2,                       # Grade C (lower risk)
    "cb_person_cred_hist_length": 8,       # enough history
    "cb_person_default_on_file": 0,        # no defaults
    "loan_amnt": 9000                      # modest loan
}

# DataFrame & scaling
input_df = pd.DataFrame([sample_data])
input_df = input_df[scaler.feature_names_in_]

print("\n📥 Input DataFrame:\n", input_df)

scaled = scaler.transform(input_df)
print("\n⚙️ Scaled Input:\n", scaled)

# Prediction
output = model.predict(scaled)
prob = float(output[0][0])

print(f"\n🔍 Raw Output: {prob:.6f}")
print("✅ Valid" if np.isfinite(prob) else "❌ Invalid")

# Decision
threshold = 0.5
decision = "Approved" if prob > threshold else "Denied"
print(f"\n🧾 Final Decision: {decision}")
