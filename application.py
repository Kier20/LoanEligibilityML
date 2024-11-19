from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model, scaler, and model columns
model = joblib.load('loan_eligibility_model.pkl')  # Trained model
scaler = joblib.load('scaler.pkl')  # Scaler for normalization
model_columns = joblib.load('model_columns.pkl')  # Feature columns


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict loan eligibility."""
    try:
        # Parse input JSON into a Python dictionary
        applicant_data = request.json
        if not applicant_data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input data to a DataFrame
        applicant_df = pd.DataFrame([applicant_data])

        # Feature Engineering: Add Total Income and Loan-to-Income Ratio
        applicant_df['TotalIncome'] = applicant_df['ApplicantIncome'] + applicant_df['CoapplicantIncome']
        applicant_df['LoanIncomeRatio'] = applicant_df['LoanAmount'] / (applicant_df['TotalIncome'] + 1)

        # Align input columns with training data columns
        applicant_df = pd.get_dummies(applicant_df)
        applicant_df = applicant_df.reindex(columns=model_columns, fill_value=0)

        # Normalize numerical features
        numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome', 'LoanIncomeRatio']
        applicant_df[numerical_columns] = scaler.transform(applicant_df[numerical_columns])

        # Predict loan eligibility and probability
        prediction = int(model.predict(applicant_df)[0])  # Convert numpy.int64 to int
        prediction_prob = float(model.predict_proba(applicant_df)[:, 1][0])  # Convert numpy.float64 to float

        # Construct response
        result = {
            "prediction": "Eligible" if prediction == 1 else "Not Eligible",
            "probability": round(prediction_prob, 2)
        }
        return jsonify(result), 200

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
