import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_file, output_file):
    # Load the dataset
    dataset = pd.read_csv(input_file)

    features_to_keep = [
        'person_age', 'person_income', 'person_emp_length',
        'person_home_ownership', 'loan_intent', 'loan_grade',
        'cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_amnt', 'loan_status'
    ]
    dataset = dataset[features_to_keep]

    # Columns to encode
    categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

    # Initialize label encoders and encode the specified columns
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        dataset[col] = label_encoders[col].fit_transform(dataset[col])

    # Handle missing values
    numeric_columns_with_missing = ['person_emp_length']
    for col in numeric_columns_with_missing:
        median_value = dataset[col].median()
        dataset[col].fillna(median_value, inplace=True)

    # Save the cleaned dataset
    dataset.to_csv(output_file, index=False)

    # Return the dataset and encoders for further use
    return dataset, label_encoders

if __name__ == "__main__":
    input_file = 'data/credit_risk_dataset.csv'  # Raw dataset
    output_file = 'data/preprocessed_data.csv'  # Cleaned dataset
    preprocess_data(input_file, output_file)
