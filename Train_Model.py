import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load dataset
data = pd.read_csv('Automated-Loan-Eligibility-Prediction-main/data/train.csv')
data = data.drop(columns=['Loan_ID'])

# Handle '3+' in 'Dependents' column
data['Dependents'] = data['Dependents'].replace('3+', '3')
data['Dependents'] = pd.to_numeric(data['Dependents'], errors='coerce')
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)

# Replace categorical columns with numeric values
data['Married'] = data['Married'].replace({'No': 0, 'Yes': 1}).astype(float)
data['Education'] = data['Education'].replace({'Not Graduate': 0, 'Graduate': 1}).astype(float)
data['Self_Employed'] = data['Self_Employed'].replace({'No': 0, 'Yes': 1}).astype(float)
data['Loan_Status'] = data['Loan_Status'].replace({'N': 0, 'Y': 1}).astype(float)

# Apply one-hot encoding to categorical features
data = pd.get_dummies(data, columns=['Gender', 'Married', 'Education', 'Property_Area'])

# Feature Engineering: Debt-to-Income Ratio (DTI)
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['LoanIncomeRatio'] = data['LoanAmount'] / (data['TotalIncome'] + 1)

# Log transformation to reduce skewness
data['ApplicantIncome'] = np.log1p(data['ApplicantIncome'])
data['CoapplicantIncome'] = np.log1p(data['CoapplicantIncome'])
data['LoanAmount'] = np.log1p(data['LoanAmount'])

# Handle missing values
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')
data['LoanAmount'] = numerical_imputer.fit_transform(data[['LoanAmount']])
data = pd.DataFrame(categorical_imputer.fit_transform(data), columns=data.columns)

# Normalize numerical columns
scaler = MinMaxScaler()
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome', 'LoanIncomeRatio']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Separate features and target
X = data.drop(['Loan_Status'], axis=1)
y = data['Loan_Status']

# Perform SMOTETomek to handle class imbalance
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X, y)

# Models for Voting
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
gbc = GradientBoostingClassifier(random_state=42)

# Voting Classifier
ensemble_model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('gbc', gbc)], voting='soft')

# Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble_model, X_res, y_res, cv=skf, scoring='accuracy')

# Train-Test Split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
ensemble_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# Display results
print("Cross-Validation Accuracy: {:.4f}".format(cv_scores.mean()))
print("Test Accuracy: {:.4f}".format(accuracy))
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", confusion_mat)

# Save the trained model
joblib.dump(ensemble_model, 'loan_eligibility_model.pkl')

# Save the scaler for preprocessing
joblib.dump(scaler, 'scaler.pkl')

# Save the columns used in the model for future reference
joblib.dump(X.columns, 'model_columns.pkl')

print("Model, scaler, and columns saved successfully!")