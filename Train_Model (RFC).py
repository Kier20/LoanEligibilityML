#Random Forest Classifier

# train_random_forest.py

import os, json, joblib, logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# Setup
logging.basicConfig(level=logging.INFO)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def train_rf_model(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    with open("reports/rf_test_evaluation.json", "w") as f:
        json.dump({"accuracy": accuracy, "classification_report": class_report}, f, indent=4)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/rf_confusion_matrix.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), importance[sorted_idx], align='center')
    plt.xticks(range(len(feature_names)), np.array(feature_names)[sorted_idx], rotation=45, ha='right')
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig("plots/rf_feature_importance.png")
    plt.close()

def cross_validate_rf(X, y, feature_names, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics, best_model, best_score = [], None, 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logging.info(f"Fold {fold + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        model = train_rf_model(X_train, y_train)
        y_pred = model.predict(X_val)

        val_accuracy = accuracy_score(y_val, y_pred)
        val_precision = precision_score(y_val, y_pred, zero_division=0)
        val_recall = recall_score(y_val, y_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_metrics.append({
            'Fold': fold + 1,
            'Accuracy': val_accuracy,
            'Precision': val_precision,
            'Recall': val_recall,
            'F1 Score': val_f1
        })

        if val_accuracy > best_score:
            best_score = val_accuracy
            best_model = model
            joblib.dump(model, "models/best_rf_model.pkl")
            logging.info(f"New best RF model saved with accuracy: {best_score:.4f}")

    pd.DataFrame(fold_metrics).to_csv("reports/rf_training_report.csv", index=False)
    return best_model

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/preprocessed_data.csv")

        le_home = LabelEncoder()
        le_intent = LabelEncoder()
        df['person_home_ownership'] = le_home.fit_transform(df['person_home_ownership'])
        df['loan_intent'] = le_intent.fit_transform(df['loan_intent'])
        joblib.dump(le_home, "models/rf_encoder_home.pkl")
        joblib.dump(le_intent, "models/rf_encoder_intent.pkl")

        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        feature_names = df.drop("loan_status", axis=1).columns.tolist()
        X = df[feature_names].values
        y = df["loan_status"].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, "models/rf_scaler.pkl")

        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

        logging.info("Starting cross-validation...")
        best_model = cross_validate_rf(X_trainval, y_trainval, feature_names)

        logging.info("Evaluating best RF model on test set...")
        evaluate_model(best_model, X_test, y_test)

        plot_feature_importance(best_model, feature_names)
        logging.info("Feature importance plot saved.")

    except Exception as e:
        logging.error(f"Training error: {e}")