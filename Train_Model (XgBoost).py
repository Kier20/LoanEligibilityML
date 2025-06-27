#XGBoost
# train_xgboost.py

import os, json, logging, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Setup
logging.basicConfig(level=logging.INFO)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def cross_validate_xgboost(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_model, best_accuracy = None, 0.0
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logging.info(f"Fold {fold + 1}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_metrics.append({
            'Fold': fold + 1,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            joblib.dump(best_model, "models/xgboost_best_model.pkl")
            logging.info(f"New best model saved with accuracy: {best_accuracy:.4f}")

    pd.DataFrame(fold_metrics).to_csv("reports/xgboost_training_report.csv", index=False)
    return best_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    with open("reports/xgboost_test_evaluation.json", "w") as f:
        json.dump({"accuracy": acc, "classification_report": report}, f, indent=4)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/xgboost_confusion_matrix.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), importance[sorted_idx], align='center')
    plt.xticks(range(len(feature_names)), np.array(feature_names)[sorted_idx], rotation=45, ha='right')
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig("plots/xgboost_feature_importance.png")
    plt.close()

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/preprocessed_data.csv")

        le_home = LabelEncoder()
        le_intent = LabelEncoder()
        df['person_home_ownership'] = le_home.fit_transform(df['person_home_ownership'])
        df['loan_intent'] = le_intent.fit_transform(df['loan_intent'])
        joblib.dump(le_home, "models/encoder_person_home_ownership.pkl")
        joblib.dump(le_intent, "models/encoder_loan_intent.pkl")

        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        X = df.drop("loan_status", axis=1).values
        y = df["loan_status"].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, "models/scaler.pkl")

        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1,
                                                                  stratify=y, random_state=42)

        logging.info("Starting XGBoost cross-validation...")
        best_model = cross_validate_xgboost(X_trainval, y_trainval)

        logging.info("Evaluating best model on test set...")
        evaluate_model(best_model, X_test, y_test)

        feature_names = df.drop("loan_status", axis=1).columns.tolist()
        plot_feature_importance(best_model, feature_names)
        logging.info("Feature importance plot saved.")

    except Exception as e:
        logging.error(f"Training error: {e}")
