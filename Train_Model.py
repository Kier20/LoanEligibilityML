import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# --- Ensure folders exist ---
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# --- Load Data ---
data = pd.read_csv("data/preprocessed_data.csv")
X = data.drop(columns=["loan_status"])
y = data["loan_status"]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
summary = []

fold = 1
for train_index, val_index in kf.split(X, y):
    print(f"\n--- Fold {fold} ---")

    # Split data
    X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
    y_train, y_val = y.iloc[train_index].copy(), y.iloc[val_index].copy()

    # SMOTE before scaling
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale after SMOTE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)

    # Save scaler for Fold 1
    if fold == 1:
        joblib.dump(scaler, "models/scaler.pkl")

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # Train model
    history = model.fit(
        X_train_scaled, y_train_resampled,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # Predict and evaluate
    y_pred = model.predict(X_val_scaled).flatten()
    y_class = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_val, y_class)
    report = classification_report(y_val, y_class, output_dict=True)
    conf = confusion_matrix(y_val, y_class)

    # --- Save Confusion Matrix Plot ---
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", xticklabels=["Denied", "Approved"], yticklabels=["Denied", "Approved"])
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"plots/confusion_matrix_fold_{fold}.png")
    plt.close()


    # --- Save Classification Report as Text ---
    report_text = classification_report(y_val, y_class)
    with open(f"reports/classification_report_fold_{fold}.txt", "w") as f:
        f.write(f"Fold {fold} Classification Report\n")
        f.write(report_text)

    # --- Save Metrics Summary Row ---
    summary.append({
        "Fold": fold,
        "Accuracy": acc,
        "Precision (Approved)": report['1']['precision'],
        "Recall (Approved)": report['1']['recall'],
        "F1-Score (Approved)": report['1']['f1-score'],
        "Support (Approved)": report['1']['support']
    })

    # Save model on Fold 1
    if fold == 1:
        model.save("models/best_model.h5")

    fold += 1

# --- Save Overall Metrics Summary CSV ---
summary_df = pd.DataFrame(summary)
summary_df.to_csv("reports/metrics_summary.csv", index=False)

# --- Plot Evaluation Metrics Across Folds ---
metrics_to_plot = ["Accuracy", "Precision (Approved)", "Recall (Approved)", "F1-Score (Approved)"]

plt.figure(figsize=(10, 6))
for metric in metrics_to_plot:
    plt.plot(summary_df["Fold"], summary_df[metric], marker='o', label=metric)

plt.title("Evaluation Metrics Across Folds")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/metrics_across_folds.png")
plt.close()

# --- Bar Plot of Average Metrics ---
avg_metrics = summary_df[metrics_to_plot].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x=avg_metrics.index, y=avg_metrics.values, palette="viridis")
plt.title("Average Evaluation Metrics Across All Folds")
plt.ylabel("Average Score")
plt.ylim(0, 1.05)
for i, v in enumerate(avg_metrics.values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.savefig("plots/average_metrics.png")
plt.close()
print("\n✅ All folds completed. Summary saved to reports/metrics_summary.csv")
