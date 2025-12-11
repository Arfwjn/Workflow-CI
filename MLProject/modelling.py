import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

# PATH & KONSTANTA
PREPROCESSING_DIR = './examScorePrediction_preprocessing/' 
DATA_PATH = PREPROCESSING_DIR + 'data_preprocessing.csv'

print("="*60)
print("Starting Model Training Pipeline")
print("="*60)

# 1. LOAD DATA & SPLIT
df_final = pd.read_csv(DATA_PATH)

X = df_final.drop(columns=['target', 'is_train'])
y = df_final['target']

X_train = X[df_final['is_train'] == True]
X_test = X[df_final['is_train'] == False]
y_train = y[df_final['is_train'] == True]
y_test = y[df_final['is_train'] == False]

y_train = y_train.reindex(X_train.index)
y_test = y_test.reindex(X_test.index)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
print(f"Features: {list(X.columns)}")

# 2. SETUP MLFLOW
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_Workflow_Klasifikasi_LogisticRegression")

# Disable autologging
mlflow.sklearn.autolog(disable=True)

# 3. TRAINING MODEL

# Create model
model = LogisticRegression(
    random_state=42, 
    max_iter=500,
)

# Training
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted', zero_division=0
)

print(f"Training completed")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# 4. LOG TO MLFLOW
print("\n[4/4] Logging to MLflow...")

# Log parameters
mlflow.log_param("max_iter", 500)
mlflow.log_param("random_state", 42)
mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_param("train_size", len(X_train))
mlflow.log_param("test_size", len(X_test))

# Log metrics
mlflow.log_metric("test_accuracy", accuracy)
mlflow.log_metric("test_precision_weighted", precision)
mlflow.log_metric("test_recall_weighted", recall)
mlflow.log_metric("test_f1_score_weighted", f1_score)

# Log model to MLflow
mlflow.sklearn.log_model(
    model, 
    "model",
    registered_model_name="ExamScorePredictor"
)

# 5. SAVE MODEL LOCALLY
model_path = PREPROCESSING_DIR + 'lr_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")