import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

# PATH & KONSTANTA
PREPROCESSING_DIR = './examScorePrediction_preprocessing/' 
DATA_PATH = PREPROCESSING_DIR + 'data_preprocessing.csv'

# 1. LOAD DATA & SPLIT
print("Memuat dan Membagi Data...")
df_final = pd.read_csv(DATA_PATH)

X = df_final.drop(columns=['target', 'is_train'])
y = df_final['target']

X_train = X[df_final['is_train'] == True]
X_test = X[df_final['is_train'] == False]
y_train = y[df_final['is_train'] == True]
y_test = y[df_final['is_train'] == False]

y_train = y_train.reindex(X_train.index)
y_test = y_test.reindex(X_test.index)

# 2. SETUP MLFLOW

# Disable autologging
mlflow.sklearn.autolog(disable=True)

# 3. TRAINING MODEL
print("Melatih Model Logistic Regression...")

# Buat dan latih model
model = LogisticRegression(
    random_state=42, 
    max_iter=500,
)

# Training
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)

# Hitung metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted', zero_division=0
)

# 4. LOG KE MLFLOW
# Log parameters
mlflow.log_param("max_iter", 500)
mlflow.log_param("random_state", 42)

# Log metrics
mlflow.log_metric("test_accuracy", accuracy)
mlflow.log_metric("test_precision_weighted", precision)
mlflow.log_metric("test_recall_weighted", recall)
mlflow.log_metric("test_f1_score_weighted", f1_score)

# Log model ke MLflow
mlflow.sklearn.log_model(model, "model")

# 5. SIMPAN MODEL LOKAL
joblib.dump(model, PREPROCESSING_DIR + 'lr_model.joblib')

print(f"\nTraining Selesai")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"Model disimpan ke: {PREPROCESSING_DIR}lr_model.joblib")