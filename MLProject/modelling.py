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

# 2. SETUP MLFLOW & AUTOLOGGING 

mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment("CI_Workflow_Klasifikasi_LogisticRegression")

# Autologging
mlflow.sklearn.autolog(disable=True)

# 3. TRAINING MODE

print("Melatih Model Logistic Regression (Autolog)...")

# Model Klasifikasi Multikelas
model = LogisticRegression(
    random_state=42, 
    max_iter=500,
)

current_run_id = "N/A - Run Failed"

with mlflow.start_run(run_name="CI_Workflow_LogisticRegression_Manual_Fix") as run:
    print("Melatih Model Logistic Regression (Manual Log)...")

    # Training model
    model.fit(X_train, y_train) 
    # Prediksi
    y_pred = model.predict(X_test)
    
    # NOTE: menggunakan logging manual karena auto logging selalu gagal
    mlflow.log_param("max_iter", 500)
    mlflow.log_param("random_state", 42)
    
    # Akurasi
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", accuracy)

    # Presisi, Recall, F1-Score
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    mlflow.log_metric("test_precision_weighted", precision)
    mlflow.log_metric("test_recall_weighted", recall)
    mlflow.log_metric("test_f1_score_weighted", f1_score)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Simpan model
    joblib.dump(model, PREPROCESSING_DIR + 'lr_model.joblib')
    
    current_run_id = run.info.run_id

print(f"Model Training Selesai. Run ID: {current_run_id}")