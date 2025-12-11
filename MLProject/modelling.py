import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
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
mlflow.sklearn.autolog() 

# 3. TRAINING MODE

print("Melatih Model Logistic Regression (Autolog)...")

# Model Klasifikasi Multikelas
model = LogisticRegression(
    random_state=42, 
    max_iter=500,
)

model.fit(X_train, y_train)

joblib.dump(model, PREPROCESSING_DIR + 'lr_model.joblib')

try:
    current_run_id = mlflow.active_run().info.run_id
except:
    current_run_id = "UNKNOWN - Autologging might have failed"

print(f"Model Training Selesai. Run ID: {mlflow.active_run().info.run_id}")