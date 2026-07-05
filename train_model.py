"""
train_model.py
Entrena el modelo Random Forest con el dataset Pima Indians Diabetes
y guarda el modelo y el scaler en archivos .pkl listos para la API.
"""

import pickle
import urllib.request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── 1. Descargar y cargar dataset ──────────────────────────────────────────
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

print("Descargando dataset...")
urllib.request.urlretrieve(URL, "diabetes.csv")
df = pd.read_csv("diabetes.csv", header=None, names=COLS)
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# ── 2. Preprocesamiento ────────────────────────────────────────────────────
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, df[col][df[col] != 0].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# ── 3. Escalado y división ─────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ── 4. Entrenar modelo ─────────────────────────────────────────────────────
print("Entrenando Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ── 5. Evaluar brevemente ──────────────────────────────────────────────────
from sklearn.metrics import recall_score, roc_auc_score
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.20).astype(int)
recall = recall_score(y_test, y_pred)
auc    = roc_auc_score(y_test, proba)
print(f"Recall  (umbral 0.20): {recall:.4f}")
print(f"ROC-AUC              : {auc:.4f}")

# ── 6. Guardar modelo y scaler ─────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Modelo guardado en model.pkl")
print("Scaler guardado en scaler.pkl")
