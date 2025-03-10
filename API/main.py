from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn

# Instancia de la aplicación FastAPI
app = FastAPI(
    title="API de Predicción - GenZ",
    description="API para predecir la carrera aspiracional de la Generación Z usando un modelo Random Forest entrenado.",
    version="1.0.0"
)

# Esquema de datos de entrada (Pydantic)
class PredictionInput(BaseModel):
    data: dict

# Carga de los archivos .pkl
try:
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    le_target = joblib.load("le_target.pkl")
except Exception as e:
    raise RuntimeError(f"Error al cargar archivos preentrenados: {e}")

@app.post("/predict")
def predict(input: PredictionInput):
    # 1) Convertir el diccionario a DataFrame
    input_data = input.data
    input_df = pd.DataFrame([input_data])

    # 2) Asegurarnos de que las columnas estén en el orden esperado
    try:
        input_df = input_df[feature_columns]
    except KeyError as e:
        raise HTTPException(status_code=400, detail="Faltan columnas en el input. Revisa las feature_columns.")

    # 3) Imputar NaN si existen
    input_df = input_df.fillna(0)

    # 4) Escalar los datos
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error en el escalado de datos.")

    # 5) Predecir
    try:
        prediction_numeric = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error en la predicción.")

    # 6) Obtener la confianza (probabilidad de la clase elegida)
    confidence = float(np.max(prediction_proba, axis=1)[0])

    # 7) Convertir la predicción numérica a la etiqueta original
    try:
        pred_label = le_target.inverse_transform(prediction_numeric)[0]
    except:
        pred_label = str(prediction_numeric[0])

    return {
        "prediction": pred_label,
        "confidence": confidence
    }

# Punto de entrada si quieres ejecutarlo con: python main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
