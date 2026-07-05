"""
app.py
API FastAPI para predicción de diabetes + frontend web integrado.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np

# ── Cargar modelo y scaler ─────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

THRESHOLD = 0.20  # umbral optimizado (recall >= 0.80)

app = FastAPI(
    title="API Predicción de Diabetes",
    description="Modelo Random Forest entrenado con el dataset Pima Indians Diabetes.",
    version="1.0.0"
)

# ── Esquema de entrada ─────────────────────────────────────────────────────
class PacienteInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }]
        }
    }

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    """Interfaz web para predicción de diabetes."""
    html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicción de Diabetes</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #f0f4f8;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 16px;
        }

        header {
            text-align: center;
            margin-bottom: 32px;
        }

        header h1 {
            font-size: 2rem;
            color: #1a3c5e;
            font-weight: 700;
        }

        header p {
            color: #5a7a99;
            margin-top: 8px;
            font-size: 0.95rem;
        }

        .card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            padding: 36px;
            width: 100%;
            max-width: 680px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #2d4a6e;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .hint {
            font-size: 0.75rem;
            color: #8aa0bb;
            font-weight: 400;
            text-transform: none;
            letter-spacing: 0;
        }

        input[type="number"] {
            border: 1.5px solid #d0dce8;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 0.95rem;
            color: #1a3c5e;
            transition: border-color 0.2s;
            outline: none;
        }

        input[type="number"]:focus {
            border-color: #2563eb;
        }

        button {
            margin-top: 28px;
            width: 100%;
            padding: 14px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }

        button:hover { background: #1d4ed8; }
        button:disabled { background: #93b4f5; cursor: not-allowed; }

        #resultado {
            margin-top: 28px;
            padding: 20px 24px;
            border-radius: 12px;
            display: none;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; } }

        .positivo {
            background: #fff1f2;
            border: 2px solid #fca5a5;
            color: #991b1b;
        }

        .negativo {
            background: #f0fdf4;
            border: 2px solid #86efac;
            color: #166534;
        }

        #resultado h2 { font-size: 1.2rem; margin-bottom: 8px; }
        #resultado p  { font-size: 0.9rem; opacity: 0.85; }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 99px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-top: 10px;
        }

        .badge-alto { background: #fca5a5; color: #7f1d1d; }
        .badge-bajo { background: #86efac; color: #14532d; }

        .prob-bar-container {
            margin-top: 12px;
            background: #e5e7eb;
            border-radius: 99px;
            height: 8px;
            overflow: hidden;
        }

        .prob-bar {
            height: 100%;
            border-radius: 99px;
            transition: width 0.6s ease;
        }

        .links {
            margin-top: 16px;
            text-align: center;
            font-size: 0.82rem;
            color: #8aa0bb;
        }

        .links a { color: #2563eb; text-decoration: none; }
        .links a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <header>
        <h1>🏥 Predicción de Diabetes</h1>
        <p>Modelo Random Forest · Dataset Pima Indians · Umbral optimizado 0.20</p>
    </header>

    <div class="card">
        <div class="grid">
            <div class="field">
                <label>Pregnancies <span class="hint">embarazos</span></label>
                <input type="number" id="Pregnancies" value="6" min="0" max="20" step="1">
            </div>
            <div class="field">
                <label>Glucose <span class="hint">mg/dL</span></label>
                <input type="number" id="Glucose" value="148" min="0" max="300" step="1">
            </div>
            <div class="field">
                <label>BloodPressure <span class="hint">mm Hg</span></label>
                <input type="number" id="BloodPressure" value="72" min="0" max="200" step="1">
            </div>
            <div class="field">
                <label>SkinThickness <span class="hint">mm</span></label>
                <input type="number" id="SkinThickness" value="35" min="0" max="100" step="1">
            </div>
            <div class="field">
                <label>Insulin <span class="hint">mu U/ml</span></label>
                <input type="number" id="Insulin" value="0" min="0" max="900" step="1">
            </div>
            <div class="field">
                <label>BMI <span class="hint">kg/m²</span></label>
                <input type="number" id="BMI" value="33.6" min="0" max="70" step="0.1">
            </div>
            <div class="field">
                <label>Pedigree <span class="hint">función pedigree</span></label>
                <input type="number" id="DiabetesPedigreeFunction" value="0.627" min="0" max="3" step="0.001">
            </div>
            <div class="field">
                <label>Age <span class="hint">años</span></label>
                <input type="number" id="Age" value="50" min="1" max="120" step="1">
            </div>
        </div>

        <button id="btn" onclick="predecir()">Analizar paciente</button>

        <div id="resultado">
            <h2 id="res-titulo"></h2>
            <p id="res-texto"></p>
            <div class="prob-bar-container">
                <div class="prob-bar" id="prob-bar"></div>
            </div>
            <p style="margin-top:6px;font-size:0.8rem;opacity:0.7">
                Probabilidad estimada: <strong id="prob-valor"></strong>
            </p>
            <span class="badge" id="res-badge"></span>
        </div>

        <div class="links">
            <a href="/docs" target="_blank">📄 Documentación API (Swagger)</a> ·
        </div>
    </div>

    <script>
        async function predecir() {
            const btn = document.getElementById('btn');
            btn.disabled = true;
            btn.textContent = 'Analizando...';

            const campos = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                            'Insulin','BMI','DiabetesPedigreeFunction','Age'];
            const datos = {};
            for (const c of campos) datos[c] = parseFloat(document.getElementById(c).value);

            try {
                const resp = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(datos)
                });
                const json = await resp.json();

                const div    = document.getElementById('resultado');
                const titulo = document.getElementById('res-titulo');
                const texto  = document.getElementById('res-texto');
                const badge  = document.getElementById('res-badge');
                const bar    = document.getElementById('prob-bar');
                const probV  = document.getElementById('prob-valor');

                const prob = (json.probabilidad * 100).toFixed(1);
                probV.textContent = prob + '%';
                bar.style.width = prob + '%';

                div.style.display = 'block';

                if (json.prediccion === 1) {
                    div.className = 'positivo';
                    titulo.textContent = '⚠️ Riesgo Alto de Diabetes';
                    texto.textContent  = 'El modelo detecta indicadores compatibles con diabetes tipo 2. Se recomienda evaluación clínica especializada.';
                    badge.textContent  = 'ALTO RIESGO';
                    badge.className    = 'badge badge-alto';
                    bar.style.background = '#ef4444';
                } else {
                    div.className = 'negativo';
                    titulo.textContent = '✅ Bajo Riesgo de Diabetes';
                    texto.textContent  = 'El modelo no detecta indicadores significativos de diabetes tipo 2. Se recomienda seguimiento preventivo rutinario.';
                    badge.textContent  = 'BAJO RIESGO';
                    badge.className    = 'badge badge-bajo';
                    bar.style.background = '#22c55e';
                }
            } catch(e) {
                alert('Error al conectar con la API: ' + e.message);
            }

            btn.disabled = false;
            btn.textContent = 'Analizar paciente';
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/health")
def health():
    """Endpoint de salud del servicio."""
    return {"status": "ok", "modelo": "RandomForest", "umbral": THRESHOLD}


@app.post("/predict")
def predict(paciente: PacienteInput):
    """
    Recibe datos clínicos de una paciente y retorna la predicción de diabetes.
    - **prediccion**: 1 = diabetes, 0 = sin diabetes
    - **probabilidad**: probabilidad estimada de diabetes (0-1)
    - **riesgo**: nivel de riesgo (Alto / Bajo)
    """
    datos = np.array([[
        paciente.Pregnancies,
        paciente.Glucose,
        paciente.BloodPressure,
        paciente.SkinThickness,
        paciente.Insulin,
        paciente.BMI,
        paciente.DiabetesPedigreeFunction,
        paciente.Age
    ]])

    datos_scaled = scaler.transform(datos)
    probabilidad = model.predict_proba(datos_scaled)[0][1]
    prediccion   = int(probabilidad >= THRESHOLD)

    return {
        "prediccion":   prediccion,
        "probabilidad": round(float(probabilidad), 4),
        "riesgo":       "Alto" if prediccion == 1 else "Bajo",
        "umbral":       THRESHOLD
    }
