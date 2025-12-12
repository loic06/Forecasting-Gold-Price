# TODO: Import your package, replace this by explicit imports of what you need
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from forecasting_gold_price.registry import load_model, TrendEngineeringTransformer, CustomFeatureSelector
import sys

app = FastAPI()

# Chargement du modèle au démarrage
try:
    sys.modules['__main__'].TrendEngineeringTransformer = TrendEngineeringTransformer
    sys.modules['__main__'].CustomFeatureSelector = CustomFeatureSelector
    model = load_model()
    print("✅ Modèle chargé en mémoire depuis le fichier local")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    model = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://127.0.0.1:8000/
@app.get("/")
def root():
    return {
        'message': "Hi ça marche"
    }

# Endpoint for 'https://127.0.0.1:8000/predict?close=2650.17&open=2650.8&high=2652.4&low=2649.7&Basis=2651.1&Upper=2670.2&Lower=2632&Up_Trend=NaN&Down_Trend=2668&KAMA=2653.29&RSI=46.5&Bollinger_Bands_b=0.32&Bollinger_BandWidth=1.04&Highest_Expansion=1.58&FEDFUNDS=4.83'

@app.get("/predict")
def predict(
    close: float,
    open_: float,
    high: float,
    low: float,
    Basis: float,
    Upper: float,
    Lower: float,
    Up_Trend: float,
    Down_Trend: float,
    KAMA: float,
    RSI: float,
    Bollinger_Bands_b: float,
    Bollinger_BandWidth: float,
    Highest_Expansion: float,
    FEDFUNDS: float
):

    # Construction du DataFrame avec les noms de features attendus
    X_pred = pd.DataFrame({
        'close': [close],
        'open': [open_],
        'high': [high],
        'low': [low],
        'Basis': [Basis],
        'Upper': [Upper],
        'Lower': [Lower],
        'Up Trend': [Up_Trend],
        'Down Trend': [Down_Trend],
        'KAMA': [KAMA],
        'RSI': [RSI],
        'Bollinger Bands %b': [Bollinger_Bands_b],
        'Bollinger BandWidth': [Bollinger_BandWidth],
        'Highest Expansion': [Highest_Expansion],
        'FEDFUNDS': [FEDFUNDS],
    })

    # Prédiction
    pred = model.predict(X_pred)

    # Conversion en type Python natif pour la sérialisation JSON
    prediction_value = float(pred[0]) if hasattr(pred, '__iter__') else float(pred)

    return {
        'prediction': prediction_value,
        'status': 'success'
    }
