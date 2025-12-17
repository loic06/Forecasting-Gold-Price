# TODO: Import your package, replace this by explicit imports of what you need
import pandas as pd
from datetime import timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from forecasting_gold_price.registry import load_model
from forecasting_gold_price.data import download_and_concat_tickers
from forecasting_gold_price.data import clean_name


app = FastAPI()

# Chargement du modèle au démarrage

try:
    app.model = load_model()
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

# Endpoint for 'https://127.0.0.1:8000/predict?'

@app.get("/predict")
def predict(end_date:str):

    tickers = [
    "^GSPC", "^DJI", "^VIX", "^GVZ", "^OVX", "^MOVE", "BOND", "^STOXX",
    "EURUSD=X", "DX-Y.NYB", "CL=F", "BZ=F", "SI=F", "PL=F", "BTC-USD", "JPM",
    "PA=F", "^TNX", "GC=F", "GDX", "EGO", "USO", "GD=F",
]

    new_end = pd.to_datetime(end_date)
    new_start = new_end-timedelta(1)
    new_start = new_start.strftime('%Y-%m-%d')

    # Construction du DataFrame avec les noms de features attendus
    X_pred = download_and_concat_tickers(tickers, new_start, end_date)
    # Replace special characters in columns name
    X_pred.columns = [clean_name(c) for c in X_pred.columns]

    # Prédiction
    pred = app.model.predict(X_pred)


    # Conversion en type Python natif pour la sérialisation JSON
    prediction_value = float(pred[0]) if hasattr(pred, '__iter__') else float(pred)

    return {
        'prediction': prediction_value,
        'status': 'success'
    }

@app.get("/get_gold")
def get_gold(end_date:str):

    new_end = pd.to_datetime(end_date)-timedelta(1)
    new_start = new_end-timedelta(5)
    new_start = new_start.strftime('%Y-%m-%d')

    ticker = ["GC=F"]

    # Construction du DataFrame avec les noms de features attendus
    gold_value = download_and_concat_tickers(ticker, new_start, new_end)
    # Replace special characters in columns name
    gold_value.columns = [clean_name(c) for c in gold_value.columns]

    return {
        'value': gold_value["GC_F_Close"],
        'status': 'success'
    }
