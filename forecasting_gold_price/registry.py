from forecasting_gold_price.params import *
from datetime import datetime
import joblib
import pickle

def save_model(model_fit):
    # Save model locally

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    champion_path = f"models/{ts}_champion_v1.pkl"
    joblib.dump(model_fit, champion_path)

    print("✅ Model saved locally")

def load_model() :
    """
    Return a saved model:
    """
    if MODEL_TARGET == "local":
        #load
        with open('models/20251216_153712_champion_v1.pkl', 'rb') as f:
            latest_model = joblib.load(f)

        print("✅ Model loaded from local disk")

        return latest_model
