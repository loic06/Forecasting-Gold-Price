from forecasting_gold_price.params import *
from datetime import datetime
import joblib
import pickle
import glob
import os

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
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, 'models')
        local_model_paths = glob.glob(f"{local_model_directory}/*.pkl")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        #load
        with open(most_recent_model_path_on_disk, 'rb') as f:
            latest_model = pickle.load(f)

        print("✅ Model loaded from local disk")

        return latest_model
