from forecasting_gold_price.data import download_and_concat_tickers , clean_name
from forecasting_gold_price.preprocessor import build_preprocessing_pipeline
from forecasting_gold_price.model import initialize_model
from forecasting_gold_price.registry import save_model

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

#--------------------------------
# --- Récupération de la data ---
#--------------------------------
tickers = [
    "^GSPC", "^DJI", "^VIX", "^GVZ", "^OVX", "^MOVE", "BOND", "^STOXX",
    "EURUSD=X", "DX-Y.NYB", "CL=F", "BZ=F", "SI=F", "PL=F", "BTC-USD", "JPM",
    "PA=F", "^TNX", "GC=F", "GDX", "EGO", "USO", "GD=F"]

result_df = download_and_concat_tickers(tickers, start_date="2000-08-30", end_date="2025-12-11")



#----------------------
# --- Preprocessing ---
#----------------------
# Drop NaN based on gold price
result_df = result_df.dropna(how='any', subset=["GC=F_Close"], axis=0)

# Replace special characters in columns name
result_df.columns = [clean_name(c) for c in result_df.columns]

# --- Settings ---
horizon_day=1
train_ratio=0.7
num_corr_threshold=0.95
plot_learning_curves=True
strategy="custom" # median, mean

target_source_col = 'GC_F_Close'

# --- Features and target ---
df = result_df.sort_index().copy()
target_col = f"{target_source_col}_t+{horizon_day}"
df[target_col] = df[target_source_col].shift(-horizon_day)

# Drop values of the shift
df = df.dropna(subset=[target_col])

# Set close column as first column
cols = list(df.columns)
cols.remove(target_source_col)
cols.insert(0, target_source_col)
df = df[cols]

feature_cols = df.drop(columns=target_col).columns
X = df[feature_cols]
y = df[target_col]

pipe_auto = build_preprocessing_pipeline(remove_features=True, strategy=strategy, exclude_for_zero_drop=['GC_F_Close']) # mean, median


#---------------------------
# --- Création du modèle ---
#---------------------------

model = initialize_model()

#-------------------------------
# --- Entrainement du modèle ---
#-------------------------------
pipe = Pipeline(steps=[
    ('preprocessing', pipe_auto),
    ('sfm', SelectFromModel(model)),
    ('model', model)
])

model_fit = pipe.fit(X, y)


#-----------------------------
# --- sauvegarde du modèle ---
#-----------------------------

save_model(model_fit)
print("✅ Le modèle est sauvegardé")
