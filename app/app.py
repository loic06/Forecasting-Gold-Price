import streamlit as st
import plotly.graph_objects as go
from datetime import date,timedelta
import requests

st.set_page_config(
    page_title="Gold Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialise l'état de démarrage pour afficher au chargement de la page la valeur de close d'aujourd'hui

if "mode" not in st.session_state:
    st.session_state.mode = "tomorrow"

st.title("🏆 Gold Price Forecasting — Neo‑Quant Dashboard")

st.subheader("🔮 Choisissez votre mode de prédiction")

colA, colB = st.columns(2)

with colA:
    if st.button("📈 Prédiction du prix de cloture de la journée"):
        st.session_state.mode = "tomorrow"
        st.toast("✅ Nouvelle prédiction générée")



with colB:
    if st.button("📅 Prédiction d'une date historique"):
        st.session_state.mode = "history"
        st.toast("✅ Nouvelle prédiction générée")

# Logique de sélection
selected_date = None


if st.session_state.mode == "tomorrow":
    selected_date = date.today()
    st.write("test 1")

elif st.session_state.mode == "history":
    selected_date = st.date_input("Sélectionnez une date passée :", max_value=date.today())
    st.toast("✅ Nouvelle prédiction générée")
    st.write("test 2")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<div class='card'><h3>Prévision du prix de cloture de l'or au {selected_date}</h3><h1>2351.2 $</h1></div>",
    unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'><h3>Confiance</h3><h1>87%</h1></div>", unsafe_allow_html=True)

# --- Call our API ---

# URL definition

# Bien vérifier que les noms des decorateurs sont les mêmes que dans fast.py
url_predict = 'https://ton-url-Seb/predict'
url_get_gold = 'https://ton-url-Seb/get_gold'

# Parameters dictionary for our API :

# bien verifier que les noms des clés du dictionnaire matchent avec les noms des parametres de notre décorateur /predict dans fast.py
params_predict = {
    "start_date": (selected_date-timedelta(days=1)).strftime('%Y-%m-%d'),
    "end_date" : selected_date.strftime('%Y-%m-%d')
    }

# param days à changer à notre guise si on veut afficher plus de profondeur de valeurs réelles past
params_get_gold = {
    "start_date": (selected_date-timedelta(days=5)).strftime('%Y-%m-%d'),
    "end_date" : selected_date.strftime('%Y-%m-%d')
    }

#3. Let's call our API using the `requests` package...
response_predict = requests.get(url_predict, params=params_predict)
reponse_get_gold = requests.get(url_get_gold,params_get_gold)

# --- SECTION : Graphique --

if (response_predict.status_code & reponse_get_gold.status_code) == 200:
# --- Génération de dates ---

    # Renseigner ici toutes les dates réelles provenant de notre response_get_gold
    real_dates =
    predicted_date = selected_date.strftime('%Y-%m-%d')        # J+1

    # --- Valeurs réelles ---
    # Renseigner ici tous les prix de cloture réelles provenant de notre response_get_gold
    real_values = reponse_get_gold.json()["Nom de la clé attribuée"]

    # --- Valeur prédite ---
    predicted_value = response_predict.json()["Nom de la clé attribuée"]

    # ✅ Détermination de la couleur selon la direction
    if predicted_value > real_values[-1]:
        pred_color = "rgba(0, 255, 0, 0.6)"   # vert estompé
    else:
        pred_color = "rgba(255, 0, 0, 0.6)"   # rouge estompé

    fig = go.Figure()

    # --- Courbe réelle (bleu) ---
    fig.add_trace(go.Scatter(
        x=real_dates,
        y=real_values,
        mode='lines',
        name='Réel',
        line=dict(color='blue', width=3)
    ))

    # --- Segment pointillé dynamique ---
    fig.add_trace(go.Scatter(
        x=[real_dates[-1], predicted_date],
        y=[real_values[-1], predicted_value],
        mode='lines+markers',
        name='Prévision',
        line=dict(color=pred_color, width=3, dash='dash'),
        marker=dict(size=10, color=pred_color)
    ))

    fig.update_layout(
        template='plotly_dark',
        showlegend=True,
        xaxis_title="Date (jour)",
        yaxis_title="Prix de cloture (dollar)"
    )

    st.plotly_chart(fig, use_container_width=True)

else : st.error("API call failed. Please check your parameters or API URL.")

# Affichage du disclaimer

st.markdown(
    """
    <div style="
        margin-top: 40px;
        padding: 18px;
        border-radius: 10px;
        background: #1f2937;
        color: #e5e7eb;
        border: 1px solid rgba(255,255,255,0.12);
        font-size: 0.9rem;
    ">
        <strong style="color:#fbbf24;">⚠️ Avertissement :</strong><br>
        Cette application est fournie uniquement à des fins d’entraînement et ne constitue pas un conseil en investissement.
        Les performances passées ne préjugent pas des performances futures. La valeur des investissements peut fluctuer et
        vous pouvez perdre tout ou partie du capital investi. Avant toute décision, évaluez soigneusement les risques et
        consultez un professionnel qualifié.
    </div>
    """,
    unsafe_allow_html=True
)
