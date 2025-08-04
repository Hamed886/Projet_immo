import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(layout="wide")
st.title("💰 Simulateur de Prix Immobilier au m²")

# Chargement des modèles optimisés
model_appart = joblib.load("models/et_appart.pkl")
model_maison = joblib.load("models/xgb_maison.pkl")

# Chargement des jeux encodés de référence
X_appart_encoded = pd.read_csv("data/annonces_ventes_68_appartements_X_test.csv", sep=";", encoding="ISO-8859-1")
X_maison_encoded = pd.read_csv("data/annonces_ventes_68_maisons_X_test.csv", sep=";", encoding="ISO-8859-1")
X_appart_raw = pd.read_csv("data/X_test_appart_raw.csv", sep=";", encoding="ISO-8859-1")
X_maison_raw = pd.read_csv("data/X_test_maison_raw.csv", sep=";", encoding="ISO-8859-1")

# Paramètres de simulation
typedebien = st.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
if typedebien == "Appartement":
    model = model_appart
    X_encoded = X_appart_encoded.copy()
    X_raw = X_appart_raw.copy()
    MAE = 351.77
else:
    model = model_maison
    X_encoded = X_maison_encoded.copy()
    X_raw = X_maison_raw.copy()
    MAE = 397.36

st.markdown("---")

mode_simulation = st.radio("Mode de simulation", ["🗂️ Choisir un bien existant", "🛠️ Entrer mes propres caractéristiques"])
X_input = pd.DataFrame(columns=X_encoded.columns)
X_input.loc[0] = 0

if mode_simulation == "🗂️ Choisir un bien existant":
    idx = st.selectbox("🔍 Sélectionne un bien existant", list(range(len(X_encoded))))
    X_input = X_encoded.iloc[[idx]].copy()
    surface = X_raw.iloc[idx].get("surface", 50)
    st.success("✅ Caractéristiques récupérées automatiquement")
else:
    surface = st.slider("Surface (m²)", 10, 300, 75)
    nb_pieces = st.slider("Nombre de pièces", 1, 10, 4)
    nb_toilettes = st.slider("Nombre de toilettes", 0, 5, 1)
    logement_neuf = st.radio("Logement neuf ?", ["Oui", "Non"])
    balcon = st.radio("Balcon ?", ["Oui", "Non"])
    cave = st.radio("Cave ?", ["Oui", "Non"])
    ascenseur = st.radio("Ascenseur ?", ["Oui", "Non"])
    bain = st.radio("Baignoire ?", ["Oui", "Non"])
    eau = st.radio("Salle d'eau ?", ["Oui", "Non"])
    parking = st.radio("Place de parking ?", ["Oui", "Non"])
    exclusivite = st.radio("Annonce exclusive ?", ["Oui", "Non"])
    dpeL = st.selectbox("Classe énergétique DPE", ["A", "B", "C", "D", "E", "F", "G"], index=3)
    exposition = st.selectbox("Exposition", ["Sud", "Est", "Nord", "Autre"])
    chauffage_energie = st.selectbox("Énergie chauffage", ["Electrique", "Gaz", "Fioul", "Bois", "Autre"])
    chauffage_systeme = st.selectbox("Système chauffage", ["Individuel", "Collectif", "Autre"])
    chauffage_mode = st.selectbox("Mode chauffage", ["Radiateur", "Plancher chauffant", "Autre"])

    dpe_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}  # Mappage cohérent avec le notebook
    X_input.loc[0, "surface"] = surface
    X_input.loc[0, "nb_pieces"] = nb_pieces
    X_input.loc[0, "nb_toilettes"] = nb_toilettes
    X_input.loc[0, "logement_neuf"] = 1 if logement_neuf == "Oui" else 0
    X_input.loc[0, "balcon"] = 1 if balcon == "Oui" else 0
    X_input.loc[0, "cave"] = 1 if cave == "Oui" else 0
    X_input.loc[0, "ascenseur"] = 1 if ascenseur == "Oui" else 0
    X_input.loc[0, "bain"] = 1 if bain == "Oui" else 0
    X_input.loc[0, "eau"] = 1 if eau == "Oui" else 0
    X_input.loc[0, "places_parking"] = 1 if parking == "Oui" else 0
    X_input.loc[0, "annonce_exclusive"] = 1 if exclusivite == "Oui" else 0
    X_input.loc[0, "dpeL"] = dpe_mapping[dpeL]
    for dir in ["sud", "est", "nord", "autre"]:
        X_input.loc[0, f"exposition_{dir}"] = 1 if exposition.lower() == dir else 0
    X_input.loc[0, "chauffage_energie"] = {"Electrique": 0, "Gaz": 1, "Fioul": 2, "Bois": 3, "Autre": 4}[chauffage_energie]
    X_input.loc[0, "chauffage_systeme"] = {"Individuel": 0, "Collectif": 1, "Autre": 2}[chauffage_systeme]
    X_input.loc[0, "chauffage_mode"] = {"Radiateur": 0, "Plancher chauffant": 1, "Autre": 2}[chauffage_mode]

# Nettoyage et matching des colonnes attendues par le modèle
X_input.drop(columns=[col for col in ["date", "typedebien_lite"] if col in X_input.columns], inplace=True, errors="ignore")
X_input_final = X_input.loc[:, model.feature_names_in_].copy()
X_input_final = X_input_final.apply(pd.to_numeric, errors='coerce')

if X_input_final.isnull().values.any():
    st.error("⛔ Certaines variables sont mal saisies ou manquantes. Vérifie les sélections.")
    st.stop()

# Prédiction
prediction = model.predict(X_input_final)[0]
st.markdown("---")
st.markdown(f"### 🌟 Prix estimé : **{prediction:.2f} €/m²**")
st.markdown(f"📊 Surface réelle : **{surface} m²**")
st.markdown(f"💶 Prix total estimé : **{prediction * surface:.0f} €**")
st.markdown(f"📉 Intervalle de confiance : **[{prediction - MAE:.2f} ; {prediction + MAE:.2f}] €/m²**")

if st.button("📊 Interprétation SHAP du modèle"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input_final)
    shap_input = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_input_final.iloc[0],
        feature_names=X_input_final.columns
    )
    shap_df = pd.DataFrame({
        "Feature": shap_input.feature_names,
        "Contribution": shap_input.values,
        "Value": shap_input.data
    })
    shap_df = shap_df.reindex(shap_df.Contribution.abs().sort_values(ascending=False).index)
    top_features = shap_df.head(10)

    colors = top_features["Contribution"].apply(lambda x: "crimson" if x > 0 else "royalblue")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_features["Contribution"],
        y=top_features["Feature"],
        orientation="h",
        text=[f"{contrib:+.2f} €/m²" for contrib in top_features["Contribution"]],
        hovertext=[f"Valeur réelle : {val:.2f}" for val in top_features["Value"]],
        marker_color=colors
    ))
    fig.update_layout(title="🔍 Explication SHAP dynamique", yaxis=dict(autorange="reversed"), height=500, width=1000)
    st.plotly_chart(fig)

    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value.item()
    st.info(f"Base value SHAP (moyenne modèle) : `{base_value:.2f} €/m²`")

    st.markdown("""
    ### 🧐 Lecture simplifiée
    - **Base value** : moyenne du modèle (prix de référence)
    - Les barres rouges : variables qui font **augmenter** le prix
    - Les barres bleues : variables qui font **baisser** le prix
    - Plus la barre est longue → plus l'impact est fort
    """)

    st.markdown("#### 💬 Interprétation automatique (top 5 variables)")
    for i, row in top_features.head(5).iterrows():
        direction = "augmente" if row["Contribution"] > 0 else "fait baisser"
        st.markdown(f"- La variable **'{row['Feature']}'** (valeur : {row['Value']:.2f}) **{direction}** le prix estimé de **{abs(row['Contribution']):.2f} €/m²**.")

st.markdown("---")
st.caption("Projet Compagnon Immobilier – Simulation dynamique des prix au m² ✨")

