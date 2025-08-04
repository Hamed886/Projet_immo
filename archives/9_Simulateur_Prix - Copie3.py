import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from prophet import Prophet


def formuler_phrase_variable(feature, value, impact):
    direction = "positif" if impact > 0 else "négatif"
    texte = ""

    if feature == "surface":
        texte = f"Avec une **surface habitable de {value:.0f} m²**, le bien bénéficie d’un atout majeur qui **valorise fortement son prix au m²**." if impact > 0 else f"La **surface limitée de {value:.0f} m²** constitue un frein potentiel qui **réduit la valeur estimée du bien**."
    elif "nb_pieces" in feature:
        texte = f"Le bien dispose de **{value:.0f} pièces**, ce qui constitue un **critère favorable** dans la valorisation." if impact > 0 else f"Le nombre de pièces (**{value:.0f}**) est relativement limité, ce qui a un **impact modéré sur la baisse du prix**."
    elif "toilettes" in feature:
        texte = f"La présence de **{value:.0f} toilettes** est un **critère de confort**, qui **favorise le positionnement prix**." if impact > 0 else f"L’absence de commodités suffisantes (**{value:.0f} toilette(s)**) limite la valorisation du bien."
    elif "score_transport" in feature:
        texte = f"Une bonne accessibilité aux transports (**score {value:.2f}**) est un **facteur clé d’attractivité**." if impact > 0 else f"Le **score transport modéré ({value:.2f})** pénalise partiellement la valorisation."
    elif "annee" in feature:
        texte = f"L’année de construction (**{value:.0f}**) récente suggère un bon état général, **renforçant la valeur estimée**." if impact > 0 else f"Le bien date de **{value:.0f}**, ce qui peut **générer des coûts d’entretien** et freiner l’intérêt."
    elif "quartier" in feature:
        texte = f"Le **quartier ({value})** est considéré comme attractif, ce qui contribue **positivement à la valorisation**." if impact > 0 else f"Le **quartier ({value})** a un **impact défavorable** sur le positionnement prix."
    elif "commune" in feature:
        texte = f"La commune (**{value}**) est recherchée, ce qui **soutient un prix élevé au m²**." if impact > 0 else f"La commune (**{value}**) est moins prisée, ce qui **pèse sur la valeur estimée**."

    if texte == "":
        direction = "positivement" if impact > 0 else "négativement"
        texte = f"La variable **{feature}**, avec une valeur de **{value:.2f}**, a contribué **{direction} au prix estimé**."

    return "🔎 " + texte







st.set_page_config(layout="wide")
st.title("💰 Simulateur de Prix Immobilier au m²")

# Chargement des modèles
model_appart = joblib.load("models/et_appart.pkl")
model_maison = joblib.load("models/xgb_maison.pkl")

# Chargement des datasets
X_appart_encoded = pd.read_csv("data/annonces_ventes_68_appartements_X_test.csv", sep=";", encoding="ISO-8859-1")
X_maison_encoded = pd.read_csv("data/annonces_ventes_68_maisons_X_test.csv", sep=";", encoding="ISO-8859-1")
X_appart_raw = pd.read_csv("data/X_test_appart_raw.csv", sep=";", encoding="ISO-8859-1")
X_maison_raw = pd.read_csv("data/X_test_maison_raw.csv", sep=";", encoding="ISO-8859-1")
y_appart = pd.read_csv("data/y_a_test.csv", sep=";", encoding="ISO-8859-1")
y_maison = pd.read_csv("data/y_m_test.csv", sep=";", encoding="ISO-8859-1")

typedebien = st.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
if typedebien == "Appartement":
    model = model_appart
    X_encoded = X_appart_encoded.copy()
    X_raw = X_appart_raw.copy()
    y_target = y_appart.copy()
    MAE = 351.77
else:
    model = model_maison
    X_encoded = X_maison_encoded.copy()
    X_raw = X_maison_raw.copy()
    y_target = y_maison.copy()
    MAE = 397.36

st.markdown("---")

mode_simulation = st.radio("Mode de simulation", ["🗂️ Choisir un bien existant", "🛠️ Entrer mes propres caractéristiques"])
X_input = pd.DataFrame(columns=X_encoded.columns)
X_input.loc[0] = 0

if mode_simulation == "🗂️ Choisir un bien existant":
    idx = st.selectbox("🔍 Sélectionne un bien existant", list(range(len(X_encoded))))
    X_input = X_encoded.iloc[[idx]].copy()
    surface = float(X_raw.iloc[idx]["surface"])
    st.success("✅ Caractéristiques récupérées automatiquement")
else:
    vars_to_edit = st.multiselect("🛠️ Choisis les variables à remplir manuellement :", [
        "surface", "nb_pieces", "nb_toilettes", "logement_neuf", "balcon", "cave", "ascenseur",
        "bain", "eau", "places_parking", "annonce_exclusive", "dpeL", "exposition",
        "chauffage_energie", "chauffage_systeme", "chauffage_mode"
    ], default=["surface", "nb_pieces", "dpeL"])

    if "surface" in vars_to_edit:
        surface = st.slider("Surface (m²)", 10, 300, 75)
        X_input.loc[0, "surface"] = surface
    if "nb_pieces" in vars_to_edit:
        X_input.loc[0, "nb_pieces"] = st.slider("Nombre de pièces", 1, 10, 4)
    if "nb_toilettes" in vars_to_edit:
        X_input.loc[0, "nb_toilettes"] = st.slider("Nombre de toilettes", 0, 5, 1)
    if "logement_neuf" in vars_to_edit:
        X_input.loc[0, "logement_neuf"] = 1 if st.radio("Logement neuf ?", ["Oui", "Non"]) == "Oui" else 0
    if "balcon" in vars_to_edit:
        X_input.loc[0, "balcon"] = 1 if st.radio("Balcon ?", ["Oui", "Non"]) == "Oui" else 0
    if "cave" in vars_to_edit:
        X_input.loc[0, "cave"] = 1 if st.radio("Cave ?", ["Oui", "Non"]) == "Oui" else 0
    if "ascenseur" in vars_to_edit:
        X_input.loc[0, "ascenseur"] = 1 if st.radio("Ascenseur ?", ["Oui", "Non"]) == "Oui" else 0
    if "bain" in vars_to_edit:
        X_input.loc[0, "bain"] = 1 if st.radio("Baignoire ?", ["Oui", "Non"]) == "Oui" else 0
    if "eau" in vars_to_edit:
        X_input.loc[0, "eau"] = 1 if st.radio("Salle d'eau ?", ["Oui", "Non"]) == "Oui" else 0
    if "places_parking" in vars_to_edit:
        X_input.loc[0, "places_parking"] = 1 if st.radio("Place de parking ?", ["Oui", "Non"]) == "Oui" else 0
    if "annonce_exclusive" in vars_to_edit:
        X_input.loc[0, "annonce_exclusive"] = 1 if st.radio("Annonce exclusive ?", ["Oui", "Non"]) == "Oui" else 0
    if "dpeL" in vars_to_edit:
        dpe_mapping = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
        dpeL = st.selectbox("Classe énergétique DPE", list(dpe_mapping.keys()), index=3)
        X_input.loc[0, "dpeL"] = dpe_mapping[dpeL]
    if "exposition" in vars_to_edit:
        exposition = st.selectbox("Exposition", ["Sud", "Est", "Nord", "Autre"])
        for dir in ["sud", "est", "nord", "autre"]:
            X_input.loc[0, f"exposition_{dir}"] = 1 if exposition.lower() == dir else 0
    if "chauffage_energie" in vars_to_edit:
        X_input.loc[0, "chauffage_energie"] = {
            "Electrique": 0, "Gaz": 1, "Fioul": 2, "Bois": 3, "Autre": 4
        }[st.selectbox("Énergie chauffage", ["Electrique", "Gaz", "Fioul", "Bois", "Autre"])]
    if "chauffage_systeme" in vars_to_edit:
        X_input.loc[0, "chauffage_systeme"] = {
            "Individuel": 0, "Collectif": 1, "Autre": 2
        }[st.selectbox("Système chauffage", ["Individuel", "Collectif", "Autre"])]
    if "chauffage_mode" in vars_to_edit:
        X_input.loc[0, "chauffage_mode"] = {
            "Radiateur": 0, "Plancher chauffant": 1, "Autre": 2
        }[st.selectbox("Mode chauffage", ["Radiateur", "Plancher chauffant", "Autre"])]

# Nettoyage final
X_input.drop(columns=[col for col in ["date", "typedebien_lite"] if col in X_input.columns], inplace=True, errors="ignore")
X_input_final = X_input.loc[:, model.feature_names_in_].copy()
X_input_final = X_input_final.apply(pd.to_numeric, errors='coerce')

if X_input_final.isnull().values.any():
    st.error("⛔ Certaines variables sont mal saisies ou manquantes.")
    st.stop()

# Prédiction
prediction = model.predict(X_input_final)[0]
surface = float(X_raw.iloc[idx]["surface"]) if mode_simulation == "🗂️ Choisir un bien existant" else float(X_input.get("surface", 75))
st.markdown("---")
st.markdown(f"### 🌟 Prix estimé : **{prediction:.2f} €/m²**")
st.markdown(f"📊 Surface réelle : **{surface} m²**")
st.markdown(f"💶 Prix total estimé : **{prediction * surface:.0f} €**")
st.markdown(f"📉 Intervalle de confiance : **[{prediction - MAE:.2f} ; {prediction + MAE:.2f}] €/m²**")

# SHAP
if st.button("📊 Interprétation SHAP"):
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
    shap_df = shap_df.reindex(shap_df["Contribution"].abs().sort_values(ascending=False).index)

    top_features = shap_df.head(10)
    colors = top_features["Contribution"].apply(lambda x: "crimson" if x > 0 else "royalblue")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_features["Contribution"],
        y=top_features["Feature"],
        orientation="h",
        text=[f"{contrib:+.2f} €/m²" for contrib in top_features["Contribution"]],
        hovertext=[f"Valeur : {val:.2f}" for val in top_features["Value"]],
        marker_color=colors
    ))
    fig.update_layout(title="🔍 SHAP – Top 10 contributions", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig)

# Prophet
if st.button("📈 Projection des prix 12 mois"):
    df_raw = X_appart_raw if typedebien == "Appartement" else X_maison_raw
    y_full = y_appart if typedebien == "Appartement" else y_maison
    df_raw = df_raw.copy()
    df_raw["prix_m2_vente"] = y_full["prix_m2_vente"]
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    ts = df_raw.groupby(df_raw["date"].dt.to_period("M"))["prix_m2_vente"].median().reset_index()
    ts.columns = ["ds", "y"]
    ts["ds"] = ts["ds"].dt.to_timestamp()

    model_p = Prophet(yearly_seasonality=True)
    model_p.fit(ts)
    future = model_p.make_future_dataframe(periods=12, freq="MS")
    forecast = model_p.predict(future)
    fig = model_p.plot(forecast)
    st.pyplot(fig)


# Remplacement des valeurs encodées par valeurs brutes pour SHAP
    if mode_simulation == "🗂️ Choisir un bien existant":
        valeurs_visibles = ["surface", "nb_pieces", "nb_toilettes", "annee_construction"]
        for var in valeurs_visibles:
            if var in X_raw.columns and var in shap_df["Feature"].values:
                true_val = float(X_raw.iloc[idx][var])
                shap_df.loc[shap_df["Feature"] == var, "Value"] = true_val
    else:
        for var in X_input.columns:
            if var in shap_df["Feature"].values:
                shap_df.loc[shap_df["Feature"] == var, "Value"] = X_input.loc[0, var]


st.markdown("---")
st.header("📊 Comparaison entre deux biens")

if st.checkbox("🔁 Activer la comparaison avec un second bien"):
    idx2 = st.selectbox("🔎 Sélectionne un second bien à comparer", list(range(len(X_encoded))), key="bien_2")
    surface2 = float(X_raw.iloc[idx2]["surface"])
    X_input2 = X_encoded.iloc[[idx2]].copy()

    X_input2.drop(columns=[col for col in ["date", "typedebien_lite"] if col in X_input2.columns], inplace=True, errors="ignore")
    X_input2_final = X_input2.loc[:, model.feature_names_in_].copy()
    X_input2_final = X_input2_final.apply(pd.to_numeric, errors='coerce')

    if X_input2_final.isnull().values.any():
        st.error("⛔ Certaines variables sont mal saisies ou manquantes dans le bien à comparer.")
    else:
        prediction2 = model.predict(X_input2_final)[0]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏠 Bien sélectionné")
            st.metric("Prix au m²", f"{prediction:.2f} €")
            st.metric("Surface", f"{surface:.0f} m²")
            st.metric("Prix total", f"{prediction * surface:.0f} €")

        with col2:
            st.subheader("🏠 Bien comparé")
            st.metric("Prix au m²", f"{prediction2:.2f} €")
            st.metric("Surface", f"{surface2:.0f} m²")
            st.metric("Prix total", f"{prediction2 * surface2:.0f} €")

        st.markdown("---")
        delta_prix = prediction - prediction2
        direction = "plus cher" if delta_prix > 0 else "moins cher"
        st.info(f"💡 Le bien sélectionné est **{abs(delta_prix):.2f} €/m² {direction}** que le bien comparé.")


        if st.checkbox("📊 Interprétation SHAP du bien comparé"):
            explainer2 = shap.TreeExplainer(model)
            shap_values2 = explainer2.shap_values(X_input2_final)
            shap_input2 = shap.Explanation(
                values=shap_values2[0],
                base_values=explainer2.expected_value,
                data=X_input2_final.iloc[0],
                feature_names=X_input2_final.columns
            )
            shap_df2 = pd.DataFrame({
                "Feature": shap_input2.feature_names,
                "Contribution": shap_input2.values,
                "Value": shap_input2.data
            })
            shap_df2 = shap_df2.reindex(shap_df2["Contribution"].abs().sort_values(ascending=False).index)

            valeurs_visibles = ["surface", "nb_pieces", "nb_toilettes", "annee_construction"]
            for var in valeurs_visibles:
                if var in X_raw.columns and var in shap_df2["Feature"].values:
                    true_val2 = float(X_raw.iloc[idx2][var])
                    shap_df2.loc[shap_df2["Feature"] == var, "Value"] = true_val2

            top_features2 = shap_df2.head(10)
            colors2 = top_features2["Contribution"].apply(lambda x: "crimson" if x > 0 else "royalblue")

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=top_features2["Contribution"],
                y=top_features2["Feature"],
                orientation="h",
                text=[f"{contrib:+.2f} €/m²" for contrib in top_features2["Contribution"]],
                hovertext=[f"Valeur : {val:.2f}" for val in top_features2["Value"]],
                marker_color=colors2
            ))
            fig2.update_layout(title="🔍 SHAP – Bien comparé", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2)


# 🔍 Interprétation automatique – Bien principal
st.markdown("### 🧠 Interprétation automatique du prix estimé")

n_top = 5
if 'shap_df' in locals():
    for i in range(min(n_top, len(shap_df))):
        row = shap_df.iloc[i]
        feat = row["Feature"]
        val = row["Value"]
        impact = row["Contribution"]
        direction = "à augmenter" if impact > 0 else "à diminuer"
        couleur = "🟢" if impact > 0 else "🔴"
        st.markdown(f"{couleur} La variable **{feat}** avec une valeur de **{val:.2f}** a contribué **{direction}** le prix estimé**.")
else:
    st.warning("⚠️ Veuillez d'abord sélectionner un bien pour afficher l’interprétation automatique.")

# 🔁 Interprétation automatique – Bien comparé (si activé)
if 'shap_df2' in locals():
    st.markdown("### 🧠 Interprétation automatique du bien comparé")
    for i in range(min(n_top, len(shap_df2))):
        row = shap_df2.iloc[i]
        feat = row["Feature"]
        val = row["Value"]
        impact = row["Contribution"]
        direction = "à augmenter" if impact > 0 else "à diminuer"
        couleur = "🟢" if impact > 0 else "🔴"
        st.markdown(f"{couleur} La variable **{feat}** avec une valeur de **{val:.2f}** a contribué **{direction}** le prix estimé**.")
else:
    st.info("ℹ️ Veuillez comparer deux biens pour afficher l’analyse automatique du bien comparé.")





# Interprétation narrative – Bien principal
try:
    st.markdown("### 🧠 Analyse narrative du prix estimé")
    for i in range(min(n_top, len(shap_df))):
        row = shap_df.iloc[i]
        st.markdown(formuler_phrase_variable(row["Feature"], row["Value"], row["Contribution"]))
except Exception as e:
    st.info("ℹ️ Veuillez d’abord sélectionner un bien pour afficher l’interprétation.")

# Interprétation narrative – Bien comparé
if 'shap_df2' in locals():
    try:
        st.markdown("### 🧠 Analyse narrative du bien comparé")
        for i in range(min(n_top, len(shap_df2))):
            row = shap_df2.iloc[i]
            st.markdown(formuler_phrase_variable(row["Feature"], row["Value"], row["Contribution"]))
    except Exception as e:
        st.info("ℹ️ Veuillez comparer deux biens pour afficher l’analyse narrative.")





