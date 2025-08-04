# simulateur_prix_v3_2.py
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
    # 🔁 Ajout identifiant unique et label utilisateur
    X_encoded = X_encoded.copy()
    X_raw = X_raw.copy()
    X_encoded["id_bien"] = X_encoded.index
    X_raw["id_bien"] = X_raw.index

    X_raw["label_bien"] = (
        X_raw["commune"].astype(str).fillna("Inconnue") + " | " +
        X_raw["surface"].fillna(0).astype(int).astype(str) + " m² | " +
        X_raw["nb_pieces"].fillna(0).astype(int).astype(str) + " pièces"
    )

    # 📦 Liste des biens avec label
    df_selection = X_encoded[["id_bien"]].merge(X_raw[["id_bien", "label_bien"]], on="id_bien", how="left")

    # 🎯 Sélection utilisateur
    selected_label = st.selectbox("🔍 Sélectionne un bien existant", df_selection["label_bien"].tolist())

    # 🧭 Trouver l'index réel
    selected_id = df_selection[df_selection["label_bien"] == selected_label]["id_bien"].values[0]
    idx = X_encoded[X_encoded["id_bien"] == selected_id].index[0]

    # ✅ Récupération des données cohérentes
    X_input = X_encoded.iloc[[idx]].copy()
    surface = X_raw.iloc[idx].get("surface", 50)

    resume = X_raw.iloc[idx][[col for col in X_raw.columns if col in ["surface", "nb_pieces", "nb_toilettes", "etage", "annee_construction", "balcon", "cave", "ascenseur", "chauffage_energie", "exposition"]]].to_frame().rename(columns={idx: "Valeur"})
    #Afficher que les colonnes remplies
    resume = resume[resume['Valeur'].notna()]
    resume = resume.reset_index().rename(columns={"index": "Caractéristique"})  
    st.success("✅ Caractéristiques récupérées automatiquement")
    with st.expander("📝 Résumé des caractéristiques du bien sélectionné"):
        st.table(resume)
        # Interprétation enrichie avec emojis
        texte = []
        if not pd.isna(X_raw.iloc[idx].get("etage")):
            texte.append(f"🏢 situé au **{int(X_raw.iloc[idx]['etage'])}ᵉ étage**")
        if not pd.isna(X_raw.iloc[idx].get("annee_construction")):
            texte.append(f"📅 construit en **{int(X_raw.iloc[idx]['annee_construction'])}**")
        if X_raw.iloc[idx].get("balcon", 0):
            texte.append("🚪 avec **balcon**")
        if X_raw.iloc[idx].get("ascenseur", 0):
            texte.append("🪜 **ascenseur disponible**")
        if X_raw.iloc[idx].get("cave", 0):
            texte.append("🧱 avec **cave**")
        if not pd.isna(X_raw.iloc[idx].get("chauffage_energie")):
            texte.append(f"🔥 chauffage : **{X_raw.iloc[idx]['chauffage_energie']}**")
        if not pd.isna(X_raw.iloc[idx].get("exposition")):
            texte.append(f"☀️ exposé **{X_raw.iloc[idx]['exposition'].lower()}**")
        if texte:
            st.markdown("📌 " + ", ".join(texte) + ".")
else:
    with st.container():
        st.markdown("### 🏠 Caractéristiques générales")
        surface = st.slider("Surface (m²)", 10, 300, 75)
        nb_pieces = st.slider("Nombre de pièces", 1, 10, 4)
        nb_toilettes = st.slider("Nombre de toilettes", 0, 5, 1)
        annee_construction = st.slider("Année de construction", min_value=1900, max_value=2023, value=2000)

    with st.container():
        st.markdown("### 🛁 Équipements")
        logement_neuf = st.radio("Logement neuf ?", ["Oui", "Non"], horizontal=True)
        balcon = st.radio("Balcon ?", ["Oui", "Non"], horizontal=True)
        cave = st.radio("Cave ?", ["Oui", "Non"], horizontal=True)
        ascenseur = st.radio("Ascenseur ?", ["Oui", "Non"], horizontal=True)
        bain = st.radio("Baignoire ?", ["Oui", "Non"], horizontal=True)
        eau = st.radio("Salle d'eau ?", ["Oui", "Non"], horizontal=True)
        parking = st.radio("Place de parking ?", ["Oui", "Non"], horizontal=True)
        exclusivite = st.radio("Annonce exclusive ?", ["Oui", "Non"], horizontal=True)

    with st.container():
        st.markdown("### 🔥 Chauffage & Énergie")
        dpeL = st.selectbox("Classe énergétique DPE", ["A", "B", "C", "D", "E", "F", "G"], index=3)
        exposition = st.selectbox("Exposition", ["Sud", "Est", "Nord", "Autre"])
        chauffage_energie = st.selectbox("Énergie chauffage", ["Electrique", "Gaz", "Fioul", "Bois", "Autre"])
        chauffage_systeme = st.selectbox("Système chauffage", ["Individuel", "Collectif", "Autre"])
        chauffage_mode = st.selectbox("Mode chauffage", ["Radiateur", "Plancher chauffant", "Autre"])

    dpe_mapping = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
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
    X_input.loc[0, "annee_construction"] = annee_construction


# Nettoyage et matching
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

    st.markdown("#### 💬 Interprétation automatique (top 5 variables)")
    for i, row in top_features.head(5).iterrows():
        direction = "augmente" if row["Contribution"] > 0 else "fait baisser"
        st.markdown(f"- La variable **'{row['Feature']}'** **{direction}** le prix estimé de **{abs(row['Contribution']):.2f} €/m²**.")

st.markdown("---")

# Carte des biens 
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
# Chargement des coordonnées
coord_path = "data/map_appartements_commune_proche_optimise.xlsx" if typedebien == "Appartement" else "data/map_maisons_commune_proche_optimise.xlsx"
coords = pd.read_excel(coord_path).rename(columns={
    "mapCoordonneesLatitude": "latitude",
    "mapCoordonneesLongitude": "longitude"
}).reset_index(drop=True)

# Fusion coordonnées + données brutes
df_map = pd.concat([X_raw.reset_index(drop=True), coords], axis=1)

# 📌 Mode bien existant
if mode_simulation == "🗂️ Choisir un bien existant":
    commune_cible = X_raw.iloc[idx].get("commune")
    surface = X_raw.iloc[idx].get("surface", 50)
    df_map = df_map[df_map["commune"] == commune_cible].reset_index(drop=True)

# 🧠 Prédiction pour tous les biens de la commune
df_predict_encoded = X_encoded.loc[df_map.index].copy()
df_predict_input = df_predict_encoded[model.feature_names_in_].apply(pd.to_numeric, errors='coerce')

# Supprimer les lignes avec valeurs manquantes
df_predict_input = df_predict_input.dropna()
df_map = df_map.loc[df_predict_input.index]  # Synchroniser les index

df_map["prix_m2"] = model.predict(df_predict_input)
df_map["prix_total"] = df_map["prix_m2"] * df_map["surface"]

# Moyenne communale
prix_moyen_commune = df_map["prix_m2"].mean()

nb_similaires = df_map["surface"].apply(lambda s: abs(s - surface) <= surface * 0.1).sum() - 1

st.markdown(f"🔍 **{nb_similaires} biens** dans la commune ont une surface similaire (±10%) au bien sélectionné.")

# 🛠️ Mode manuel

if mode_simulation == "🗂️ Choisir un bien existant":
    bien_selectionne = X_raw.iloc[[idx]].merge(coords, left_index=True, right_index=True)
    bien_selectionne = bien_selectionne[bien_selectionne["commune"] == commune_cible]
    lat_sel = bien_selectionne["latitude"].values[0]
    lon_sel = bien_selectionne["longitude"].values[0]
elif mode_simulation == "🛠️ Entrer mes propres caractéristiques":
    commune_cible = st.selectbox("📍 Sélectionne une commune", df_map["commune"].dropna().sort_values().unique())
    df_map = df_map[df_map["commune"] == commune_cible].reset_index(drop=True)
    lat_sel = df_map["latitude"].mean()
    lon_sel = df_map["longitude"].mean()
    surface = df_map["surface"].mean()

afficher_heatmap = st.toggle("Afficher la heatmap des prix au m²", value=True)
# Création de la carte
m = folium.Map(location=[lat_sel, lon_sel], zoom_start=11)

for i, row in df_map.iterrows():
    if mode_simulation == "🗂️ Choisir un bien existant" and abs(row["latitude"] - lat_sel) < 0.0001 and abs(row["longitude"] - lon_sel) < 0.0001:
        color = "red"
        radius = 8
    elif abs(row.get("surface", 0) - surface) <= surface * 0.1:
        color = "green"
        radius = 5
    else:
        color = "blue"
        radius = 4

    popup = f"""
    <b>Surface</b> : {row.get('surface', 'n/a')} m²<br>
    <b>Pièces</b> : {row.get('nb_pieces', 'n/a')}<br>
    <b>Étage</b> : {row.get('etage', 'n/a')}<br>
    <b>DPE</b> : {row.get('dpe', 'n/a')}<br>
    <b>Chauffage</b> : {row.get('chauffage_energie', 'n/a')}<br>
    <b>Commune</b> : {row.get('commune', 'n/a')}<br>
    <b>Prix estimé</b> : {row.get('prix_m2', 0):.2f} €/m²<br>
    <b>Prix total</b> : {row.get('prix_total', 0):,.0f} €
    """

    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=popup
    ).add_to(m)

# 🔴 Point fictif animé en mode manuel (🏢 ou 🏠 avec effet pulsation)
if mode_simulation == "🛠️ Entrer mes propres caractéristiques":
    popup = f"""
    <b>📍 Vous êtes ici</b><br>
    <b>Commune sélectionnée</b> : {commune_cible}<br>
    <i>(Point approximatif au centre de la commune)</i>
    """
    icon_html = f"""
    <div style='font-size:24px; animation: pulse 1.5s infinite;'>
        {"🏢" if typedebien == "Appartement" else "🏠"}
    </div>
    <style>
    @keyframes pulse {{
      0% {{ transform: scale(1); }}
      50% {{ transform: scale(1.3); }}
      100% {{ transform: scale(1); }}
    }}
    </style>
    """
    icon = folium.DivIcon(html=icon_html)
    folium.Marker(
        location=[lat_sel, lon_sel],
        popup=popup,
        icon=icon
    ).add_to(m)

# Affichage carte
st.markdown("## 🗺️ Carte des biens dans la commune")

# 🔥 Heatmap des prix au m² (si données disponibles et si activée)
if afficher_heatmap and "prix_m2" in df_map.columns and "latitude" in df_map.columns and "longitude" in df_map.columns:
    heat_data = [
        [row["latitude"], row["longitude"], row["prix_m2"]]
        for _, row in df_map.iterrows()
        if not pd.isna(row["latitude"]) and not pd.isna(row["longitude"]) and not pd.isna(row["prix_m2"])
    ]
    if heat_data:
        HeatMap(heat_data, radius=15, max_zoom=13, blur=10, min_opacity=0.4).add_to(m)

folium_static(m)

# -------------------------------------------------------------------
# 🗺️ Carte Choroplèthe des prix moyens au m² par commune (vue globale)
# -------------------------------------------------------------------
import json
from branca.colormap import linear
import folium
from streamlit_folium import folium_static


# 🔁 Bloc GLOBAL pour carte choroplèthe et top 10 (ne dépend pas de commune sélectionnée)
df_all_encoded = X_encoded.copy()
df_all_raw = X_raw.copy()
df_all_coords = pd.read_excel(coord_path).rename(columns={
    "mapCoordonneesLatitude": "latitude",
    "mapCoordonneesLongitude": "longitude"
}).reset_index(drop=True)

df_global_map = pd.concat([df_all_raw.reset_index(drop=True), df_all_coords], axis=1)
df_global_input = df_all_encoded[model.feature_names_in_].apply(pd.to_numeric, errors='coerce')
df_global_input = df_global_input.dropna()
df_global_map = df_global_map.loc[df_global_input.index].copy()

df_global_map["prix_m2"] = model.predict(df_global_input)



st.markdown("## 🗺️ Carte des prix moyens par commune (vue globale)")

try:
    # Charger le GeoJSON des communes
    geo_path = "data/communes_haut_rhin.geojson"
    with open(geo_path, encoding="utf-8") as f:
        geo_json = json.load(f)

    # Calcul global du prix moyen au m² par commune (avec toutes les données, pas filtrées)
    prix_par_commune_global = df_global_map.groupby("commune")[["prix_m2"]].mean().reset_index()

    # 🎨 Création de la palette dynamique selon les valeurs de prix
    colormap = linear.YlOrRd_09.scale(
        prix_par_commune_global["prix_m2"].min(),
        prix_par_commune_global["prix_m2"].max()
    )
    colormap.caption = "Prix moyen au m² (€)"

    # Carte centrée sur le Haut-Rhin
    m_choro = folium.Map(location=[47.8, 7.3], zoom_start=9)

    folium.Choropleth(
        geo_data=geo_json,
        name="Prix moyen au m²",
        data=prix_par_commune_global,
        columns=["commune", "prix_m2"],
        key_on="feature.properties.nom",
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        nan_fill_color="gray",
        legend_name="Prix moyen au m² (€)"
    ).add_to(m_choro)

    folium.LayerControl().add_to(m_choro)
    folium_static(m_choro)

    # 🏅 Top 10 des communes les plus chères
    st.markdown("### 🏅 Top 10 des communes les plus chères")
    top10_chères = prix_par_commune_global.sort_values(by="prix_m2", ascending=False).head(10)
    st.table(top10_chères.rename(columns={"commune": "Commune", "prix_m2": "Prix moyen (€/m²)"}))

    # 🪙 Top 10 des communes les moins chères
    st.markdown("### 🪙 Top 10 des communes les moins chères")
    top10_pas_chères = prix_par_commune_global.sort_values(by="prix_m2", ascending=True).head(10)
    st.table(top10_pas_chères.rename(columns={"commune": "Commune", "prix_m2": "Prix moyen (€/m²)"}))

except Exception as e:
    st.error(f"Erreur lors de la génération de la carte choroplèthe : {e}")


# Légende
with st.expander("ℹ️ Légende des couleurs sur la carte"):
    st.markdown("""
    - 🔴 Bien sélectionné  
    - 🟢 Biens avec une surface similaire (±10%)  
    - 🔵 Autres biens de la même commune  
    - 🏢 / 🏠 : votre position fictive (mode manuel)  
    - 🌡️ **Heatmap** (optionnelle) : plus la couleur est **chaude (rouge/jaune)**, plus le **prix au m² estimé est élevé**
    """)