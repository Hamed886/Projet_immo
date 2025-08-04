
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title="🏠 Prédiction Prix m²", layout="wide")
st.title("📊 Interface de Prédiction du Prix au m²")

st.markdown("Cette interface permet de prédire automatiquement le **prix au m²** sur les données de test en comparant au prix réel et en visualisant sur une carte interactive.")

# Choix du type de bien
type_bien = st.selectbox("🏘️ Type de bien", ["Appartement", "Maison"])

# Mapping des chemins
paths = {
    "Appartement": {
        "model": "models/et_appart.pkl",
        "x_test": "data/annonces_ventes_68_appartements_X_test.csv",
        "y_test": "data/y_a_test.csv",
        "coord": "data/map_appartements.xlsx"
    },
    "Maison": {
        "model": "models/xgb_maison.pkl",
        "x_test": "data/annonces_ventes_68_maisons_X_test.csv",
        "y_test": "data/y_m_test.csv",
        "coord": "data/map_maisons.xlsx"
    }
}

# Chargement du modèle
model_path = paths[type_bien]["model"]
if not os.path.exists(model_path):
    st.error(f"❌ Modèle introuvable : {model_path}")
    st.stop()
model = joblib.load(model_path)
st.success(f"✅ Modèle chargé : {os.path.basename(model_path)}")

# Chargement des données
def load_csv(path):
    return pd.read_csv(path, sep=";", encoding="ISO-8859-1")

X_test = load_csv(paths[type_bien]["x_test"])
y_test = load_csv(paths[type_bien]["y_test"])
coords = pd.read_excel(paths[type_bien]["coord"])

# Alignement des colonnes
model_columns = model.feature_names_in_.tolist()
X_test_encoded = pd.get_dummies(X_test)
for col in model_columns:
    if col not in X_test_encoded.columns:
        X_test_encoded[col] = 0
X_test_encoded = X_test_encoded[model_columns]

# Prédictions
y_pred = model.predict(X_test_encoded)
df_resultat = X_test.copy()
df_resultat["prix_m2_predit"] = y_pred
df_resultat["prix_m2_reel"] = y_test.values
df_resultat["ecart"] = df_resultat["prix_m2_reel"] - df_resultat["prix_m2_predit"]

# Scores
rmse = mean_squared_error(df_resultat["prix_m2_reel"], df_resultat["prix_m2_predit"]) ** 0.5
mae = mean_absolute_error(df_resultat["prix_m2_reel"], df_resultat["prix_m2_predit"])
r2 = r2_score(df_resultat["prix_m2_reel"], df_resultat["prix_m2_predit"])

# Résultats
st.markdown("## 📋 Résultats des prédictions")
st.dataframe(df_resultat[["prix_m2_predit", "prix_m2_reel", "ecart"]].head(10))

# Scores
st.markdown("## 📊 Évaluation du modèle")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:,.2f} €/m²")
col2.metric("MAE", f"{mae:,.2f} €/m²")
col3.metric("R²", f"{r2:.2%}")

# Carte Folium
st.markdown("## 🗺️ Carte interactive des écarts")
st.markdown(
    """
    **🟢 Vert** : Prédiction **proche du prix réel** (écart < 250 €/m²)  
    **🔴 Rouge** : Prédiction **éloignée du prix réel** (écart ≥ 250 €/m²)
    """
)

coords = coords.rename(columns={
    "mapCoordonneesLatitude": "latitude",
    "mapCoordonneesLongitude": "longitude"
}).reset_index(drop=True)

df_map = pd.concat([df_resultat.reset_index(drop=True), coords], axis=1)
m = folium.Map(location=[47.75, 7.3], zoom_start=10)

for _, row in df_map.iterrows():
    popup = f"<b>Prévu</b> : {row['prix_m2_predit']:.0f} €<br><b>Réel</b> : {row['prix_m2_reel']:.0f} €<br><b>Écart</b> : {abs(row['ecart']):.0f} €"
    color = "green" if abs(row["ecart"]) < 250 else "red"
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=popup
    ).add_to(m)

# Légende personnalisée
legend_html = """
<div style='position: fixed; 
     bottom: 40px; left: 40px; width: 200px; height: 80px; 
     background-color: white; z-index:9999; font-size:14px;
     border:1px solid grey; padding: 10px; border-radius: 5px;'>
<b>Légende :</b><br>
🟢 Prédiction correcte (< 250 €)<br>
🔴 Prédiction éloignée (≥ 250 €)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
folium_static(m)

# Téléchargement
st.download_button(
    "📥 Télécharger les résultats",
    df_resultat.to_csv(index=False).encode(),
    file_name=f"predictions_{type_bien.lower()}.csv"
)
