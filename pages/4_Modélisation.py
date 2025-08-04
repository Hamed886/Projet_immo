import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Modélisation", layout="wide")

st.title("📊 Modélisation des prix au m²")

st.markdown("## 🔍 Objectif")
st.markdown("L’objectif est de prédire le **prix au m²** des biens immobiliers à partir des caractéristiques enrichies des annonces. "
            "Nous avons séparé les modèles pour les **appartements** et les **maisons**, afin de tenir compte des différences de structure, de volume et de variables explicatives.")

st.markdown("## Modèles testés")
st.markdown("Nous avons testé 16 modèles de régression pour chaque type de bien (Appartements et Maisons), en évaluant leurs performances à l’aide des métriques suivantes :")
st.markdown("- RMSE : Erreur quadratique moyenne")
st.markdown("- MAE : Erreur absolue moyenne")
st.markdown("- R² : Coefficient de détermination")

# Résultats Appartements
st.markdown("### 📈 Résultats Appartements")
df_appartements = pd.DataFrame({
    "Modèle": ["ExtraTrees", "XGBoost", "LightGBM", "RandomForest", "GradientBoosting", "MLP", "LinearRegression", "Ridge", "Lasso", "ElasticNet", "DecisionTree", "AdaBoost", "SVR", "Dummy", "KNN", "GaussianProcess"],
    "RMSE": [519.14, 524.30, 528.04, 530.96, 600.57, 658.98, 697.81, 702.76, 706.65, 725.40, 782.23, 831.01, 1071.85, 1073.90, 1139.83, 2760.44],
    "MAE": [348.92, 371.19, 384.56, 372.77, 441.79, 490.84, 528.12, 530.36, 532.82, 547.94, 512.92, 664.04, 862.83, 877.53, 868.75, 2542.09],
    "R²": [0.7663, 0.7616, 0.7582, 0.7555, 0.6872, 0.6234, 0.5777, 0.5717, 0.5670, 0.5437, 0.4694, 0.4011, 0.0037, -0.00006, -0.1266, -5.6074]
})
st.dataframe(df_appartements, use_container_width=True)

# Résultats Maisons
st.markdown("### 🏠 Résultats Maisons")
df_maisons = pd.DataFrame({
    "Modèle": ["XGBoost", "LightGBM", "RandomForest", "ExtraTrees", "GradientBoosting", "MLP", "Ridge", "LinearRegression", "Lasso", "ElasticNet", "AdaBoost", "DecisionTree", "KNN", "SVR", "Dummy", "GaussianProcess"],
    "RMSE": [574.29, 579.64, 594.22, 601.68, 634.32, 650.04, 705.67, 705.93, 707.62, 730.50, 785.36, 863.95, 899.70, 949.24, 950.90, 2888.75],
    "MAE": [415.66, 414.93, 415.57, 416.72, 452.95, 471.33, 518.07, 518.29, 519.84, 534.27, 601.80, 592.54, 678.35, 724.77, 725.69, 2683.95],
    "R²": [0.6348, 0.6279, 0.6090, 0.5991, 0.5544, 0.5321, 0.4486, 0.4482, 0.4456, 0.4091, 0.3170, 0.1735, 0.1037, 0.0023, -0.0011, -8.2398]
})
st.dataframe(df_maisons, use_container_width=True)

# Affichage des visuels comparatifs sauvegardés
st.markdown("### 📊 Visualisations comparatives des performances")
image_paths = [
    "data/rmse_maisons.png",
    "data/mae_maisons.png",
    "data/r2_maisons.png",
    "data/rmse_apparts.png",
    "data/mae_apparts.png",
    "data/r2_apparts.png"
]

for path in image_paths:
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"Image non trouvée : {path}")

st.markdown("## 🎯 Sélection des modèles pour la suite")
st.success("➡️ **ExtraTrees** est retenu pour les **appartements** (meilleure performance globale - R² = 0.7663).\n"
           "➡️ **XGBoost** est retenu pour les **maisons** (meilleur compromis RMSE/MAE/R²).")

st.markdown("## 📊 Analyse comparative Apparts vs Maisons")
st.markdown("""
### 🔍 **Performance globale**
- Les performances sont **nettement meilleures sur les Apparts** que sur les Maisons :
  - **R² maximal 0.766 (ExtraTrees)** pour les Apparts
  - **R² maximal 0.634 (XGBoost)** pour les Maisons
- Cela suggère une **meilleure homogénéité** et/ou un **volume de données plus important** sur les Apparts, facilitant la modélisation.

### ⚠️ **Sensibilité aux erreurs extrêmes**
- L'écart entre **RMSE et MAE** reste **plus marqué sur les Maisons**, traduisant une **sensibilité accrue aux valeurs extrêmes** dans ce segment.
- Cela pourrait être dû à une **plus grande diversité** de types de biens, de surfaces ou de localisations.

### 🧠 **Hiérarchie des modèles**
- Les **modèles d’ensemble** (ExtraTrees, XGBoost, RandomForest) surpassent nettement les **modèles linéaires** (Ridge, SVR) sur les deux types de biens.
- Cela confirme que la complexité des interactions entre variables est mieux captée par des **approches non linéaires et robustes**.

### 📈 **Apports visuels**
- Les **graphiques R², MAE et RMSE** permettent une **lecture immédiate des performances** et mettent en évidence :
  - La **supériorité d’ExtraTrees** sur les Apparts
  - La **robustesse de XGBoost** sur les Maisons
- Ces visuels **justifient les modèles retenus** pour l’étape suivante.
""")
