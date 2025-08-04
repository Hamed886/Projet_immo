import streamlit as st

st.set_page_config(page_title="Mon Compagnon Immobilier", layout="wide")

st.title("🏠 Mon Compagnon Immobilier")
st.subheader("Projet DataScientest – Data Scientist | Soutenance Finale")

st.markdown("___")

# Objectifs
st.header("🎯 Objectifs du projet")
st.markdown("""
Le projet **Mon Compagnon Immobilier** a pour but d’accompagner les acheteurs dans leurs décisions en :
- Estimant le **prix au m² d’un bien** (maison ou appartement)
- **Prévoyant l’évolution des prix** immobiliers selon les zones géographiques
- Proposant une **aide à la décision territoriale** enrichie via des données socio-économiques, géographiques et de services

L’objectif est double :
1. **Prédiction individuelle** du prix au m² d’un bien donné
2. **Analyse temporelle et territoriale** pour anticiper les dynamiques de prix

""")

# Données utilisées
st.header("📁 Données utilisées")
st.markdown("""
- **Annonces immobilières** : plus de 20 000 annonces du Haut-Rhin
- **Données DVF** (valeurs foncières) : historiques des transactions
- **Open Data INSEE / BPE** : population, équipements, démographie, transports
- **Géodonnées** : géolocalisation, commune, statut urbain/rural
- **Autres sources** : criminalité, scores enrichis, données socio-économiques

Toutes les données ont été collectées, nettoyées et fusionnées pour obtenir un jeu complet et structuré.
""")

# Méthodologie
st.header("🧪 Méthodologie")
st.markdown("""
1. **Exploration des données** : visualisations, corrélations, outliers
2. **Nettoyage approfondi** : traitement des NA, valeurs aberrantes, uniformisation
3. **Feature engineering** :
   - `surf_par_piece`, `score_services`, ...
   - Encodages ordinals (DPE, INSEE_COM)
4. **Split Appartements / Maisons** pour pipelines indépendants
5. **Modélisation supervisée** + cross-validation + optimisations

""")

# Pipeline
st.header("⚙️ Pipeline de traitement")
st.markdown("""
- **Préprocessing robuste** : suppression des fuites de cible, encodage, imputations
- **Enrichissement externe** : fusion avec bases INSEE, BPE, scoring équipements
- **Séparation Appartements vs Maisons** pour améliorer les performances
- **Optimisation via Optuna** sur RMSE

Le pipeline est industrialisé, réplicable, et prêt pour une mise en production (déploiement API ou application Streamlit).
""")

# Résultats modèles
st.header("📈 Résultats des modèles")
st.markdown("""
### 🔹 Appartements
- Meilleur modèle : **ExtraTrees optimisé**
- R² test : **0.78**
- RMSE test : **503 €**
- MAE test : **351 €**

### 🔸 Maisons
- Meilleur modèle : **XGBoost optimisé**
- R² test : **0.66**
- RMSE test : **552 €**
- MAE test : **397 €**

> Les modèles d’ensemble surpassent les linéaires. Les appartements sont plus prédictibles que les maisons (marché plus homogène).
""")

# Interprétabilité
st.header("🧠 Interprétabilité & SHAP")
st.markdown("""
- Utilisation de **SHAP** pour interprétation globale et locale des prédictions
- Génération de **rapports interactifs Shapash** pour restitution métier
- Variables clés :
  - `surface`, `année construction`, `localisation`, `DPE`, `score transport`

Des visualisations (summary plots, waterfall, dependence) permettent d’expliquer chaque prédiction de manière lisible.
""")

# Série temporelle
st.header("⏳ Analyse temporelle – Données DVF")
st.markdown("""
- Modèle utilisé : **Prophet**
- Résolution : **mensuelle** par région/département/statut urbain
- Prévision sur 6 mois glissants
- MAPE global :
   - < 1.2 dans les zones urbaines denses
   - > 5 dans les zones rurales à faible volume

Ce module permet d’**anticiper les tendances régionales** et de compléter l’analyse ponctuelle.
""")

# Ouvertures
st.header("🚀 Ouvertures IA & Améliorations futures")
st.markdown("""
1. **Limites des données structurées** :
   - Les modèles expliquent bien les appartements mais peinent avec les maisons.
   - Pourquoi ? Les éléments subjectifs (cachet, potentiel, luminosité…) ne sont pas captés par les variables tabulaires.
         
2. **Multimodalité**
   Fusionner photos, textes et données structurées pour capter la vraie valeur d’un bien :
   - Analyse d’images (ResNet / EfficientNet)
   - Analyse de textes (CamemBERT)
   - Données tabulaires enrichies
   - Objectif : R² > 0.85 sur tous les biens
            
3. **Vision long terme**
   Construire un simulateur dynamique :
   - Prédiction des tendances par quartier (LSTM, Transformers)
   -Intégration via MLOps pour industrialisation
            
4. **Anticipation des défis**: 
   - Gouvernance des données (RGPD, scraping)
   - Qualité des données (bruit, pré-traitement)
   - Coût infrastructurel (GPU, cloud)
   - Interprétabilité des modèles multimodaux (SHAP, LIME)
""")