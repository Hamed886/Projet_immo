import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
<div style="text-align:center; padding-top: 10px;">
    <h1 style="font-size: 2.3rem; font-weight: 1200; line-height: 2.0; margin: 0;">
        🚀 Ouverture<br>
        <span style="color:#FF4B4B;">D’un Modèle Robuste à une Expertise Immobilière Augmentée</span>
    </h1>
</div>
""", unsafe_allow_html=True)

# CSS pour ajuster les marges et rendre le tout plus compact
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.st-emotion-cache-16txtl3 {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.8, 1.2])

with col1:
    st.info("🎯 **Notre Constat : La Limite des Données Structurées**", icon="🎯")
    st.markdown("""
    Nos modèles sont performants (R² de 0.78 sur les appartements) , mais ils expliquent moins bien la variance pour les maisons (R² de 0.66).

    **Pourquoi ?** Car la valeur d'un bien unique (cachet, état, potentiel) est invisible dans les chiffres. Elle se cache dans les **photos** et les **textes d'annonces**.
    """)

    st.warning("🔭 **Vision Long Terme : Simulateur Dynamique**", icon="🔭")
    st.markdown("""
    - **Modélisation prédictive des tendances** par quartier (LSTM, Transformers).
    - **Déploiement en production (MLOps)** pour un système auto-apprenant.
    """)


with col2:
    st.success("🏆 **Notre Priorité Stratégique : L'IA Multimodale**", icon="🏆")
    st.markdown("""
    Pour dépasser les limites actuelles, l'étape la plus logique est de capturer cette information non-structurée en fusionnant trois types de données.

    **1. Vision par Ordinateur : Quantifier l'inquantifiable**
    - **Analyse des photos** avec `ResNet` ou `EfficientNet` pour détecter des concepts comme "lumineux", "à rénover", "haut standing".

    **2. NLP : Décoder l'argumentaire de vente**
    - **Analyse des descriptions** avec `CamemBERT` pour extraire des signaux clés : "vue dégagée", "quartier calme", "travaux à prévoir".

    🎯 **Objectif :** Un modèle unique fusionnant **Tabulaire + Texte + Image** pour expliquer la variance des prix des maisons et viser un **R² > 0.85** sur tous les segments.
    """)


with st.expander("🤔 **Anticipation des Défis**"):
    st.markdown("""
    - **Gouvernance des données :** La collecte par scraping doit être robuste et respecter le cadre légal (RGPD, droits d'auteur).
    - **Qualité des données :** Les photos et textes sont souvent bruités et nécessitent un pré-traitement complexe.
    - **Coût infrastructurel :** L'entraînement de modèles de Deep Learning est gourmand en ressources de calcul (GPU).
    - **Interprétabilité :** Expliquer les prédictions d'un modèle multimodal est un défi technique, bien que des outils comme SHAP soient prometteurs.
    """)

