import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from prophet import Prophet
import os, gc

from PIL import Image


st.set_page_config(page_title="Analyse prix m²", layout="wide")


def commentaire(texte):
    st.markdown(
        f"""
        <p style="font-size:16px; font-style:italic; color:grey;">
        {texte}
        </p>
        """, unsafe_allow_html=True
    )


# --- Titre ---
st.title("⏳ Analyse Temporelle des Prix Immobiliers")

st.markdown("""
<div style="font-family:Arial; font-size:18px">

<h3><b>Données utilisées</b></h3>

<ul>
  <li><b>DVF (Demandes de Valeurs Foncières)</b><br>
  Transactions immobilières en France (valeur foncière, surface, type de bien, date).</li>
  
  <li><b>Table régions/départements/communes (20230823-communes-departement-region.csv)</b><br>
  Pour enrichir les données géographiques afin d'affiner au niveau du département.</li>
  
  <li><b>Table Statut urbain (STATUT_COM_UU.csv)</b><br>
  Classification des communes (Banlieue, commune isolée, commune centrale, Hors unité urbaine).</li>
</ul>

<h3><b>Préparation & Méthodologie </b></h3>

<ul>
  <li>Exclusion des <b>valeurs aberrantes</b> (prix/m² extrêmes).</li>
  <li>Calcul du <b>prix au m²</b> par transaction.</li>
  <li>Aggrégation par mois des transactions</li>
  <li>Groupement par <b>région</b>, <b>département</b>, et <b>statut urbain</b>.</li>
  <li>Application d’un <b>logarithme</b> sur le prix/m² pour lisser les distributions.</li>
  <li><b>PROPHET</b> pour la prévision temporelle par département et statut urbain.</li>
</ul>




</div>
""", unsafe_allow_html=True)







# ================================
# Bloc Répartitions
# ================================
st.subheader("Répartitions")
choix_repartition = st.selectbox(
    " ",
    ["Transactions par région", "Transactions par statut"]
)

if choix_repartition == "Transactions par région":
    filepath = "data/graphes/transactions_par_region.png"
    st.image(filepath)

elif choix_repartition == "Transactions par statut":
    filepath = "data/graphes/transactions_par_statut.png"
    st.image(filepath)

# ================================
# Bloc Résultats
# ================================
st.subheader("Résultats")
choix = st.selectbox(
    " ",
    ["Tableau", "Barplot par région", "Transactions vs MAPE", 
     "Carte Appartement", "Carte Maison","Densité","Prédiction"]
)

if choix == "Tableau":
    filepath = "data/graphes/tableau.png"
    st.image(filepath)

elif choix == "Barplot par région":
    filepath = "data/graphes/barplot_par_region.png"
    st.image(filepath)
    commentaire("Ce graphique compare les erreurs de prédiction (MAPE) par région et par type de bien. "
                "Les barres permettent d'identifier les zones et catégories où le modèle est le plus fiable.")

elif choix == "Transactions vs MAPE":
    filepath = "data/graphes/transactions_vs_mape.png"
    st.image(filepath)

    commentaire("Ce nuage de points montre la relation entre le volume de transactions et la précision des prédictions. "
                "On observe que les départements avec peu de données présentent en général un MAPE plus élevé.")

elif choix == "Carte Appartement":
    filepath = "data/graphes/carte_appartement.png"
    st.image(filepath)
    commentaire("Cette carte illustre la précision des prédictions pour les appartements "
                "selon les départements et leur statut urbain.")

elif choix == "Carte Maison":
    filepath = "data/graphes/carte_maison.png"
    st.image(filepath)
    commentaire("Cette carte illustre la précision des prédictions pour les maisons "
                "selon les départements et leur statut urbain.")
    
elif choix == "Densité":
    filepath = "data/graphes/carte_densite.png"
    st.image(filepath)

elif choix == "Prédiction":
    filepath = "data/graphes/Figure_prophet_92.png"
    st.image(filepath)