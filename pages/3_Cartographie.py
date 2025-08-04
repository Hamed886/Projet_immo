import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Cartes Immobilières", layout="wide")

st.title("🗺️ Visualisation des cartes immobilières")

# Choix de la carte
carte = st.selectbox("Choisissez une carte à afficher :", [
    "Carte des prix immobiliers",
    "Fusion zones et biens",
    "Carte avec les données enrichies"

])

# Dictionnaires de mapping
carte_fichiers = {
    "Carte des prix immobiliers": "carte_prix_immobilier.html",
    "Fusion zones et biens": "fusion_zones_et_biens.html",
    "Carte avec les données enrichies": "carte.html"
    }

carte_commentaires = {
    "Carte des prix immobiliers": """
> **Description :** Cette carte affiche la répartition géographique des prix immobiliers par zone.  
> **Utilité :** Identifier les zones les plus chères ou les plus abordables pour orienter les investissements ou comparer les marchés.
""",
    "Fusion zones et biens": """
> **Description :** Cette carte présente la fusion des zones géographiques avec les biens immobiliers disponibles ou étudiés.  
> **Utilité :** Permet de visualiser la densité ou la couverture des biens selon les zones, utile pour la sectorisation commerciale ou l’analyse de couverture.

""",
    "Carte avec les données enrichies": """
> **Description :** Cette carte affiche les données immobilières enrichies avec des informations complémentaires (ex : socio-démographie, environnement, etc).
> **Utilité :** Permet une analyse plus fine et contextualisée du marché immobilier pour une prise de décision éclairée.
"""
}

# Affichage de la carte
nom_fichier = carte_fichiers[carte]

if os.path.exists(nom_fichier):
    with open(nom_fichier, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=700)

    # Affichage du commentaire lié
    st.markdown(carte_commentaires[carte])
else:
    st.error(f"⚠️ Le fichier {nom_fichier} est introuvable.")
