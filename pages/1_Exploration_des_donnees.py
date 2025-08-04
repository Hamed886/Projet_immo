import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current directory to path


st.set_page_config(page_title="Exploration des Données", layout="wide")

st.title("🔍 Exploration des Données Enrichies")

st.markdown("## 🧬 Sources de Données Mobilisées")
st.markdown("""
Notre jeu de données a été constitué à partir de **plusieurs sources complémentaires**, toutes détaillées dans le rapport final :

- 🏠 **Annonces immobilières** : base brute des biens en vente sur le Haut-Rhin (prix, surface, caractéristiques, DPE…)
- 📊 **Données INSEE** : variables socio-démographiques, équipements, commerces, santé, culture, mobilité
- 💶 **Revenu fiscal de référence** : indicateur du pouvoir d’achat local
- 🗺️ **Informations géographiques** : codes postaux, noms de communes, zones urbaines/rurales
- 📌 **Enrichissements métiers** : scores synthétiques (économie, santé, culture), ratios de dotation communale, données de criminalité
- 📈 **Données DVF** : valeurs foncières pour les séries temporelles et prévisions de prix
""")

st.markdown('## 🎯 Objectif')            
st.markdown("""

**Objectif** : montrer les insights les plus significatifs

**Visualisations sélectionnées** :
1. Aperçu rapide du dataset
2. Taux de valeurs manquantes
3. Distribution du `prix_m2_vente` (cible de la modélisation)
4. Impact de la classe énergétique (DPE)
5. Corrélations entre variables numériques

➡️ Données utilisées : `annonces_ventes_enrichies_rvf_bpe.csv`
""")

try:
    # Chargement du fichier enrichi final utilisé pour le rapport
    df = pd.read_csv("data/annonces_ventes_enrichies_rvf_bpe.csv", sep=";", encoding="latin1")
    st.success("✅ Données chargées avec succès.")

    # Sélection rapide de colonnes pertinentes si présentes
    st.markdown("### 🧾 Aperçu rapide du dataset (5 premières lignes)")
    st.dataframe(df.head())

    # Nettoyage léger : filtrer les valeurs aberrantes de prix_m2 pour lisibilité
    if "prix_m2_vente" in df.columns:
        df = df[df["prix_m2_vente"].between(500, 8000)]

    # 1️⃣ Taux de valeurs manquantes
    st.markdown("### 1️⃣ Analyse rapide des valeurs manquantes")
    na_percent = df.isna().mean().sort_values(ascending=False) * 100
    na_filtered = na_percent[na_percent > 5]

    if not na_filtered.empty:
        st.dataframe(na_filtered.round(1).to_frame(name="Taux de NA (%)"))
        st.markdown("""
        Certaines variables présentent un taux significatif de valeurs manquantes :  
        - `dpeL` : classe énergétique, à recoder ou exclure selon les cas  
        - `surface_terrain`, `loyer`, `balcon` : souvent absentes, mais peu influentes sur la cible `prix_m2_vente`
        """)

    # 2️⃣ Distribution du prix au m² (log scale)
    st.markdown("### 2️⃣ Distribution du Prix au m² (log scale)")
    st.markdown("""
    La variable `prix_m2_vente` est la **cible principale** du projet.  
    Cette distribution log-normale justifie le recours à une transformation logarithmique et au filtrage des outliers.
    """)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    log_prices = np.log10(df["prix_m2_vente"])
    sns.histplot(log_prices, bins=50, kde=True, ax=ax1, color="#1f77b4")
    ax1.set_title("Distribution du prix au m² (log10)", fontsize=14)
    ax1.set_xlabel("log₁₀(prix_m2) → échelle logarithmique", fontsize=12)
    ax1.set_ylabel("Nombre d’occurrences", fontsize=12)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Annotation facultative pour la soutenance
    median_val = np.median(df["prix_m2_vente"])
    ax1.axvline(np.log10(median_val), color='red', linestyle='--')
    ax1.text(np.log10(median_val)+0.1, ax1.get_ylim()[1]*0.9,
         f"Médiane : {int(median_val)} €/m²", color="red")
    ax1.set_xscale("log")
    ax1.set_title("Distribution du prix au m² (log scale)")
    st.pyplot(fig1)

    st.markdown("""
    La distribution du prix_m2 est fortement asymétrique et suit une loi log-normale, comme souvent en immobilier.
    La transformation logarithmique permet d’atténuer cette asymétrie, de stabiliser la variance, et de rendre les modèles plus robustes.
    La médiane à 2 582 €/m² est retenue comme valeur de référence pour les visualisations.
    Le filtrage des valeurs extrêmes (outliers) permet de mieux visualiser la distribution sans les points extrêmes qui pourraient fausser l’analyse.
    """)

    # 3️⃣ Analyse du DPE (classe énergétique)
    if "dpeL" in df.columns:
        st.markdown("### 2️⃣ Impact du DPE sur le prix au m²")
        st.markdown("""
        Le **DPE (Diagnostic de Performance Énergétique)** est une variable catégorielle ordinale importante.  
        Elle impacte la valeur perçue d’un bien → plus la classe est basse (A, B…), plus le bien est valorisé.
        """)

        # Définir un ordre explicite des classes DPE
        ordre_dpe = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'NS', 'VI', '0']
        df['dpeL'] = pd.Categorical(df['dpeL'], categories=ordre_dpe, ordered=True)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x="dpeL", y="prix_m2_vente", ax=ax2, order=ordre_dpe, palette="Set2")
        ax2.set_title("Boxplot prix_m2 selon la classe DPE (dpeL)")
        st.pyplot(fig2)

    st.markdown("""
    Les biens classés A, B, C présentent une valeur médiane plus élevée, ce qui traduit une meilleure valorisation à la vente.
    À l’inverse, les biens classés F ou G sont en moyenne moins chers.
    On observe aussi une forte dispersion des prix, surtout pour les classes A à D, ce qui suggère l’influence croisée d’autres variables (localisation, surface, etc.).
    Les modalités comme "NS", "0", "VI" (non spécifié, vide ou invalide) doivent être traitées ou filtrées en amont pour ne pas biaiser le modèle.
    """)

    # 4️⃣ Matrice de corrélation entre variables numériques
    st.markdown("### 4️⃣ Corrélation entre les variables numériques")
    st.markdown("""
    La matrice de corrélation permet de visualiser les redondances et liens linéaires entre variables quantitatives.  
    Cela permet de guider la **sélection de features** et de détecter la multicolinéarité.
    """)
    numeric_df = df.select_dtypes(include=["float64", "int64"]).dropna(axis=1)
    corr_matrix = numeric_df.corr()

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", ax=ax3)
    ax3.set_title("Matrice de corrélation des variables numériques")
    st.pyplot(fig3)

    st.markdown("""
    L’analyse de la matrice de corrélation montre :
    - Une corrélation modérée entre prix_m2_vente et le revenu fiscal moyen
    - Les scores socio-économiques liés à l’éducation, la culture, la santé, etc.
    - Des relations faibles ou nulles entre plusieurs variables, justifiant l’usage de modèles complexes (forêts aléatoires, boosting).
    - Une colinéarité forte entre surface, nb_pieces, prix_bien, mensualiteFinance ainsi que les paires *_brut et *_ratio_1000 des indicateurs INSEE
    """)

except Exception as e:
    st.error(f"❌ Erreur lors du chargement ou traitement des données : {e}")

    st.markdown("""
    L’analyse de la matrice de corrélation montre :
    - Une corrélation modérée entre prix_m2_vente et le revenu fiscal moyen
    - Les scores socio-économiques liés à l’éducation, la culture, la santé, etc.
Des relations faibles ou nulles entre plusieurs variables, justifiant l’usage de modèles complexes (forêts aléatoires, boosting).
Une colinéarité forte entre :

surface, nb_pieces, prix_bien, mensualiteFinance

Les paires *_brut et *_ratio_1000 des indicateurs INSEE
    """)

# 5️⃣ Conclusion
st.markdown("## Conclusion de l'exploration")
st.markdown("""
Cette exploration des données a permis de mettre en lumière plusieurs insights clés :
- La distribution des prix au m² révèle des disparités importantes selon les zones géographiques et les caractéristiques des biens.
- Le DPE apparaît comme un facteur déterminant de la valorisation immobilière, avec des classes énergétiques plus élevées corrélées à des prix plus élevés.
- La matrice de corrélation a mis en évidence des relations intéressantes entre certaines variables, tout en soulignant la nécessité d'approches de modélisation avancées pour capturer la complexité des interactions.
""")

# 6️⃣ Liens utiles
st.markdown("## 🔗Liens utiles")
st.markdown("""
- [Données DVF](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/)
- [Base permanente des équipements](https://www.insee.fr/fr/statistiques/2011101)
""")

# 7️⃣ Affichage du rapport df d'exploration
import os
import streamlit as st
import streamlit.components.v1 as components

st.markdown("## 📄 Rapport d'exploration des données")
file_name = "data/rapport_DF.html"

if not os.path.exists(file_name):
    st.error(f"❌ Le fichier `{file_name}` est introuvable. Vérifie le chemin ou génère le rapport.")
else:
    with st.expander("ℹ️ À propos du rapport", expanded=True):
        st.markdown(f"**✅ Rapport généré :** `{file_name}`")
        st.markdown("Ce rapport contient une **analyse exploratoire complète** :")
        st.markdown("- Statistiques descriptives")
        st.markdown("- Visualisations automatiques")
        st.markdown("- Analyse des corrélations")
        st.markdown("- Détection des valeurs manquantes")
    
    with open(file_name, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 💡 Affichage intégré du rapport HTML
    st.markdown("---")
    components.html(html_content, height=1000, scrolling=True)