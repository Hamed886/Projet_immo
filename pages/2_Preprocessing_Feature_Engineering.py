
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Préprocessing & Feature Engineering", layout="wide")
st.title("🛠️ Préprocessing & Feature Engineering")

# 🎯 Objectif
st.markdown("## 🎯 Objectif")
st.markdown("""
Cette section décrit l’ensemble du pipeline de transformation des données brutes en un jeu exploitable pour l'entraînement des modèles de prédiction.

Nous avons appliqué une série d'étapes de nettoyage, d’enrichissement externe et de **feature engineering métier** pour maximiser la performance et la robustesse des modèles.
""")
st.markdown("---")

# 📥 Illustration du pipeline
st.markdown("## 🔄 Pipeline de transformation")
st.image("data/image.png", caption="Processus de préparation des données")
st.markdown("---")

st.markdown("### 🧩 Décryptage du pipeline de transformation")
st.markdown("""
Ce pipeline est le cœur du traitement des données avant modélisation. Il suit une logique métier rigoureuse et s’appuie sur les apports du rapport final (section 3).

🔹 **Données brutes**  
→ Annonces immobilières issues de différentes sources : DVF, annonces 68, bases INSEE, BPE, etc.  
→ Variables souvent bruitées, incomplètes ou redondantes.

🔹 **Jonction des données de référence géographique**  
→ Intégration de référentiels INSEE (code commune, typologie urbaine, IRIS)  
→ Permet une granularité territoriale fine pour enrichir chaque bien

🔹 **Ajout d’un score d’accessibilité**  
→ Construction de scores BPE (équipements par commune) + revenu fiscal moyen  
→ Calcul brut + score pour 1000 habitants (densité ajustée à la population)  
→ Objectif : donner du **contexte local** au modèle (qualité de vie, services)

🔹 **Données enrichies prêtes pour la modélisation**  
→ Dataset final nettoyé, structuré, sans NA, enrichi et prêt à être passé dans les modèles supervisés.  
→ Ce traitement augmente la **robustesse, la valeur explicative** et l’**interprétabilité** des modèles ML.
""")

# 📦 Chargement du dataset pré-nettoyé
st.markdown("## 📦 Données enrichies et nettoyées")
try:
    df = pd.read_csv("data/annonces_ventes_enrichies_rvf_bpe.csv", sep=";", encoding="latin1")
    st.success(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
except Exception as e:
    st.error(f"Erreur lors du chargement du dataset : {e}")
st.markdown("---")

# 💡 Split Train/Test
st.markdown("### 🧪 Split train/test intégré au pipeline")
st.info("Le split train/test a été réalisé **avant tout traitement** afin d’éviter les fuites de données (data leakage). Toutes les imputations, encodages et scalings ont été faits uniquement sur les données d’entraînement.")
st.markdown("---")

# 🧼 Étapes de nettoyage
st.markdown("## 🧼 Nettoyage et transformation initiale")
st.markdown("""
- Suppression des colonnes avec fuite de cible (`prix_maison`, `prix_terrain`, etc.)
- Suppression des colonnes vides ou trop bruitées (`videophone`, `typedebien`, `mensualiteFinance`, etc.)
- Nettoyage de la variable `dpeL` (classe énergétique) et recodage ordonné (A → G)
- Gestion des valeurs aberrantes : suppression des lignes avec `prix_m2_vente` < 500 ou > 8000 €/m²
""")
st.markdown("---")

# 🧱 Feature Engineering
st.markdown("## 🧱 Variables dérivées et enrichissement externe")
st.markdown("""
Variables ajoutées pour renforcer la valeur prédictive :

- `surf_par_piece` : Surface habitable / nb de pièces
- `surface_anormale` : Anomalie si surf/piece < 6 ou > 60 m²
- `score_total_bpe` : équipements globaux dans la commune
- `score_ratio_bpe` : équipements pour 1000 habitants
- `revenu_fiscal_moyen` : indicateur socio-éco
- `typologie_territoriale` : zone rurale, mixte ou urbaine

📌 Données issues de croisements INSEE, BPE, DVF, géolocalisation.
""")
st.markdown("---")

# 📊 Visualisations
with st.expander("📊 Visualisations post-traitement"):

    if "prix_m2_vente" in df.columns:
        st.markdown("### ➤ Distribution du prix au m² (nettoyé)")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        sns.histplot(df["prix_m2_vente"].dropna(), bins=50, kde=True, ax=ax2, color="#1f77b4")
        ax2.set_title("Distribution du prix au m² après nettoyage")
        st.pyplot(fig2)

        st.markdown("#### Analyse de la distribution du prix au m²")
        st.markdown("""
        La distribution montre une **concentration des prix au m² entre 500 et 6000 €**, avec un pic autour de 3000 €/m².  
        Elle est **asymétrique à droite**, ce qui reflète :
        - des biens standards très présents (zones péri-urbaines du Haut-Rhin),
        - et une minorité de biens plus chers ou atypiques.

        Ce graphique **confirme l’efficacité du nettoyage** :
        - Suppression des valeurs aberrantes `< 500 €` ou `> 8000 €`,
        - Plus aucune valeur aberrante vers zéro ou extrême droite.

        Ce nettoyage améliore la robustesse des futurs modèles.  
        """)



# 📌 Impact métier
st.markdown("## Impact métier du preprocessing")
st.markdown("""
- 🔧 Nettoyage des extrêmes → évite les biais sur la prédiction des prix
- 🧠 Enrichissement externe → contexte local indispensable à l’estimation réelle
- 🏘️ Typologie territoriale → meilleur reflet de l’attractivité des zones
- 📈 Prêt pour industrialisation → structure compatible avec déploiement en API ou batch processing
""")
st.markdown("---")


# ✅ Résumé final
st.markdown("## Résumé")
st.markdown("""
✔️ Dataset nettoyé, enrichi et prêt pour modélisation  
✔️ Pipeline reproductible, traçable et documenté  
✔️ Variables explicatives de qualité métier  
✔️ Aucun NA dans les variables retenues

Prochaine étape : modélisation (régressions, modèles d’ensemble).
""")
