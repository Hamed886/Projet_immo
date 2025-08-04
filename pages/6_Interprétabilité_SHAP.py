import streamlit as st
import os

st.set_page_config(page_title="🔍 Interprétabilité SHAPASH", layout="wide")

st.title("🧠 Interprétation des Modèles - SHAPASH")
st.markdown("Sélectionnez un **modèle** et un **type de bien** pour explorer les rapports générés avec Shapash.")

# 📁 Mapping des rapports disponibles
rapport_map = {
    "Appartements": {
        "RandomForest": "rapport_shapash_randomforest_appart.html",
        "ExtraTrees": "rapport_shapash_extratrees_appart.html",
        "LightGBM": "rapport_shapash_lightgbm_appart.html",
        "XGBoost": "rapport_shapash_xgboost_appart.html",
    },
    "Maisons": {
        "RandomForest": "rapport_shapash_randomforest_maison.html",
        "ExtraTrees": "rapport_shapash_extratrees_maison.html",
        "LightGBM": "rapport_shapash_lightgbm_maison.html",
        "XGBoost": "rapport_shapash_xgboost_maison.html",
    }
}

col1, col2 = st.columns(2)
with col1:
    bien = st.selectbox("🏠 Type de Bien", ["Appartements", "Maisons"])
with col2:
    modele = st.selectbox("📈 Modèle", ["RandomForest", "ExtraTrees", "LightGBM", "XGBoost"])

# 🔎 Synthèses spécifiques
if bien == "Appartements" and modele == "ExtraTrees":
    st.info("""
    🔍 **Synthèse sur le modèle ExtraTrees - Appartements** :

    Le modèle ExtraTrees montre une structure d'interprétabilité très marquée autour des **indicateurs énergétiques et géographiques**. La variable **`dpeL`**, qui semble représenter une classe énergétique peu performante, domine largement les contributions. Cela indique que le modèle pénalise fortement les biens à faible performance énergétique.

    Viennent ensuite la **longitude** (`mapCoordonneesLongitude`) et le **score GES** (`ges_class_encoded`), montrant que le modèle est sensible à la **localisation** est-ouest et à l'empreinte environnementale du bien. L’**accessibilité aux transports** est également bien captée, renforçant la pertinence du modèle pour des contextes urbains où la mobilité est un critère fort.

    La variable **`annee_construction`**, corrélée à la vétusté, agit aussi de manière significative. En revanche, des variables souvent prioritaires comme `surface` ou `nb_pieces` sont ici reléguées au second plan, ce qui peut surprendre. Cela s’explique probablement par une structure d’interactions non linéaires détectée par ExtraTrees, qui favorise certaines combinaisons géo-énergétiques plus discriminantes.

    En résumé, ce modèle est **hautement interprétable**, mais sa logique peut parfois surprendre par rapport aux intuitions métier classiques. Il met en avant des critères **énergétiques, temporels et contextuels** comme clés de valorisation.
    """)

if bien == "Appartements" and modele == "LightGBM":
    st.info("""
    🔍 **Synthèse sur le modèle LightGBM - Appartements** :

    Ce modèle met en avant une combinaison équilibrée de critères **géographiques**, **dimensionnels** et **énergétiques**. Les variables les plus contributives sont **`mapCoordonneesLongitude`**, **`dpeD`**, **`score_transport_ratio_1000`** et **`surface`**, reflétant un intérêt particulier pour l’emplacement, l’efficacité énergétique moyenne et la taille du logement.

    Le modèle capte bien la **mobilité locale** et l’**ancienneté** du bien via les variables `annee_construction`, `annee`, et `logement_neuf`. Les variables énergétiques de classe intermédiaire (`dpeD`, `ges_class_encoded`) montrent une sensibilité modérée sans effet de seuil extrême.

    Le modèle offre une **lecture intuitive et structurée**, bien alignée avec les critères attendus pour des logements urbains, tout en gardant une bonne généralisabilité.
    """)

if bien == "Appartements" and modele == "RandomForest":
    st.info("""
    🔍 **Synthèse sur le modèle RandomForest - Appartements** :

    Le modèle RandomForest accorde un **poids majeur à `dpeC`**, illustrant une forte valorisation des logements avec une performance énergétique correcte. Viennent ensuite les variables **structurelles et géographiques** comme `surface`, `mapCoordonneesLongitude`, et `score_transport_ratio_1000`.

    Il en ressort un profil d’interprétation orienté vers l’**équilibre entre dimension physique, localisation et critères énergétiques**. Toutefois, une dépendance excessive à une seule classe énergétique (`dpeC`) pourrait induire un biais si les données sont déséquilibrées ou mal renseignées.
    """)

if bien == "Appartements" and modele == "XGBoost":
    st.info("""
    🔍 **Synthèse sur le modèle XGBoost - Appartements** :

    Le modèle XGBoost donne une place centrale à **`score_transport_ratio_1000`**, **`surface`**, **`dpeD`** et **`annee`**, soulignant l’importance de **l’accessibilité**, **la taille**, **la performance énergétique moyenne**, et **l’ancienneté**.

    Le modèle fait preuve d’une répartition fluide des importances, sans domination excessive, ce qui est typique d’un gradient boosting bien régularisé. Il est particulièrement **adapté aux zones urbaines** où la mobilité et l’énergie jouent un rôle fort, tout en intégrant des critères classiques comme la superficie.
    """)

if bien == "Maisons" and modele == "RandomForest":
    st.info("""
    🔍 **Synthèse sur le modèle RandomForest - Maisons** :

    Le modèle RandomForest met en avant la variable **`dpeC`** de manière écrasante, signe que la **performance énergétique correcte** constitue un facteur majeur de valorisation dans l’évaluation des maisons. Ce poids peut traduire une forte sensibilité du modèle aux seuils réglementaires ou aux comportements d’achat éco-responsables.

    En second plan, on retrouve des variables **géographiques** (`mapCoordonneesLongitude`) et **dimensionnelles** (`surface_terrain`, `surface`), ce qui confirme que **localisation et taille du bien** forment une base solide pour la prédiction.

    Le modèle intègre également des indicateurs **structurels** (`annee_construction`, `annee`) et **contextuels** (`score_transport_ratio_1000`, `Revenu_fiscal...`), suggérant une bonne prise en compte de l’accessibilité, du tissu urbain et de l’ancienneté.

    Toutefois, la **domination très marquée de `dpeC`** pourrait entraîner un **biais de pondération** si cette variable est mal renseignée ou trop corrélée à d’autres. Il conviendra de la surveiller attentivement en production.

    Globalement, le modèle offre une interprétabilité correcte, avec des signaux clairs, mais un **déséquilibre à modérer** sur le plan énergétique.
    """)

if bien == "Maisons" and modele == "LightGBM":
    st.info("""
    🔍 **Synthèse sur le modèle LightGBM - Maisons** :

    Le modèle LightGBM valorise principalement des critères **dimensionnels**. Les variables **`surface`** et **`surface_terrain`** sont en tête, confirmant que la **taille habitable** et la **taille de la parcelle** sont des leviers majeurs de valorisation pour les maisons.

    La variable **`dpeC`**, indicateur de **performance énergétique correcte**, joue également un rôle important, mais moins dominant que dans d’autres modèles. Cela traduit une prise en compte équilibrée de l'efficacité énergétique, sans excès.

    Viennent ensuite des variables **structurelles** comme **`annee`**, **`annee_construction`**, et **`mapCoordonneesLongitude`**, illustrant l’impact de l’ancienneté du bien et de sa localisation est-ouest. Le **score transport** et le **revenu fiscal local** complètent cette logique, introduisant un facteur **contextuel et socio-économique** dans la prédiction.

    Ce modèle se distingue donc par une **hiérarchie logique et intuitive** des facteurs explicatifs : taille du bien > contexte urbain > ancienneté > énergie. Il reste **très interprétable** et bien adapté à la diversité des profils de maisons.
    """)

if bien == "Maisons" and modele == "XGBoost":
    st.info("""
    🔍 **Synthèse sur le modèle XGBoost - Maisons** :

    Le modèle XGBoost identifie un **équilibre pertinent** entre critères **énergétiques, dimensionnels et géographiques**. En tête, on retrouve **`dpeC`**, indicateur d’une **bonne performance énergétique**, soulignant l’intérêt croissant pour l’efficacité énergétique dans les biens résidentiels.

    Les variables **`surface`** et **`surface_terrain`** confirment le rôle central de la **taille habitable et foncière** dans la valorisation d’un bien. Ensuite, **`mapCoordonneesLongitude`** et **`annee`** traduisent l’impact de la **localisation est-ouest** et de l’**ancienneté** sur le prix, particulièrement vrai en zones périurbaines.

    On note aussi l’influence notable d’indicateurs de **contexte urbain** tels que `score_transport_ratio_1000` et `Revenu_fiscal...`, indiquant que XGBoost parvient à intégrer les **composantes socio-économiques et d’accessibilité**.

    Ce modèle se distingue par une **distribution progressive et cohérente des importances**, sans variable ultra-dominante. Cela le rend **interprétable, robuste**, et apte à généraliser sur une diversité de profils de maisons.
    """)

if bien == "Maisons" and modele == "ExtraTrees":
    st.info("""
    🔍 **Synthèse sur le modèle ExtraTrees - Maisons** :

    Ce modèle ExtraTrees valorise principalement des variables **géographiques et structurelles**. En tête, on trouve **`mapCoordonneesLongitude`** et **`logement_neuf`**, montrant que la **position longitudinale** et le fait qu’un bien soit neuf influencent fortement l’estimation. Cela suggère que l’emplacement à l’est ou à l’ouest d’une zone urbaine a un effet différenciant sur la valeur.

    La variable **`dpeL`**, qui semble refléter une mauvaise performance énergétique, reste fortement contributrice, mais n’est pas dominante. Cela montre que le modèle intègre bien ce critère sans en faire un biais systématique.

    D’autres variables comme **`Revenu_fiscal_de_r_f_r_...`**, **`annee_construction`**, et **`score_transport_ratio_1000`** traduisent un modèle qui capte **le profil socio-économique du quartier, l’ancienneté et l’accessibilité**. Enfin, des variables dimensionnelles comme `surface`, `surface_terrain` ou `nb_pieces` apparaissent dans le milieu du classement, apportant une base physique à l’estimation.

    En résumé, ce modèle est **bien équilibré**, multi-critères, et offre une **interprétation robuste** en milieu résidentiel périurbain ou rural, où la localisation et la nouveauté du bâti prennent souvent le pas sur des critères classiques.
    """)



# 📄 Affichage du rapport
file_name = rapport_map[bien][modele]
file_path = os.path.join("reports", file_name)

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=800, scrolling=True)
else:
    st.warning(f"🚧 Rapport introuvable : {file_path}")