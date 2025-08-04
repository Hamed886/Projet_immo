import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.graph_objects as go
from functools import lru_cache
import logging
from datetime import datetime

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la page
st.set_page_config(
    page_title="Simulateur Prix Immobilier",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache pour les modèles et données
@st.cache_resource
def load_models():
    """Charge les modèles une seule fois et les met en cache"""
    try:
        model_appart = joblib.load("models/et_appart.pkl")
        model_maison = joblib.load("models/xgb_maison.pkl")
        logger.info("Modèles chargés avec succès")
        return model_appart, model_maison
    except Exception as e:
        logger.error(f"Erreur chargement modèles: {e}")
        st.error("Impossible de charger les modèles. Vérifiez les fichiers.")
        st.stop()

@st.cache_data
def load_reference_data():
    """Charge les données de référence avec cache"""
    try:
        X_appart_encoded = pd.read_csv("data/annonces_ventes_68_appartements_X_test.csv", 
                                      sep=";", encoding="ISO-8859-1")
        X_maison_encoded = pd.read_csv("data/annonces_ventes_68_maisons_X_test.csv", 
                                     sep=";", encoding="ISO-8859-1")
        X_appart_raw = pd.read_csv("data/X_test_appart_raw.csv", 
                                 sep=";", encoding="ISO-8859-1")
        X_maison_raw = pd.read_csv("data/X_test_maison_raw.csv", 
                                 sep=";", encoding="ISO-8859-1")
        return X_appart_encoded, X_maison_encoded, X_appart_raw, X_maison_raw
    except Exception as e:
        logger.error(f"Erreur chargement données: {e}")
        st.error("Impossible de charger les données de référence.")
        st.stop()

# Constantes
DPE_MAPPING = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
CHAUFFAGE_ENERGIE_MAP = {"Electrique": 0, "Gaz": 1, "Fioul": 2, "Bois": 3, "Autre": 4}
CHAUFFAGE_SYSTEME_MAP = {"Individuel": 0, "Collectif": 1, "Autre": 2}
CHAUFFAGE_MODE_MAP = {"Radiateur": 0, "Plancher chauffant": 1, "Autre": 2}

# Statistiques de marché (à adapter selon vos données réelles)
MARKET_STATS = {
    "Appartement": {
        "prix_median": 2850,
        "prix_q1": 2200,
        "prix_q3": 3500,
        "tendance": "+3.2%"
    },
    "Maison": {
        "prix_median": 2450,
        "prix_q1": 1900,
        "prix_q3": 3100,
        "tendance": "+2.8%"
    }
}

def validate_inputs(form_data):
    """Valide les données d'entrée"""
    errors = []
    
    if form_data["surface"] <= 0:
        errors.append("La surface doit être positive")
    
    if form_data["surface"] > 500:
        errors.append("Surface inhabituellement grande (>500m²)")
    
    if form_data["nb_pieces"] > form_data["surface"] / 10:
        errors.append("Ratio pièces/surface inhabituel")
    
    if form_data["nb_toilettes"] > form_data["nb_pieces"]:
        errors.append("Plus de toilettes que de pièces ?")
    
    return errors

def encode_input_optimized(form, template_df):
    """Version optimisée de l'encodage avec validation"""
    X_input = pd.DataFrame(0, index=[0], columns=template_df.columns)
    
    # Mapping direct des valeurs numériques
    numeric_mappings = {
        "surface": form["surface"],
        "nb_pieces": form["nb_pieces"],
        "nb_toilettes": form["nb_toilettes"],
        "dpeL": DPE_MAPPING[form["dpe"]],
        "chauffage_energie": CHAUFFAGE_ENERGIE_MAP[form["chauffage_energie"]],
        "chauffage_systeme": CHAUFFAGE_SYSTEME_MAP[form["chauffage_systeme"]],
        "chauffage_mode": CHAUFFAGE_MODE_MAP[form["chauffage_mode"]]
    }
    
    # Mapping des booléens
    boolean_mappings = {
        "logement_neuf": form["logement_neuf"] == "Oui",
        "balcon": form["balcon"] == "Oui",
        "cave": form["cave"] == "Oui",
        "ascenseur": form["ascenseur"] == "Oui",
        "bain": form["bain"] == "Oui",
        "eau": form["eau"] == "Oui",
        "places_parking": form["parking"] == "Oui",
        "annonce_exclusive": form["exclusivite"] == "Oui"
    }
    
    # Application des mappings
    for col, val in numeric_mappings.items():
        if col in X_input.columns:
            X_input.loc[0, col] = val
    
    for col, val in boolean_mappings.items():
        if col in X_input.columns:
            X_input.loc[0, col] = int(val)
    
    # Exposition (one-hot encoding)
    exposition_col = f"exposition_{form['exposition'].lower()}"
    if exposition_col in X_input.columns:
        X_input.loc[0, exposition_col] = 1
    
    return X_input

def create_market_comparison(prediction, type_bien):
    """Crée une comparaison avec le marché"""
    stats = MARKET_STATS[type_bien]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Votre estimation",
            f"{prediction:.0f} €/m²",
            f"{((prediction/stats['prix_median'])-1)*100:.1f}% vs médiane"
        )
    
    with col2:
        st.metric(
            "Prix médian marché",
            f"{stats['prix_median']} €/m²",
            stats['tendance']
        )
    
    with col3:
        st.metric(
            "1er quartile",
            f"{stats['prix_q1']} €/m²"
        )
    
    with col4:
        st.metric(
            "3ème quartile",
            f"{stats['prix_q3']} €/m²"
        )

def generate_recommendation(prediction, stats, features):
    """Génère des recommandations basées sur la prédiction"""
    recommendations = []
    
    # Analyse du positionnement prix
    if prediction < stats['prix_q1']:
        recommendations.append("💡 Prix très compétitif - Bien positionné pour une vente rapide")
    elif prediction > stats['prix_q3']:
        recommendations.append("⚠️ Prix élevé - Assurez-vous que les prestations justifient ce niveau")
    
    # Analyse DPE
    if features.get('dpeL', 0) > 4:  # E, F ou G
        recommendations.append("🔋 Une amélioration énergétique pourrait augmenter la valeur")
    
    # Surface
    if features.get('surface', 0) < 30:
        recommendations.append("📐 Petit bien - Idéal pour investissement locatif")
    
    return recommendations

# Interface principale
def main():
    st.markdown('<h1 class="main-header">🏠 Simulateur de Prix Immobilier au m²</h1>', 
                unsafe_allow_html=True)
    
    # Chargement des ressources
    model_appart, model_maison = load_models()
    X_appart_encoded, X_maison_encoded, X_appart_raw, X_maison_raw = load_reference_data()
    
    # Sidebar pour les paramètres principaux
    with st.sidebar:
        st.header("🎯 Configuration")
        typedebien = st.radio(
            "Type de bien",
            ["Appartement", "Maison"],
            help="Le type de bien influence significativement le prix"
        )
        
        mode_simulation = st.radio(
            "Mode de simulation",
            ["🗂️ Bien existant", "🛠️ Personnalisé"],
            help="Choisissez un bien de référence ou créez le vôtre"
        )
        
        # Affichage des performances du modèle
        st.markdown("### 📊 Performance du modèle")
        if typedebien == "Appartement":
            st.info(f"MAE: 351.77 €/m²\nR²: ~0.85")
        else:
            st.info(f"MAE: 397.36 €/m²\nR²: ~0.82")
    
    # Sélection du modèle et des données
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
    
    # Corps principal
    if mode_simulation == "🗂️ Bien existant":
        idx = st.selectbox(
            "🔍 Sélectionne un bien existant",
            list(range(len(X_encoded))),
            format_func=lambda x: f"Bien #{x} - {X_raw.iloc[x].get('commune', 'Non précisée')}"
        )
        X_input = X_encoded.iloc[[idx]].copy()
        surface = X_raw.iloc[idx].get("surface", 50)
        commune = X_raw.iloc[idx].get("commune", "Non précisée")
        st.success("✅ Caractéristiques récupérées automatiquement")
    
    else:
        st.markdown("### 🏗️ Caractéristiques du bien")
        
        # Organisation en colonnes pour une meilleure UX
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Caractéristiques principales**")
            surface = st.number_input("Surface (m²)", 10, 300, 75, step=5)
            nb_pieces = st.number_input("Nombre de pièces", 1, 10, 4)
            nb_toilettes = st.number_input("Nombre de toilettes", 0, 5, 1)
            dpe = st.selectbox("Classe énergétique DPE", list(DPE_MAPPING.keys()), index=3)
        
        with col2:
            st.markdown("**Équipements**")
            logement_neuf = st.checkbox("Logement neuf")
            balcon = st.checkbox("Balcon")
            cave = st.checkbox("Cave")
            ascenseur = st.checkbox("Ascenseur")
            bain = st.checkbox("Baignoire")
            eau = st.checkbox("Salle d'eau")
            parking = st.checkbox("Place de parking")
        
        with col3:
            st.markdown("**Chauffage et autres**")
            exposition = st.selectbox("Exposition", ["Sud", "Est", "Nord", "Autre"])
            chauffage_energie = st.selectbox("Énergie", list(CHAUFFAGE_ENERGIE_MAP.keys()))
            chauffage_systeme = st.selectbox("Système", list(CHAUFFAGE_SYSTEME_MAP.keys()))
            chauffage_mode = st.selectbox("Mode", list(CHAUFFAGE_MODE_MAP.keys()))
            exclusivite = st.checkbox("Annonce exclusive")
        
        # Construction du formulaire
        form = {
            "surface": surface,
            "nb_pieces": nb_pieces,
            "nb_toilettes": nb_toilettes,
            "logement_neuf": "Oui" if logement_neuf else "Non",
            "balcon": "Oui" if balcon else "Non",
            "cave": "Oui" if cave else "Non",
            "ascenseur": "Oui" if ascenseur else "Non",
            "bain": "Oui" if bain else "Non",
            "eau": "Oui" if eau else "Non",
            "parking": "Oui" if parking else "Non",
            "exclusivite": "Oui" if exclusivite else "Non",
            "dpe": dpe,
            "exposition": exposition,
            "chauffage_energie": chauffage_energie,
            "chauffage_systeme": chauffage_systeme,
            "chauffage_mode": chauffage_mode,
        }
        
        # Validation
        errors = validate_inputs(form)
        if errors:
            for error in errors:
                st.warning(f"⚠️ {error}")
        
        X_input = encode_input_optimized(form, X_encoded)
        commune = "Non précisée"
    
    # Bouton de prédiction
    if st.button("🎯 Estimer le prix", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            try:
                # Préparation des données
                X_input.drop(columns=[col for col in ["date", "typedebien_lite"] 
                                    if col in X_input.columns], inplace=True, errors="ignore")
                X_input_final = X_input.loc[:, model.feature_names_in_].copy()
                X_input_final = X_input_final.apply(pd.to_numeric, errors='coerce')
                
                if X_input_final.isnull().values.any():
                    st.error("⛔ Données manquantes détectées")
                    st.stop()
                
                # Prédiction
                prediction = model.predict(X_input_final)[0]
                
                # Affichage des résultats
                st.markdown("---")
                st.markdown("## 📊 Résultats de l'estimation")
                
                # Box principale de prédiction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.metric(
                        "Prix estimé au m²",
                        f"{prediction:.0f} €/m²",
                        f"± {MAE:.0f} €/m²"
                    )
                with col2:
                    prix_total = prediction * surface
                    st.metric(
                        "Prix total estimé",
                        f"{prix_total:,.0f} €".replace(',', ' '),
                        f"Pour {surface} m²"
                    )
                with col3:
                    confidence = (1 - MAE/prediction) * 100
                    st.metric(
                        "Confiance",
                        f"{confidence:.0f}%"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Comparaison marché
                st.markdown("### 📈 Positionnement marché")
                create_market_comparison(prediction, typedebien)
                
                # Recommandations
                st.markdown("### 💡 Recommandations")
                recommendations = generate_recommendation(
                    prediction, 
                    MARKET_STATS[typedebien],
                    X_input_final.to_dict('records')[0]
                )
                for rec in recommendations:
                    st.info(rec)
                
                # Analyse SHAP
                with st.expander("🔍 Comprendre la prédiction (Analyse SHAP)", expanded=False):
                    with st.spinner("Calcul des contributions SHAP..."):
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_input_final)
                        
                        # Préparation des données SHAP
                        shap_input = shap.Explanation(
                            values=shap_values[0],
                            base_values=explainer.expected_value,
                            data=X_input_final.iloc[0],
                            feature_names=X_input_final.columns
                        )
                        
                        # DataFrame SHAP
                        shap_df = pd.DataFrame({
                            "Feature": shap_input.feature_names,
                            "Contribution": shap_input.values,
                            "Value": shap_input.data
                        })
                        shap_df = shap_df.reindex(
                            shap_df.Contribution.abs().sort_values(ascending=False).index
                        )
                        top_features = shap_df.head(10)
                        
                        # Visualisation améliorée
                        colors = ["#FF6B6B" if x > 0 else "#4ECDC4" 
                                for x in top_features["Contribution"]]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=top_features["Contribution"],
                            y=top_features["Feature"],
                            orientation="h",
                            text=[f"{contrib:+.0f} €/m²" 
                                for contrib in top_features["Contribution"]],
                            textposition="outside",
                            hovertemplate="<b>%{y}</b><br>" +
                                        "Contribution: %{x:+.2f} €/m²<br>" +
                                        "Valeur: %{customdata}<br>" +
                                        "<extra></extra>",
                            customdata=top_features["Value"],
                            marker_color=colors,
                            marker_line_color="rgba(0,0,0,0.1)",
                            marker_line_width=1
                        ))
                        
                        fig.update_layout(
                            title={
                                'text': "Top 10 des facteurs influençant le prix",
                                'font': {'size': 20}
                            },
                            xaxis_title="Impact sur le prix (€/m²)",
                            yaxis_title="",
                            yaxis=dict(autorange="reversed"),
                            height=500,
                            template="plotly_white",
                            font=dict(size=12),
                            showlegend=False,
                            margin=dict(l=150, r=50, t=50, b=50)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interprétation automatique
                        st.markdown("#### 💭 Interprétation automatique")
                        base_value = explainer.expected_value
                        if isinstance(base_value, np.ndarray):
                            base_value = base_value.item()
                        
                        st.info(f"Prix de base du modèle : **{base_value:.0f} €/m²**")
                        
                        for i, row in top_features.head(5).iterrows():
                            impact = "augmente" if row["Contribution"] > 0 else "diminue"
                            st.write(f"• **{row['Feature']}** = {row['Value']:.1f} "
                                   f"{impact} le prix de **{abs(row['Contribution']):.0f} €/m²**")
                
                # Export des résultats
                if st.button("📥 Exporter le rapport", type="secondary"):
                    rapport = f"""
RAPPORT D'ESTIMATION IMMOBILIÈRE
================================
Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Type de bien: {typedebien}
Commune: {commune}

ESTIMATION
----------
Prix au m²: {prediction:.2f} €
Surface: {surface} m²
Prix total: {prediction * surface:.0f} €
Intervalle de confiance: [{prediction - MAE:.2f} ; {prediction + MAE:.2f}] €/m²

CARACTÉRISTIQUES PRINCIPALES
---------------------------
{pd.DataFrame(form.items(), columns=['Caractéristique', 'Valeur']).to_string(index=False)}

ANALYSE SHAP - TOP 5 FACTEURS
-----------------------------
{top_features.head(5).to_string(index=False)}
"""
                    st.download_button(
                        label="Télécharger le rapport",
                        data=rapport,
                        file_name=f"estimation_{typedebien.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
            except Exception as e:
                logger.error(f"Erreur lors de la prédiction: {e}")
                st.error(f"Erreur lors du calcul: {str(e)}")

if __name__ == "__main__":
    main()


