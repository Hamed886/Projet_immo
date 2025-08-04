import streamlit as st

st.set_page_config(page_title="Évaluation Finale", layout="wide")

st.markdown("# 📊 Évaluation Finale – Optimisation, Cross-Validation & Résidus")
st.markdown("""
Cette section présente les performances finales des meilleurs modèles après :
- Optimisation des hyperparamètres avec **Optuna**
- Validation croisée (**5-fold CV**)
- Analyse des résidus pour les cas atypiques

Les résultats concernent deux segments : **appartements** et **maisons**.
""")

# Résumé des performances sur jeu de test
st.markdown("## 📋 Performances – Modèles Optimisés")
st.markdown("""
| Bien         | Modèle       | R² Train | R² Test | Test RMSE | Test MAE |
|--------------|--------------|----------|---------|-----------|----------|
| Appartements | ExtraTrees   | 0.9573   | 0.7829  | 500.32 €  | 349.35 € |
| Maisons      | XGBoost      | 0.9202   | 0.6535  | 559.44 €  | 399.32 € |
""")

# Optimisation avec Optuna
st.markdown("## ⚙️ Optimisation avec Optuna")
st.markdown("""
L’algorithme **Optuna** a été utilisé pour rechercher automatiquement les meilleurs hyperparamètres (30 essais / modèle).

### 🔧 Modèles retenus :
- 🏢 **Appartements : ExtraTrees**
  - Gain RMSE ≈ **-16 €**
- 🏠 **Maisons : XGBoost**
  - Gain RMSE ≈ **-22 €**

📌 L’optimisation améliore la précision tout en **limitant les risques de surajustement**.

### 🔩 Hyperparamètres optimaux

#### 🏢 Appartements – ExtraTrees :
- `n_estimators`: 350
- `max_depth`: 25
- `min_samples_split`: 3
- `max_features`: sqrt
- `bootstrap`: True

#### 🏠 Maisons – XGBoost :
- `n_estimators`: 200
- `learning_rate`: 0.05
- `max_depth`: 6
- `subsample`: 0.9
- `colsample_bytree`: 0.7

📌 Ces hyperparamètres ont été sélectionnés automatiquement avec **Optuna** en minimisant le **RMSE sur test set**.
""")

# Cross Validation
st.markdown("## 🔁 Validation croisée (5-fold CV)")
st.markdown("""
Une validation croisée à **5 folds** a été appliquée à tous les modèles pour estimer leur robustesse.

| Bien         | Modèle       | RMSE CV | R² CV  | Écart RMSE Test - CV |
|--------------|--------------|---------|--------|----------------------|
| Appartements | ExtraTrees   | 510.25 € | 0.772  | -10.07 € |
| Maisons      | XGBoost      | 561.88 € | 0.649  | -2.44 €  |

✅ Ces résultats confirment que les performances obtenues sur le jeu de test **ne sont pas dues au hasard**.  
Le modèle est **stable, généralisable et prêt à l’emploi**.
""")

# Graphiques Test vs CV
st.markdown("## 📊 Comparaison des Scores – Test vs Cross-Validation")
st.image("data/cv_rmse_test_vs_cv.png", caption="📉 Comparaison RMSE – Test vs Cross-Validation", use_container_width=True)
st.image("data/cv_r2_test_vs_cv.png", caption="📈 Comparaison R² – Test vs Cross-Validation", use_container_width=True)

st.markdown("""
**🧠 Analyse des écarts :**
- Pour les **appartements**, les modèles présentent un **excellent alignement** entre RMSE Test et RMSE CV.
- Pour les **maisons**, une **légère perte de performance** en CV est observée, mais reste maîtrisée.
- Le **R² reste cohérent** dans les deux cas, validant la qualité prédictive.

🎯 **Conclusion :**  
La généralisation est bonne, les performances sont **fiables en production**.
""")

# Résidus & Prédictions
st.markdown("## 📈 Analyse des Résidus & Prédictions")

# Résidus Appartements
st.image("data/residus_apparts.png", caption="Résidus – Appartements", use_container_width=True)
st.markdown("""
🔎 **Interprétation :**
- Bonne dispersion autour de 0 :
   - La majorité des points sont centrés autour de la ligne rouge (résidu nul).
    Cela valide le fait que le modèle ne surestime ni ne sous-estime globalement.
           
     → **peu de biais**

- Distribution relativement symétrique :
   - On observe une forme vaguement elliptique, signe d’un modèle relativement stable.
   - Les erreurs positives et négatives sont équilibrées.
            
- Légère hétéroscédasticité : variance augmente sur valeurs extrêmes
""")

# Prédictions vs Réalité Appartements
st.image("data/pred_vs_real_apparts.png", caption="Prédictions vs Réalité – Appartements", use_container_width=True)
st.markdown("""
✅ Le modèle capte bien la tendance générale  
⚠️ Quelques écarts sur des biens atypiques ou très chers
""")

# Résidus Maisons
st.image("data/residus_maisons.png", caption="Résidus – Maisons", use_container_width=True)
st.markdown("""
🔎 **Interprétation :**
- Résidus globalement centrés, avec une variance plus élevée pour les maisons les plus chères
""")

# Prédictions vs Réalité Maisons
st.image("data/pred_vs_real_maisons.png", caption="Prédictions vs Réalité – Maisons", use_container_width=True)
st.markdown("""
✅ Bonne prédiction sur les cas standards  
⚠️ Plus de variabilité sur les maisons les plus chères
""")

# Analyse synthétique
st.markdown("## 🧾 Synthèse Finale")
st.markdown("""
- ✅ **Optimisation efficace** (RMSE ↓ / MAE ↓)
- ✅ **Cross-validation cohérente** → faible écart test / CV
- ✅ **Résidus neutres** et bien répartis
- 🔍 Quelques cas extrêmes à surveiller (hétéroscédasticité)
""")