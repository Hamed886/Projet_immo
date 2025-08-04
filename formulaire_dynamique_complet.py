
import streamlit as st

def generer_formulaire_dynamique(X_input, variables):
    for var in variables:
        if "surface" in var and "terrain" not in var:
            X_input.loc[0, var] = st.slider(f"{var} (m²)", 10, 300, 75)
        elif "nb" in var or "etage" in var:
            X_input.loc[0, var] = st.number_input(f"{var}", min_value=0, step=1, value=1)
        elif var.startswith("exposition_"):
            pass  # handled separately
        elif var in ["chauffage_energie", "chauffage_systeme", "chauffage_mode"]:
            options = {
                "chauffage_energie": ["Electrique", "Gaz", "Fioul", "Bois", "Autre"],
                "chauffage_systeme": ["Individuel", "Collectif", "Autre"],
                "chauffage_mode": ["Radiateur", "Plancher chauffant", "Autre"]
            }
            choix = st.selectbox(f"{var}", options[var])
            X_input.loc[0, var] = options[var].index(choix)
        elif var == "dpeL":
            ordre_dpe = ["A", "B", "C", "D", "E", "F", "G"]
            dpe = st.selectbox("Classe énergétique DPE", ordre_dpe, index=3)
            X_input.loc[0, var] = 7 - ordre_dpe.index(dpe)
        elif var.startswith("exposition_"):
            continue  # handled after
        elif var.startswith("typedebien_") or var.startswith("commune_"):
            continue  # trop nombreux ou hors scope manuel
        else:
            X_input.loc[0, var] = st.radio(f"{var} ?", ["Oui", "Non"]) == "Oui"

    # Exposition groupée (si colonnes présentes)
    expositions = [v for v in variables if v.startswith("exposition_")]
    if expositions:
        choix_exp = st.selectbox("Exposition", ["Sud", "Est", "Nord", "Autre"])
        for dir in ["sud", "est", "nord", "autre"]:
            col_name = f"exposition_{dir}"
            if col_name in variables:
                X_input.loc[0, col_name] = 1 if choix_exp.lower() == dir else 0

    return X_input
