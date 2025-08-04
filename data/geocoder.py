import pandas as pd
import reverse_geocoder as rg
import os

# === Dossier du script (où se trouve aussi le fichier d'entrée) ===
DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# === Nom du fichier d'entrée et de sortie (dans le même dossier que le script) ===
NOM_FICHIER_ENTREE = "X_test_appart_raw.csv"
NOM_FICHIER_SORTIE = "X_test_appart_raw_avec_communes.csv"

# === Construction des chemins absolus ===
CHEMIN_ENTREE = os.path.join(DOSSIER_SCRIPT, NOM_FICHIER_ENTREE)
CHEMIN_SORTIE = os.path.join(DOSSIER_SCRIPT, NOM_FICHIER_SORTIE)

# === Vérification existence fichier d'entrée ===
if not os.path.exists(CHEMIN_ENTREE):
    raise FileNotFoundError(f"❌ Fichier introuvable : {CHEMIN_ENTREE}")

# === Chargement du fichier ===
df = pd.read_csv(CHEMIN_ENTREE, encoding="8859-1", sep=";")
df.columns = df.columns.str.lower()

# === Identification auto des colonnes latitude/longitude ===
lat_cols = [col for col in df.columns if "lat" in col]
lon_cols = [col for col in df.columns if "lon" in col]

if not lat_cols or not lon_cols:
    raise ValueError("❌ Les colonnes latitude ou longitude sont manquantes dans le fichier d'entrée.")

lat_col = lat_cols[0]
lon_col = lon_cols[0]

for i in range(0, len(coords), batch_size):
    batch_coords = coords[i:i + batch_size]
    try:
        results = rg.search(batch_coords, mode=1)
        communes += [res["name"] for res in results]
    except Exception as e:
        print(f"❌ Erreur lors du reverse geocoding du batch {i}: {e}")
        communes += [None] * len(batch_coords)
batch_size = 1000
communes = []
for i in range(0, len(coords), batch_size):
    batch_coords = coords[i:i + batch_size]
    results = rg.search(batch_coords, mode=1)
    communes += [res["name"] for res in results]

# === Ajout de la colonne "commune"
df["commune_reverse_geocoder"] = communes

# === Sauvegarde
df.to_csv(CHEMIN_SORTIE, index=False, encoding="8859-1", sep=";")
print(f"✅ Fichier enrichi exporté : {CHEMIN_SORTIE}")

