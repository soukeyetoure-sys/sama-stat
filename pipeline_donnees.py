import os
import json
import re
import asyncio
import nest_asyncio
import pandas as pd
from llama_parse import LlamaParse
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

# 1. FONCTIONS DE NETTOYAGE (Espaces, symboles, conversion automatique de nombres)
def nettoyer_texte(texte):
    if not isinstance(texte, str):
        return texte
    texte = re.sub(r'[\xa0\u200b\u200e\u200f\t]', ' ', texte)
    texte = re.sub(r'\n{3,}', '\n\n', texte)
    return re.sub(r' +', ' ', texte).strip()

def nettoyer_valeur_numerique(valeur):
    if isinstance(valeur, (int, float)) or not isinstance(valeur, str):
        return valeur
    val_clean = valeur.strip().replace(" ", "").replace(",", ".")
    if re.match(r'^-?\d+$', val_clean):
        return int(val_clean)
    if re.match(r'^-?\d+\.\d+$', val_clean):
        return float(val_clean)
    return valeur

# 2. CONFIGURATION LLAMAPARSE POUR LES TABLEAUX ET LE TEXTE
# ✅ NOUVELLE CONFIGURATION
parser = LlamaParse(
    result_type="markdown",
    language="fr",
    user_prompt=(
        "Extrais l'intégralité du contenu sans aucune omission. "
        "Conserve fidèlement tous les tableaux au format Markdown structuré."
    ),
    verbose=True
)

# 3. PIPELINE D'EXTRACTION ET NETTOYAGE AUTOMATIQUE
async def lancer_pipeline(dossier_entree, dossier_sortie):
    # Création automatique du dossier de sortie s'il n'existe pas
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
        print(f"📁 Dossier de sortie créé : {dossier_sortie}")

    fichiers = sorted(os.listdir(dossier_entree))
    base_consolidee = []

    for fichier in fichiers:
        chemin = os.path.join(dossier_entree, fichier)
        if not os.path.isfile(chemin):
            continue

        print(f"\n--------------------------------------------------")
        print(f"🔄 Traitement & Nettoyage de : {fichier}")
        nom_base, _ = os.path.splitext(fichier)

        # 📊 A. Traitement des CSV
        if fichier.lower().endswith('.csv'):
            try:
                try:
                    df = pd.read_csv(chemin, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(chemin, encoding='latin-1')
                
                df = df.where(pd.notnull(df), None)
                
                donnees_matrice = [
                    {nettoyer_texte(k): nettoyer_valeur_numerique(nettoyer_texte(v)) for k, v in row.items()}
                    for row in df.to_dict(orient="records")
                ]
                
                obj = {
                    "nom_fichier": fichier,
                    "type": "CSV",
                    "colonnes": [nettoyer_texte(c) for c in df.columns],
                    "donnees": donnees_matrice
                }
                base_consolidee.append(obj)
                
                # Sauvegarde du CSV nettoyé
                with open(os.path.join(dossier_sortie, f"{nom_base}_clean.json"), "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=4)
                    
                print(f"✅ CSV extrait et nettoyé avec succès.")

            except Exception as e:
                print(f"❌ Erreur CSV ({fichier}): {e}")

        # 📑 B. Traitement des PDF avec LlamaParse
        elif fichier.lower().endswith('.pdf'):
            try:
                docs = await parser.aload_data(chemin)
                pages = [
                    {
                        "page": i + 1,
                        "texte_markdown": nettoyer_texte(d.text)
                    }
                    for i, d in enumerate(docs)
                ]
                
                obj = {
                    "nom_fichier": fichier,
                    "type": "PDF",
                    "total_pages": len(docs),
                    "pages": pages
                }
                base_consolidee.append(obj)

                # Sauvegarde du PDF extrait et nettoyé
                with open(os.path.join(dossier_sortie, f"{nom_base}_clean.json"), "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=4)
                    
                print(f"✅ PDF extrait et nettoyé avec succès ({len(docs)} pages).")

            except Exception as e:
                print(f"❌ Erreur LlamaParse ({fichier}): {e}")

    # 📁 C. Fichier récapitulatif consolidé
    with open(os.path.join(dossier_sortie, "00_base_nettoyee_complete.json"), "w", encoding="utf-8") as f:
        json.dump(base_consolidee, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 EXCELLENT ! Toutes les données du dossier '{dossier_entree}' ont été extraites, nettoyées et rangées dans '{dossier_sortie}/'")

if __name__ == "__main__":
    # Exécution sur le dossier 'data'
    DOSSIER_SOURCE = "./data"
    DOSSIER_DESTINATION = "./data_nettoyees"
    
    if os.path.exists(DOSSIER_SOURCE):
        asyncio.run(lancer_pipeline(DOSSIER_SOURCE, DOSSIER_DESTINATION))
    else:
        print(f"⚠️ Le dossier '{DOSSIER_SOURCE}' est introuvable. Assurez-vous qu'il existe à la racine du projet.")