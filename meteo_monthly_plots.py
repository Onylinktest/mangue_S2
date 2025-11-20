import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================
# PARAMÈTRES À ADAPTER
# ==========================

# Dossiers d'entrée / sortie
METEO_DIR = Path("output/meteo")
IMG_DIR = Path("output/img")
CSV_DIR = Path("output/csv")

# Nom du fichier météo à traiter (dans METEO_DIR)
METEO_FILE = "meteo_2019-08-01_2025-03-01_55.23416_-21.06223.txt"

# Séparateur utilisé dans le fichier météo
SEP = ";"


# ==========================
# FONCTIONS
# ==========================

def load_meteo_file(filepath: Path) -> pd.DataFrame:
    """Charge le fichier météo et prépare les colonnes."""
    df = pd.read_csv(filepath, sep=SEP)
    
    # Conversion de la date
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Colonnes numériques à convertir
    num_cols = ["rr", "glot", "tn", "tm", "tx", "etp"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def monthly_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrégation mensuelle :
    - rr, glot : somme mensuelle
    - tn, tm, tx, etp : moyenne mensuelle
    """
    agg_dict = {
        "rr": "sum",
        "glot": "sum",
        "tn": "mean",
        "tm": "mean",
        "tx": "mean",
        "etp": "mean",
    }
    # On ne garde que les colonnes présentes
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    df_month = df.resample("M", on="Date").agg(agg_dict)
    return df_month


def plot_combined_rr_tm(df_month: pd.DataFrame, outpath: Path, title_suffix=""):
    """Graphique combiné précipitations (barres) + tm (courbe)."""
    if not {"rr", "tm"}.issubset(df_month.columns):
        print("Impossible de faire le graphique combiné (rr + tm manquants).")
        return
    
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    
    # Histogramme des précipitations en steelblue
    ax1.bar(df_month.index, df_month["rr"], width=20, color="steelblue")
    ax1.set_ylabel("Précipitations mensuelles (mm)")
    
    # Courbe de température moyenne en orange
    ax2 = ax1.twinx()
    ax2.plot(df_month.index, df_month["tm"], marker="o", color="orange")
    ax2.set_ylabel("Température moyenne mensuelle (°C)")
    
    title = "Précipitations mensuelles (bleu) et température moyenne mensuelle (orange)"
    if title_suffix:
        title += f"\n{title_suffix}"
    plt.title(title)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_monthly_single_var(df_month: pd.DataFrame, var: str, outpath: Path):
    """Graphique mensuel pour une seule variable."""
    if var not in df_month.columns:
        print(f"Variable {var} absente, pas de graphique.")
        return
    
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    
    if var == "rr":
        # rr en histogramme
        ax.bar(df_month.index, df_month[var], width=20)
    else:
        # Autres variables en courbe
        ax.plot(df_month.index, df_month[var], marker="o")
    
    ax.set_xlabel("Date")
    
    ylabels = {
        "rr": "Précipitations mensuelles (mm)",
        "glot": "Rayonnement global (somme mensuelle)",
        "tn": "Température minimale moyenne (°C)",
        "tm": "Température moyenne (°C)",
        "tx": "Température maximale moyenne (°C)",
        "etp": "ETP moyenne (mm/j)",
    }
    ax.set_ylabel(ylabels.get(var, var))
    ax.set_title(f"{var} - agrégation mensuelle")
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    # Création des dossiers de sortie
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    
    meteo_path = METEO_DIR / METEO_FILE
    if not meteo_path.exists():
        raise FileNotFoundError(f"Fichier météo introuvable : {meteo_path}")
    
    # 1. Chargement
    df = load_meteo_file(meteo_path)
    
    # 2. Agrégation mensuelle
    df_month = monthly_aggregation(df)
    
    # 3. Sauvegarde CSV
    base_name = meteo_path.stem  # nom de fichier sans extension
    csv_out = CSV_DIR / f"{base_name}_monthly.csv"
    df_month.to_csv(csv_out, sep=";")
    print(f"CSV mensuel enregistré dans : {csv_out}")
    
    # 4. Graphique combiné rr/tm
    img_combined = IMG_DIR / f"{base_name}_monthly_rr_tm.png"
    plot_combined_rr_tm(df_month, img_combined, title_suffix=base_name)
    print(f"Graphique combiné rr/tm enregistré dans : {img_combined}")
    
    # 5. Graphiques par variable
    vars_to_plot = ["rr", "glot", "tn", "tm", "tx", "etp"]
    for var in vars_to_plot:
        img_path = IMG_DIR / f"{base_name}_monthly_{var}.png"
        plot_monthly_single_var(df_month, var, img_path)
        if var in df_month.columns:
            print(f"Graphique mensuel pour {var} enregistré dans : {img_path}")


if __name__ == "__main__":
    main()
