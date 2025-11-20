#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyse d'une parcelle de mangue Sentinel-2 :
- Calcul des indices NDVI, MSI, CI_RE
- Séries temporelles (moyenne ± écart-type)
- Courbes brutes et lissées (Savitzky–Golay)
- Barres verticales tous les 3 mois (à partir du 1er janvier de chaque année)
- ACP (PCA) sur NDVI, MSI, CI_RE
- Export des figures (PNG) et données agrégées / scores PCA (CSV)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --------------------------------------------------------------------
# 1. Paramètres utilisateur
# --------------------------------------------------------------------

# Dossier où se trouve le CSV d’entrée
DATA_DIR = Path("data")

# Nom du fichier CSV (dans DATA_DIR)
INPUT_FILE = "S2_L2A_mango_pixel_time_series_parcelle4.csv"

# Dossier racine pour les sorties
OUTPUT_DIR = Path("output")


# --------------------------------------------------------------------
# 2. Fonctions utilitaires
# --------------------------------------------------------------------

def ensure_output_dirs(output_dir: Path):
    """Crée les sous-dossiers output/images et output/csv si besoin."""
    img_dir = output_dir / "images"
    csv_dir = output_dir / "csv"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, csv_dir


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule NDVI, MSI, CI_RE à partir des bandes B4, B5, B8, B11."""
    # Cast en float
    B4 = df["B4"].astype(float)
    B5 = df["B5"].astype(float)
    B8 = df["B8"].astype(float)
    B11 = df["B11"].astype(float)

    # NDVI
    denom_ndvi = (B8 + B4).replace(0, np.nan)
    df["NDVI"] = (B8 - B4) / denom_ndvi

    # MSI
    denom_msi = B8.replace(0, np.nan)
    df["MSI"] = B11 / denom_msi

    # CI_RE
    denom_cire = B5.replace(0, np.nan)
    df["CI_RE"] = B8 / denom_cire - 1

    # Nettoyage des inf / -inf
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def aggregate_time_series(df: pd.DataFrame, indices: list[str]) -> pd.DataFrame:
    """Agrège par date : mean et std pour chaque indice."""
    ts = df.groupby("date")[indices].agg(["mean", "std"]).reset_index()

    # Renommer les colonnes
    new_cols = ["date"]
    for idx in indices:
        new_cols.extend([f"{idx}_mean", f"{idx}_std"])
    ts.columns = new_cols

    ts = ts.sort_values("date")
    return ts


def build_quarter_dates(ts: pd.DataFrame) -> list[pd.Timestamp]:
    """Construit la liste des dates (1er janv, 1er avril, 1er juillet, 1er oct) pour chaque année."""
    if ts.empty:
        return []
    start_year = ts["date"].dt.year.min()
    end_year = ts["date"].dt.year.max()
    quarter_dates: list[pd.Timestamp] = []
    for y in range(start_year, end_year + 1):
        for m in [1, 4, 7, 10]:
            quarter_dates.append(pd.Timestamp(year=y, month=m, day=1))
    return quarter_dates


def add_quarter_lines(ax, quarter_nums: np.ndarray):
    """Ajoute des lignes verticales trimestrielles (en coordonnées num/date2num)."""
    for qn in quarter_nums:
        ax.axvline(qn, color="gray", linestyle="--", alpha=0.3)


def savgol_smooth(y: np.ndarray, max_window: int = 11, polyorder: int = 3) -> np.ndarray:
    """Lissage Savitzky–Golay robuste aux petites tailles d'échantillons."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y.copy()

    # Choix de la fenêtre (impair, <= n)
    if n >= max_window:
        window = max_window
    else:
        window = (n // 2) * 2 + 1
    if window >= n:
        window = n - 1 if (n - 1) % 2 == 1 else max(3, n - 2)

    if window < 3:
        return y.copy()

    return savgol_filter(y, window_length=window, polyorder=min(polyorder, window - 1))


def plot_mean_std(ts: pd.DataFrame,
                  idx: str,
                  quarter_nums: np.ndarray,
                  img_dir: Path,
                  basename: str):
    """Graphe moyenne ± std pour un indice, avec barres trimestrielles."""
    if ts.empty:
        return

    dates = ts["date"]
    x = mdates.date2num(dates)
    mean = ts[f"{idx}_mean"].astype(float).values
    std = ts[f"{idx}_std"].astype(float).values

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot_date(x, mean, "-", label="Moyenne", color="tab:orange")
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, color="tab:orange")

    add_quarter_lines(ax, quarter_nums)

    ax.set_title(f"{idx} — moyenne ± écart-type (barres tous les 3 mois)")
    ax.set_xlabel("Date")
    ax.set_ylabel(idx)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_file = img_dir / f"{idx}_mean_std_{basename}.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_raw_smooth(ts: pd.DataFrame,
                    idx: str,
                    quarter_nums: np.ndarray,
                    img_dir: Path,
                    basename: str):
    """Graphe courbe brute + lissée pour un indice, avec barres trimestrielles."""
    if ts.empty:
        return

    dates = ts["date"]
    x = mdates.date2num(dates)
    y = ts[f"{idx}_mean"].astype(float).values
    y_smooth = savgol_smooth(y, max_window=11, polyorder=3)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot_date(x, y, "-", alpha=0.4, label="Brut", color="goldenrod")
    ax.plot_date(x, y_smooth, "-", label="Lissé", linewidth=2, color="tab:blue")

    add_quarter_lines(ax, quarter_nums)

    ax.set_title(f"{idx} — courbe brute et lissée (barres tous les 3 mois)")
    ax.set_xlabel("Date")
    ax.set_ylabel(idx)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_file = img_dir / f"{idx}_raw_smooth_{basename}.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)


def run_pca(df_indices: pd.DataFrame,
            indices: list[str],
            csv_dir: Path,
            img_dir: Path,
            basename: str):
    """ACP sur NDVI, MSI, CI_RE + export scores et figure PC1/PC2."""
    if df_indices.empty:
        return

    # Standardisation
    X = df_indices[indices].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 2 composantes
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    df_scores = df_indices.copy()
    df_scores["PC1"] = pcs[:, 0]
    df_scores["PC2"] = pcs[:, 1]

    # Export des scores PCA
    pca_csv = csv_dir / f"pca_scores_{basename}.csv"
    df_scores[["PC1", "PC2", "date"]].to_csv(pca_csv, index=False)

    # Scatter plot PC1 vs PC2
    var_ratio = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df_scores["PC1"], df_scores["PC2"], alpha=0.3, s=8)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% variance)")
    ax.set_title(f"PCA – NDVI, MSI, CI_RE ({basename})")
    plt.tight_layout()

    pca_png = img_dir / f"PCA_NDVI_MSI_CIRE_{basename}.png"
    plt.savefig(pca_png, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------
# 3. Main
# --------------------------------------------------------------------

def main():
    # Dossiers de sortie
    img_dir, csv_dir = ensure_output_dirs(OUTPUT_DIR)

    # Fichier d'entrée
    csv_path = DATA_DIR / INPUT_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable : {csv_path}")

    basename = csv_path.stem  # nom sans extension

    # Chargement
    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError("La colonne 'date' est absente du CSV.")

    df["date"] = pd.to_datetime(df["date"])

    # Filtrage SCL == 4 (végétation) si SCL existe
    if "SCL" in df.columns:
        df = df[df["SCL"] == 4].copy()
    else:
        df = df.copy()

    # Calcul indices
    df = compute_indices(df)

    # Liste des indices
    indices = ["NDVI", "MSI", "CI_RE"]

    # Suppression des NaN sur indices
    df = df.dropna(subset=indices + ["date"])

    if df.empty:
        raise ValueError("Aucune donnée valide après filtrage et calcul des indices.")

    # Agrégation temporelle
    ts = aggregate_time_series(df, indices)

    # Export time series agrégées
    ts_csv = csv_dir / f"ts_indices_{basename}.csv"
    ts.to_csv(ts_csv, index=False)

    # Dates trimestrielles (pour les barres verticales)
    quarter_dates = build_quarter_dates(ts)
    quarter_nums = mdates.date2num(quarter_dates) if quarter_dates else np.array([])

    # Graphes pour chaque indice
    for idx in indices:
        plot_mean_std(ts, idx, quarter_nums, img_dir, basename)
        plot_raw_smooth(ts, idx, quarter_nums, img_dir, basename)

    # ACP sur les pixels individuels
    df_for_pca = df[["date"] + indices].copy()
    run_pca(df_for_pca, indices, csv_dir, img_dir, basename)

    print(f"Analyse terminée pour {csv_path}")
    print(f"Images dans : {img_dir.resolve()}")
    print(f"CSV dans : {csv_dir.resolve()}")


if __name__ == "__main__":
    main()
