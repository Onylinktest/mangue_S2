import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns

#!/usr/bin/env python3
"""
analyse_s2.py

Lire un CSV de séries temporelles de pixels Sentinel-2 (colonnes B2,B3,B4,B5,B6,B7,B8,B11,B12 et une colonne date)
Calculer un ensemble d'indices (végétation, red-edge, hydrique, sol, ratios) et tracer pour chaque indice la moyenne ± écart-type au fil du temps.

Usage:
    python analyse_s2.py data/S2_L2A_mango_pixel_time_series_parcelle4.csv --datecol date --outdir results

Le script:
- détecte et parse la colonne date
- calcule les indices (gestion des divisions par zéro -> NaN)
- groupe par date et calcule mean/std
- génère et sauvegarde un PNG par indice et un CSV résumé des indices par date
"""


import matplotlib.pyplot as plt

sns.set(style="whitegrid", context="talk")


REQUIRED_BANDS = {"B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"}


def safe_div(num, den):
    """Divide arrays, returning NaN where denominator is zero or NaN."""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def compute_indices(df):
    """Return a DataFrame of indices computed from band columns in df."""
    B = {b: df[b].astype("float64") for b in REQUIRED_BANDS if b in df.columns}
    # helper lambdas
    B2 = B.get("B2")
    B3 = B.get("B3")
    B4 = B.get("B4")
    B5 = B.get("B5")
    B6 = B.get("B6")
    B7 = B.get("B7")
    B8 = B.get("B8")
    B11 = B.get("B11")
    B12 = B.get("B12")

    idx = pd.DataFrame(index=df.index)

    # Végétation
    idx["NDVI"] = safe_div(B8 - B4, B8 + B4)
    idx["EVI2"] = 2.5 * safe_div(B8 - B4, B8 + 2.4 * B4 + 1)
    idx["SAVI"] = 1.5 * safe_div(B8 - B4, B8 + B4 + 0.5)
    idx["GNDVI"] = safe_div(B8 - B3, B8 + B3)
    idx["VARI"] = safe_div(B3 - B4, B3 + B4 - B2)

    # Red-edge
    idx["NDRE"] = safe_div(B8 - B5, B8 + B5)
    idx["NDRE2"] = safe_div(B8 - B6, B8 + B6)
    idx["NDRE3"] = safe_div(B8 - B7, B8 + B7)
    idx["CIre"] = safe_div(B8, B5) - 1
    idx["MTCI"] = safe_div(B6 - B5, B5 - B4)

    # Hydriques / stress
    idx["NDWI"] = safe_div(B3 - B8, B3 + B8)  # McFeeters
    idx["NDMI"] = safe_div(B8 - B11, B8 + B11)
    idx["MSI"] = safe_div(B11, B8)

    # Sol / surface nue
    idx["BSI"] = safe_div(B11 + B4 - B8 - B2, B11 + B4 + B8 + B2)
    idx["NDSI_soil"] = safe_div(B11 - B12, B11 + B12)

    # Ratios simples
    idx["R_rededge_red"] = safe_div(B5, B4)
    idx["R_green_red"] = safe_div(B3, B4)

    return idx


def plot_index_time_series(dates, mean_ser, std_ser, name, outpath, smooth_days=0):
    """Plot mean ± std over time and save figure."""
    plt.figure(figsize=(10, 4.5))
    if smooth_days and len(mean_ser) >= smooth_days:
        mean_plot = mean_ser.rolling(window=smooth_days, center=True, min_periods=1).mean()
        std_plot = std_ser.rolling(window=smooth_days, center=True, min_periods=1).mean()
    else:
        mean_plot = mean_ser
        std_plot = std_ser

    plt.plot(dates, mean_plot, label=f"{name} mean", color="tab:blue")
    plt.fill_between(dates, mean_plot - std_plot, mean_plot + std_plot, color="tab:blue", alpha=0.25,
                     label="±1 std")
    plt.scatter(dates, mean_ser, s=8, color="k", alpha=0.6)
    plt.title(f"{name} — moyenne ± écart-type")
    plt.xlabel("Date")
    plt.ylabel(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Calcul d'indices Sentinel-2 et tracés mean±std")
    default_csv = str(Path("data") / "S2_L2A_mango_pixel_time_series_parcelle4.csv")
    p.add_argument("csv", nargs="?", default=default_csv,
                   help=f"Chemin vers le CSV (par défaut: {default_csv})")
    p.add_argument("--datecol", default="date", help="Nom de la colonne date (par défaut 'date')")
    p.add_argument("--outdir", default="results", help="Dossier de sortie pour PNG/CSV")
    p.add_argument("--smooth", type=int, default=7, help="Lissage (window days) pour affichage, 0 pour aucun")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Fichier non trouvé: {csv_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Lecture CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise SystemExit(f"Erreur lecture CSV: {exc}")

    if args.datecol not in df.columns:
        # tenter quelques noms communs
        for alt in ("date", "Date", "time", "timestamp", "acq_date"):
            if alt in df.columns:
                df = df.rename(columns={alt: args.datecol})
                break
    if args.datecol not in df.columns:
        raise SystemExit(f"Colonne date introuvable. Colonnes du CSV: {list(df.columns)}")

    # parse date
    df[args.datecol] = pd.to_datetime(df[args.datecol], errors="coerce")
    if df[args.datecol].isna().all():
        raise SystemExit("Impossible de parser la colonne date. Vérifiez le format.")

    missing = REQUIRED_BANDS - set(df.columns)
    if missing:
        raise SystemExit(f"Colonnes de bandes manquantes dans le CSV: {sorted(list(missing))}")

    # calcul indices
    indices = compute_indices(df)
    indices[args.datecol] = df[args.datecol]

    # grouper par date et calculer mean/std (au jour)
    grouped = indices.groupby(args.datecol).agg(["mean", "std"])
    # grouped is MultiIndex columns: (index, mean/std)
    # convertir à format plus simple
    dates = grouped.index
    summary = pd.DataFrame(index=dates)
    for col in indices.columns.drop(args.datecol):
        summary[col + "_mean"] = grouped[(col, "mean")]
        summary[col + "_std"] = grouped[(col, "std")]

    # sauvegarder résumé
    summary_csv = outdir / f"{csv_path.stem}_indices_summary.csv"
    summary.to_csv(summary_csv, index=True, date_format="%Y-%m-%d")
    print(f"Résumé sauvegardé: {summary_csv}")

    # tracer chaque indice
    for col in indices.columns.drop(args.datecol):
        mean_ser = summary[col + "_mean"]
        std_ser = summary[col + "_std"]
        # skip if no valid data
        if mean_ser.dropna().empty:
            print(f"Indice {col} sans données valides -> skip")
            continue
        outpng = outdir / f"{csv_path.stem}_{col}.png"
        plot_index_time_series(dates, mean_ser, std_ser, col, outpng, smooth_days=args.smooth)
        print(f"Figure sauvegardée: {outpng}")

    print("Terminé.")


if __name__ == "__main__":
    main()