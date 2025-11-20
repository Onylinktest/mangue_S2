import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import re
import json

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


def _extract_pixel_id(df):
    """Create a stable pixel identifier across dates.
    Prefer extracting the trailing integer from `system:index` (e.g. ..._12).
    If unavailable, fall back to parsing the coordinates from the `.geo` column.
    Returns a Series of string ids.
    """
    if "system:index" in df.columns:
        # try to extract trailing index
        m = df["system:index"].astype(str).str.extract(r"_(\d+)$")[0]
        if m.notna().all():
            return m.astype(int).astype(str)

    # fallback: parse coordinates from .geo (format contains [lon,lat])
    if ".geo" in df.columns:
        def parse_coord(val):
            try:
                s = str(val)
                # find two floats inside brackets
                nums = re.findall(r"(-?\d+\.\d+|-?\d+)", s)
                if len(nums) >= 2:
                    lon = float(nums[0])
                    lat = float(nums[1])
                    return f"{lon:.6f}_{lat:.6f}"
            except Exception:
                pass
            return None

        return df[".geo"].apply(parse_coord)

    # last resort: use row number
    return df.index.astype(str)


def carto_variabilite(df, datecol, metric="NDVI", outdir="results", csv_stem="data"):
    """Compute per-pixel variability (std) for periods August Year N -> March Year N+1.

    Produces a CSV with per-pixel std per seasonal window and a simple scatter map PNG.
    """
    if metric not in df.columns:
        raise ValueError(f"Métrique '{metric}' introuvable dans les colonnes du DataFrame")

    # ensure date column is datetime
    df = df.copy()
    df[datecol] = pd.to_datetime(df[datecol], errors="coerce")
    df = df.dropna(subset=[datecol])

    df["year"] = df[datecol].dt.year
    df["month"] = df[datecol].dt.month

    df["pixel_id"] = _extract_pixel_id(df)

    # parse lon/lat for mapping (first occurrence per pixel)
    coords = {}
    if ".geo" in df.columns:
        for pid, group in df.groupby("pixel_id"):
            val = group[".geo"].iloc[0]
            if pd.isna(val):
                coords[pid] = (None, None)
                continue
            s = str(val)
            nums = re.findall(r"(-?\d+\.\d+|-?\d+)", s)
            if len(nums) >= 2:
                lon = float(nums[0]); lat = float(nums[1])
                coords[pid] = (lon, lat)
            else:
                coords[pid] = (None, None)
    else:
        for pid in df["pixel_id"].unique():
            coords[pid] = (None, None)

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    rows = []
    # for each possible starting year N (we compute N -> N+1 March)
    for N in range(min_year, max_year + 1):
        # select Aug..Dec of N and Jan..Mar of N+1
        mask = ((df["year"] == N) & (df["month"] >= 8)) | ((df["year"] == (N + 1)) & (df["month"] <= 3))
        window = df[mask]
        if window.empty:
            continue
        grouped = window.groupby("pixel_id")[metric].agg(["std", "count"]).rename(columns={"std": f"std_{N}_{N+1}", "count": f"count_{N}_{N+1}"})
        if rows == []:
            outdf = grouped.copy()
        else:
            outdf = outdf.join(grouped, how="outer")

    if 'outdf' not in locals():
        raise SystemExit("Aucune donnée disponible pour les fenêtres Aug->Mar dans les années du jeu de données.")

    outdf = outdf.reset_index().rename(columns={"pixel_id": "pixel_id"})
    # add lon/lat
    outdf["lon"] = outdf["pixel_id"].map(lambda x: coords.get(x, (None, None))[0])
    outdf["lat"] = outdf["pixel_id"].map(lambda x: coords.get(x, (None, None))[1])

    out_csv = Path(outdir) / f"{csv_stem}_pixel_variability_{metric}.csv"
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(out_csv, index=False)

    # Create a simple scatter map for the most recent window computed (last columns)
    # find last std column
    std_cols = [c for c in outdf.columns if str(c).startswith("std_")]
    if std_cols:
        last_std = std_cols[-1]
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6))
            valid = outdf.dropna(subset=["lon", "lat", last_std])
            sc = plt.scatter(valid["lon"].astype(float), valid["lat"].astype(float), c=valid[last_std].astype(float), cmap="viridis", s=10)
            plt.colorbar(sc, label=f"{metric} std ({last_std})")
            plt.title(f"Variabilité pixels {metric} ({last_std})")
            plt.xlabel("lon")
            plt.ylabel("lat")
            png = Path(outdir) / f"{csv_stem}_pixel_variability_{metric}.png"
            plt.tight_layout()
            plt.savefig(png, dpi=150)
            plt.close()
            print(f"Carto sauvegardée: {png}")
        except Exception as exc:
            print(f"Impossible de produire la figure carto: {exc}")

    print(f"CSV carto sauvegardé: {out_csv}")
    return outdf


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
    p.add_argument("--scl-col", default="SCL",
                   help="Nom de la colonne SCL dans le CSV (par défaut: 'SCL'). Si absente, aucun filtrage SCL n'est appliqué.")
    p.add_argument("--scl-values", default="4,5",
                   help="Liste séparée par des virgules des valeurs SCL à conserver (par défaut: '4,5'). Ex: '4,5' or '4' or '4,5,6'")
    p.add_argument("--no-carto", action="store_true", help="Désactiver la génération de carto (la carto est exécutée par défaut)")
    p.add_argument("--metric", default="NDVI", help="Nom de l'indice ou bande pour la variabilité (défauts 'NDVI')")
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

    # Filtrer sur les valeurs SCL demandées (par défaut 4 et 5)
    scl_col = args.scl_col
    if scl_col in df.columns:
        # parser la liste d'entiers fournie
        try:
            scl_values = {int(x.strip()) for x in str(args.scl_values).split(",") if x.strip() != ""}
        except Exception:
            raise SystemExit(f"Argument --scl-values invalide: {args.scl_values}")
        # convertir SCL en numérique si besoin
        df[scl_col] = pd.to_numeric(df[scl_col], errors="coerce")
        before_n = len(df)
        df = df[df[scl_col].isin(scl_values)]
        after_n = len(df)
        if df.empty:
            raise SystemExit(f"Après filtrage SCL ({scl_col} in {sorted(scl_values)}), aucun enregistrement restant.")
        print(f"Filtrage SCL appliqué: colonne '{scl_col}', valeurs {sorted(scl_values)}. Lignes: {before_n} -> {after_n}.")
    else:
        print(f"Colonne SCL '{scl_col}' introuvable dans le CSV — aucun filtrage SCL appliqué.")

    # calcul indices
    indices = compute_indices(df)
    indices[args.datecol] = df[args.datecol]
    # intégrer les indices calculés dans le DataFrame origine (par pixel/observation)
    for c in indices.columns:
        df[c] = indices[c]

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

    # générer cartographie de variabilité par pixel (activée par défaut)
    if not args.no_carto:
        try:
            carto_variabilite(df, args.datecol, metric=args.metric, outdir=outdir, csv_stem=csv_path.stem)
        except Exception as exc:
            print(f"Erreur durant génération carto: {exc}")

    print("Terminé.")


if __name__ == "__main__":
    main()