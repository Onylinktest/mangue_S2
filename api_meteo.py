import requests
import os

# -------------------------------------------------------------------
# Paramètres à PERSONNALISER
# -------------------------------------------------------------------
LOGIN = "gbrunel"        # <-- remplace par ton login Smartis
API_KEY = "K3fFQPo7SfMyQcC5TVFHNNUOyAi1yc"     # <-- remplace par ta clé API Smartis

# Coordonnées de la parcelle
LON = 55.23416244032644
LAT = -21.062228897606758

# Période (année 2019)
START_DATE = "2019-08-01"
END_DATE = "2025-03-01"

# Format des données : "raw" ou "csv"
DATA_FORMAT = "raw"
# -------------------------------------------------------------------


def fetch_meteo_smartis(lon, lat, start_date, end_date, login, api_key, data_format="raw", debug=False):
    """
    Interroge l'API Smartis WSMeteo et renvoie le texte brut de la réponse.
    """
    base_url = "https://smartis.re/api/WSMeteo"

    params = {
        "long": lon,
        "lat": lat,
        "startdate": start_date,
        "enddate": end_date,
        "login": login,
        # 🔵 Correct : 'apikey' en minuscules
        "apikey": api_key,
        "format": data_format
    }

    response = requests.get(base_url, params=params, timeout=30)

    if debug:
        print("URL appelée :")
        print(response.url)
        print("Code HTTP :", response.status_code)
        print("Texte réponse :")
        print(response.text[:1000])

    if response.status_code == 200:
        return response.text
    elif response.status_code == 400:
        raise RuntimeError("Erreur 400 - Bad Request (paramètres incorrects).")
    elif response.status_code == 401:
        raise RuntimeError("Erreur 401 - Unauthorized (login / apikey incorrect).")
    elif response.status_code == 422:
        raise RuntimeError(f"Erreur 422 - Validation error : {response.text}")
    else:
        raise RuntimeError(f"Erreur HTTP {response.status_code} : {response.text}")


def main():
    data_text = fetch_meteo_smartis(
        lon=LON,
        lat=LAT,
        start_date=START_DATE,
        end_date=END_DATE,
        login=LOGIN,
        api_key=API_KEY,
        data_format=DATA_FORMAT,
        debug=True,
    )

    out_dir = os.path.join("output", "meteo")
    os.makedirs(out_dir, exist_ok=True)

    extension = "csv" if DATA_FORMAT == "csv" else "txt"
    outfile = os.path.join(
        out_dir,
        f"meteo_{START_DATE}_{END_DATE}_{LON:.5f}_{LAT:.5f}.{extension}"
    )

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(data_text)

    print(f"✔ Données météo sauvegardées dans : {outfile}")


if __name__ == "__main__":
    main()