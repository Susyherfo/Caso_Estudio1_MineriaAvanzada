import requests
import pandas as pd

def cargar_datos_eia(api_key):

    url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"

    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[respondent][]": "NYIS",
        "start": "2023-01-01",
        "end": "2023-12-31"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None

    data = response.json()

    registros = []

    for item in data["response"]["data"]:
        registros.append({
            "period": item["period"],
            "eia_demand": float(item["value"])
        })

    df = pd.DataFrame(registros)
    df["period"] = pd.to_datetime(df["period"])

    return df