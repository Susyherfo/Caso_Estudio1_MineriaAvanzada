import requests
import pandas as pd

def cargar_datos_clima():

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=9.93&longitude=-84.08&"
        "hourly=temperature_2m,relativehumidity_2m,windspeed_10m"
    )

    try:
        response = requests.get(url)
        data = response.json()

        df_weather = pd.DataFrame({
            "period": data["hourly"]["time"],
            "temperature": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relativehumidity_2m"],
            "wind_speed": data["hourly"]["windspeed_10m"]
        })

        df_weather["period"] = pd.to_datetime(df_weather["period"])

        return df_weather

    except:
        return None