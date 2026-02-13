"""
modulo de preprocesamiento
carga, limpia y prepara el dataset de consumo de energia
"""

import pandas as pd


def load_and_clean_data(filepath: str,
                        sample_size: int = 50000,
                        preserve_time_order: bool = True) -> pd.DataFrame:
    """
    carga y limpia el dataset

    parametros:
    - filepath: ruta del archivo
    - sample_size: tamaño máximo del dataset
    - preserve_time_order: mantiene estructura temporal
    """

    # detectar separador automaticamente
    df = pd.read_csv(filepath, sep=None, engine='python')

    # convertir nombres de columnas a minusculas
    df.columns = df.columns.str.lower()

    # reemplazar valores '?' por nulos
    df.replace("?", pd.NA, inplace=True)

    # verificar columnas date y time
    if 'date' in df.columns and 'time' in df.columns:

        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'
        )

        df.drop(columns=['date', 'time'], inplace=True)

    else:
        raise ValueError("no se encontraron columnas 'date' y 'time'")

    # eliminar datetime inválidos
    df.dropna(subset=['datetime'], inplace=True)

    # establecer indice temporal
    df.set_index('datetime', inplace=True)

    # convertir columnas a numerico
    df = df.apply(pd.to_numeric, errors='coerce')

    # eliminar filas con valores nulos
    df.dropna(inplace=True)

    # ordenar cronologicamente
    df.sort_index(inplace=True)

    # control correcto de muestreo
    if sample_size is not None and len(df) > sample_size:

        if preserve_time_order:
            # mantener estructura temporal (para ARIMA y LSTM)
            df = df.iloc[:sample_size]
        else:
            # solo si se quiere muestreo aleatorio
            df = df.sample(n=sample_size, random_state=42)
            df.sort_index(inplace=True)

    return df
