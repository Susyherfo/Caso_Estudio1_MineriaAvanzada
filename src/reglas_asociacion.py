"""
Módulo de Reglas de Asociación
BCD-7213 – Minería de Datos Avanzada – Caso de Estudio #2

Aplica minería de reglas de asociación sobre el dataset de
consumo energético doméstico.

Flujo:
1. Discretizar variables continuas en categorías (bajo/medio/alto)
2. Construir transacciones por hora del día
3. Encontrar itemsets frecuentes con FP-Growth (mlxtend)
4. Generar reglas con métricas de soporte, confianza y lift
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Discretización
# ─────────────────────────────────────────────

def discretizar_consumo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables continuas en categorías discretas
    para la minería de itemsets.
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    # ── Manejo de fecha/hora ─────────────────────────────
    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"], errors="coerce")
        df = df.set_index("period")

    # Asegurar que el índice sea datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El DataFrame debe tener un índice datetime o columna 'period'")

    # ── Conversión a numérico (CLAVE para evitar errores) ──
    cols_numericas = [
        "global_active_power",
        "voltage",
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3"
    ]

    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminar filas con nulos en variables clave
    df = df.dropna(subset=["global_active_power", "voltage"])

    # ── Consumo global ──────────────────────────────────
    q33 = df["global_active_power"].quantile(0.33)
    q66 = df["global_active_power"].quantile(0.66)

    def cat_consumo(x):
        if x <= q33:
            return "consumo_bajo"
        elif x <= q66:
            return "consumo_medio"
        return "consumo_alto"

    df["cat_consumo"] = df["global_active_power"].apply(cat_consumo)

    # ── Voltaje ─────────────────────────────────────────
    vmed = df["voltage"].median()

    def cat_voltaje(x):
        if x < vmed * 0.99:
            return "voltaje_bajo"
        elif x > vmed * 1.01:
            return "voltaje_alto"
        return "voltaje_normal"

    df["cat_voltaje"] = df["voltage"].apply(cat_voltaje)

    # ── Submedidores (activo si > 0) ─────────────────────
    df["sub1_activo"] = df["sub_metering_1"] > 0
    df["sub2_activo"] = df["sub_metering_2"] > 0
    df["sub3_activo"] = df["sub_metering_3"] > 0

    # ── Franja horaria ───────────────────────────────────
    h = df.index.hour

    def franja(x):
        if 0 <= x < 6:
            return "franja_madrugada"
        elif 6 <= x < 12:
            return "franja_mañana"
        elif 12 <= x < 20:
            return "franja_tarde"
        return "franja_noche"

    df["franja_horaria"] = pd.Series(h, index=df.index).apply(franja)

    # ── Día laboral vs fin de semana ─────────────────────
    df["tipo_dia"] = np.where(
        df.index.dayofweek < 5,
        "dia_laboral",
        "fin_de_semana"
    )

    return df

# ─────────────────────────────────────────────
# 2. Construcción de transacciones
# ─────────────────────────────────────────────

def construir_transacciones(df_disc: pd.DataFrame,
                             sample_n: int = 20000) -> pd.DataFrame:
    """
    Construye la matriz de transacciones (one-hot) necesaria
    para FP-Growth / Apriori.

    Cada registro del dataset se convierte en una transacción
    que contiene los ítems activos en ese instante.
    """
    cols_cat = [
        "cat_consumo", "cat_voltaje",
        "franja_horaria", "tipo_dia"
    ]
    cols_bool = [
        "sub1_activo", "sub2_activo", "sub3_activo"
    ]

    # Muestrear para que FP-Growth no tarde demasiado
    if len(df_disc) > sample_n:
        df_disc = df_disc.sample(n=sample_n, random_state=42)

    # One-hot para columnas categóricas
    df_ohe = pd.get_dummies(df_disc[cols_cat])

    # Booleanos como enteros
    for c in cols_bool:
        label = c.replace("_activo", "")
        df_ohe[f"{label}_activo"]   = df_disc[c].astype(bool)
        df_ohe[f"{label}_inactivo"] = (~df_disc[c]).astype(bool)

    return df_ohe.astype(bool)


# ─────────────────────────────────────────────
# 3. Minería de itemsets y reglas
# ─────────────────────────────────────────────

def minar_reglas(df_transacciones: pd.DataFrame,
                 min_support: float = 0.05,
                 min_confidence: float = 0.60,
                 min_lift: float = 1.0,
                 max_reglas: int = 50) -> pd.DataFrame:
    """
    Aplica FP-Growth para encontrar itemsets frecuentes
    y genera reglas de asociación.

    Parámetros:
    - min_support:    frecuencia mínima del itemset (5 %)
    - min_confidence: confianza mínima de la regla (60 %)
    - min_lift:       lift mínimo para filtrar reglas triviales

    Retorna DataFrame con columnas:
    antecedents, consequents, support, confidence, lift
    """
    try:
        from mlxtend.frequent_patterns import fpgrowth, association_rules
    except ImportError:
        raise ImportError(
            "mlxtend no está instalado. "
            "Ejecuta: pip install mlxtend"
        )

    # Itemsets frecuentes
    itemsets = fpgrowth(
        df_transacciones,
        min_support=min_support,
        use_colnames=True
    )

    if itemsets.empty:
        return pd.DataFrame()

    # Generar reglas
    reglas = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Filtrar por lift
    reglas = reglas[reglas["lift"] >= min_lift]

    # Convertir frozensets a strings legibles
    reglas["antecedents"] = reglas["antecedents"].apply(
        lambda x: " & ".join(sorted(x))
    )
    reglas["consequents"] = reglas["consequents"].apply(
        lambda x: " & ".join(sorted(x))
    )

    # Ordenar por lift descendente
    reglas = reglas.sort_values("lift", ascending=False)

    # Columnas de interés
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    return reglas[cols].head(max_reglas).reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. Pipeline completo
# ─────────────────────────────────────────────

def pipeline_asociacion(df: pd.DataFrame,
                         min_support: float = 0.05,
                         min_confidence: float = 0.60,
                         min_lift: float = 1.0) -> dict:
    """
    Ejecuta el pipeline completo de reglas de asociación:
    discretización → transacciones → FP-Growth → reglas.

    Retorna diccionario con:
    - df_discretizado: DataFrame con columnas categóricas
    - df_transacciones: matriz one-hot
    - reglas: DataFrame con las reglas encontradas
    - stats: estadísticas básicas del dataset de transacciones
    """
    df_disc  = discretizar_consumo(df)
    df_trans = construir_transacciones(df_disc)
    reglas   = minar_reglas(
        df_trans,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift
    )

    stats = {
        "n_transacciones": len(df_trans),
        "n_items": df_trans.shape[1],
        "n_reglas": len(reglas),
        "lift_max": float(reglas["lift"].max()) if not reglas.empty else 0,
        "conf_media": float(reglas["confidence"].mean()) if not reglas.empty else 0,
    }

    return {
        "df_discretizado": df_disc,
        "df_transacciones": df_trans,
        "reglas": reglas,
        "stats": stats
    }