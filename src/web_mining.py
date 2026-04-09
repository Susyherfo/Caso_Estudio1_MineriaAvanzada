"""
Módulo de Web Mining
BCD-7213 – Minería de Datos Avanzada – Caso de Estudio #2

Implementa tres subáreas de Web Mining aplicadas al contexto
de consumo energético:

1. Web Content Mining  – Extrae y analiza noticias/artículos
                         sobre eficiencia energética
2. Web Usage Mining    – Consume la API de EIA para obtener
                         datos de demanda eléctrica real
3. Web Structure Mining – Analiza estructura de páginas de
                          organismos energéticos

Todos los resultados se contextualizan con el dataset de
consumo doméstico para crear un análisis integrado.
"""

import requests
import pandas as pd
import numpy as np
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Web Content Mining
# ─────────────────────────────────────────────

FUENTES_ENERGIA = [
    {
        "nombre": "IEA – Electricity",
        "url": "https://www.iea.org/topics/electricity",
        "tipo": "organismo_internacional"
    },
    {
        "nombre": "EIA – Residential Energy",
        "url": "https://www.eia.gov/energyexplained/use-of-energy/homes.php",
        "tipo": "agencia_gubernamental"
    },
    {
        "nombre": "EIA – Electricity Explained",
        "url": "https://www.eia.gov/energyexplained/electricity/",
        "tipo": "agencia_gubernamental"
    },
]

KEYWORDS_ENERGIA = [
    "consumption", "efficiency", "demand", "power", "electricity",
    "energy", "renewable", "household", "voltage", "metering",
    "smart grid", "peak", "appliance", "conservation", "load"
]


def extraer_texto_pagina(url: str, timeout: int = 8) -> dict:
    """
    Descarga y extrae texto plano de una página web.
    Retorna título, texto limpio y metadatos básicos.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; EnergyMiner/1.0; "
            "Academic Research Project)"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        return {
            "url": url,
            "status": "error",
            "error": str(e),
            "texto": "",
            "n_palabras": 0
        }

    html = resp.text

    # Extraer título
    titulo_match = re.search(r"<title[^>]*>(.*?)</title>", html,
                              re.IGNORECASE | re.DOTALL)
    titulo = titulo_match.group(1).strip() if titulo_match else "Sin título"
    titulo = re.sub(r"\s+", " ", titulo)

    # Remover scripts, estilos, comentarios y etiquetas HTML
    texto = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
    texto = re.sub(r"<script[^>]*>.*?</script>", " ", texto,
                   flags=re.IGNORECASE | re.DOTALL)
    texto = re.sub(r"<style[^>]*>.*?</style>", " ", texto,
                   flags=re.IGNORECASE | re.DOTALL)
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"&[a-z]+;", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    palabras = texto.lower().split()

    return {
        "url": url,
        "status": "ok",
        "titulo": titulo,
        "texto": texto[:5000],          # primeros 5k caracteres
        "n_palabras": len(palabras),
        "palabras": palabras
    }


def analizar_frecuencia_keywords(resultado: dict,
                                  keywords: list = None) -> dict:
    """
    Cuenta la frecuencia de keywords energéticos en el texto
    extraído y calcula un score de relevancia.
    """
    if keywords is None:
        keywords = KEYWORDS_ENERGIA

    palabras = resultado.get("palabras", [])
    if not palabras:
        return {"keywords": {}, "score_relevancia": 0}

    conteos = {}
    for kw in keywords:
        kw_lower = kw.lower()
        conteos[kw_lower] = sum(
            1 for p in palabras if kw_lower in p
        )

    total_kw = sum(conteos.values())
    score    = total_kw / max(len(palabras), 1) * 1000  # por mil palabras

    return {
        "keywords": conteos,
        "total_menciones": total_kw,
        "score_relevancia": round(score, 2)
    }


def ejecutar_content_mining(fuentes: list = None) -> pd.DataFrame:
    """
    Ejecuta Web Content Mining sobre las fuentes definidas.
    Retorna DataFrame con análisis de cada página.
    """
    if fuentes is None:
        fuentes = FUENTES_ENERGIA

    resultados = []

    for fuente in fuentes:
        raw      = extraer_texto_pagina(fuente["url"])
        analisis = analizar_frecuencia_keywords(raw)

        fila = {
            "fuente":          fuente["nombre"],
            "tipo":            fuente["tipo"],
            "url":             fuente["url"],
            "status":          raw["status"],
            "n_palabras":      raw.get("n_palabras", 0),
            "score_relevancia": analisis["score_relevancia"],
            "total_keywords":  analisis.get("total_menciones", 0),
            "titulo":          raw.get("titulo", "N/A"),
        }

        # Agregar conteo individual de keywords más relevantes
        top_kw = ["energy", "consumption", "electricity",
                  "demand", "efficiency"]
        for kw in top_kw:
            fila[f"kw_{kw}"] = analisis["keywords"].get(kw, 0)

        resultados.append(fila)

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────
# 2. Web Usage Mining – API EIA
# ─────────────────────────────────────────────

def cargar_datos_eia(api_key: str) -> pd.DataFrame | None:
    """
    Consume la API de EIA para obtener datos de demanda
    eléctrica del operador NYISO (Nueva York).

    Esto representa Web Usage Mining: análisis de datos
    publicados por sistemas de información web.
    """
    url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
    params = {
        "api_key":              api_key,
        "frequency":            "daily",
        "data[0]":              "value",
        "facets[respondent][]": "NYIS",
        "start":                "2023-01-01",
        "end":                  "2023-12-31"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
    except Exception:
        return None

    data = response.json()
    registros = []

    for item in data.get("response", {}).get("data", []):
        try:
            registros.append({
                "period":     item["period"],
                "eia_demand": float(item["value"])
            })
        except (KeyError, ValueError):
            continue

    if not registros:
        return None

    df = pd.DataFrame(registros)
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values("period").reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. Web Structure Mining
# ─────────────────────────────────────────────

def analizar_estructura_web(url: str, timeout: int = 8) -> dict:
    """
    Analiza la estructura de una página web:
    - Número y tipos de enlaces (internos / externos)
    - Profundidad de la jerarquía de headings
    - Densidad de contenido informacional
    - Presencia de datos estructurados

    Este análisis refleja el subárea de Web Structure Mining,
    que estudia la organización y navegabilidad de sitios web.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; StructureMiner/1.0; "
            "Academic Research Project)"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        return {"url": url, "status": "error", "error": str(e)}

    html  = resp.text
    dominio = re.search(r"https?://([^/]+)", url)
    dominio = dominio.group(1) if dominio else ""

    # ── Conteo de headings ──────────────────────────────
    headings = {}
    for nivel in range(1, 7):
        matches = re.findall(
            rf"<h{nivel}[^>]*>(.*?)</h{nivel}>",
            html, re.IGNORECASE | re.DOTALL
        )
        headings[f"h{nivel}"] = len(matches)

    # ── Análisis de enlaces ──────────────────────────────
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    enlaces_internos  = [h for h in hrefs if dominio in h or h.startswith("/")]
    enlaces_externos  = [h for h in hrefs if h.startswith("http")
                         and dominio not in h]
    enlaces_datos     = [h for h in hrefs
                         if any(h.endswith(ext)
                                for ext in [".csv", ".json", ".xml",
                                            ".xlsx", ".zip"])]

    # ── Datos estructurados ─────────────────────────────
    tiene_json_ld   = bool(re.search(
        r'<script[^>]+type=["\']application/ld\+json', html,
        re.IGNORECASE
    ))
    tiene_open_graph = bool(re.search(
        r'<meta[^>]+property=["\']og:', html, re.IGNORECASE
    ))
    tiene_tablas     = len(re.findall(r"<table", html, re.IGNORECASE))

    # ── Texto limpio para densidad ───────────────────────
    texto = re.sub(r"<[^>]+>", " ", html)
    texto = re.sub(r"\s+", " ", texto).strip()
    n_palabras = len(texto.split())

    return {
        "url":                url,
        "status":             "ok",
        "dominio":            dominio,
        "headings":           headings,
        "total_headings":     sum(headings.values()),
        "n_enlaces_internos": len(enlaces_internos),
        "n_enlaces_externos": len(enlaces_externos),
        "n_enlaces_datos":    len(enlaces_datos),
        "tiene_json_ld":      tiene_json_ld,
        "tiene_open_graph":   tiene_open_graph,
        "n_tablas":           tiene_tablas,
        "n_palabras":         n_palabras,
        "densidad_contenido": round(n_palabras / max(len(html) / 1000, 1), 2),
    }


def ejecutar_structure_mining(urls: list = None) -> pd.DataFrame:
    """
    Ejecuta Web Structure Mining sobre varias páginas.
    """
    if urls is None:
        urls = [f["url"] for f in FUENTES_ENERGIA]

    filas = []
    for url in urls:
        est = analizar_estructura_web(url)
        filas.append({
            "url":               est.get("url", url),
            "status":            est.get("status", "error"),
            "total_headings":    est.get("total_headings", 0),
            "enlaces_internos":  est.get("n_enlaces_internos", 0),
            "enlaces_externos":  est.get("n_enlaces_externos", 0),
            "enlaces_datos":     est.get("n_enlaces_datos", 0),
            "tablas":            est.get("n_tablas", 0),
            "tiene_json_ld":     est.get("tiene_json_ld", False),
            "n_palabras":        est.get("n_palabras", 0),
            "densidad_contenido": est.get("densidad_contenido", 0),
        })

    return pd.DataFrame(filas)


# ─────────────────────────────────────────────
# 4. Pipeline completo de Web Mining
# ─────────────────────────────────────────────

def pipeline_web_mining(api_key: str = None) -> dict:
    """
    Ejecuta los tres tipos de Web Mining y retorna
    los resultados consolidados.
    """
    resultado = {}

    # Content Mining
    resultado["content"] = ejecutar_content_mining()

    # Structure Mining
    resultado["structure"] = ejecutar_structure_mining()

    # Usage Mining (API EIA – opcional)
    if api_key:
        resultado["usage_eia"] = cargar_datos_eia(api_key)
    else:
        resultado["usage_eia"] = None

    return resultado