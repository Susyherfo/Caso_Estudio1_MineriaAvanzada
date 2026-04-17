"""
BCD-7213 – Minería de Datos Avanzada
Caso de Estudio #2 – Dashboard Integrado

Autoras: Susana Herrera Fonseca & Kendra Gutiérrez
Universidad LEAD · I Cuatrimestre 2025

Temas integrados:
  - Web Mining       (content, structure, usage via API EIA)
  - Redes Neuronales (MLP, LSTM, CNN1D, Autoencoder, Regresión)
  - Reglas de Asociación (FP-Growth sobre consumo energético)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

# ── módulos del proyecto ───────────────────────────────────────
from src.preprocesamiento   import load_and_clean_data
from src.clasificacion      import split_data, train_logistic_regression, train_random_forest
from src.k_fold             import aplicar_kfold
from src.series_temporales  import (prepare_time_series, train_test_split_time_series,
                                     run_arima, run_holt_winters)
from src.hiperparametrizacion import ModelEvaluator
from src.redes_neuronales   import (entrenar_mlp, entrenar_lstm, entrenar_cnn1d,
                                     entrenar_autoencoder, entrenar_red_regresion)
from src.reglas_asociacion  import pipeline_asociacion
from src.web_mining         import (ejecutar_content_mining, ejecutar_structure_mining,
                                     cargar_datos_eia)
from sklearn.metrics import r2_score, mean_squared_error

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL DE STREAMLIT
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Energy Mining Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ──────────────────────────────────────────
st.markdown("""
<style>
  /* Fuente base */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* Header principal */
  .main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.8rem;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .main-header h1 {
    color: #e8f4f8;
    font-size: 2rem;
    font-weight: 600;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
  }
  .main-header p {
    color: #7ec8e3;
    font-size: 0.9rem;
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Tarjetas de métricas custom */
  .metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
  }
  .metric-card .val {
    font-size: 2rem;
    font-weight: 600;
    color: #1e3a5f;
    font-family: 'IBM Plex Mono', monospace;
  }
  .metric-card .lbl {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Badge de sección */
  .section-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #7ec8e3;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 0.6rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  /* Separador temático */
  .topic-divider {
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 2rem 0 1.5rem;
  }

  /* Panel de info */
  .info-panel {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #1e40af;
    margin-bottom: 1rem;
  }

  /* Tabla de reglas */
  .stDataFrame { font-size: 0.82rem; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0f1923;
  }
  [data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPERS DE VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════

def plot_history(history: dict, titulo: str):
    """Grafica curvas de entrenamiento (loss y métrica)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("#f8fafc")

    metricas = [k for k in history.keys() if not k.startswith("val_")]
    loss_key = "loss"
    met_key  = [m for m in metricas if m != "loss"]
    met_key  = met_key[0] if met_key else None

    for ax, (key, label) in zip(axes, [
        (loss_key, "Loss (MSE / BCE)"),
        (met_key,  "Métrica")
    ]):
        if key and key in history:
            ax.plot(history[key],      label="Entrenamiento", color="#1e3a5f", lw=2)
            val_key = f"val_{key}"
            if val_key in history:
                ax.plot(history[val_key], label="Validación",
                        color="#7ec8e3", lw=2, linestyle="--")
            ax.set_title(label, fontsize=11, pad=8)
            ax.set_xlabel("Época")
            ax.legend(fontsize=9)
            ax.set_facecolor("#f0f4f8")
            ax.spines[["top","right"]].set_visible(False)
        else:
            ax.axis("off")

    fig.suptitle(titulo, fontsize=13, fontweight="600", y=1.01)
    plt.tight_layout()
    return fig


def plot_prediccion_serie(y_real, y_pred, titulo: str, n_puntos=200):
    """Grafica predicción vs real para modelos de serie temporal."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f0f4f8")

    n = min(n_puntos, len(y_real))
    ax.plot(range(n), y_real[:n],  label="Real",      color="#1e3a5f", lw=1.5)
    ax.plot(range(n), y_pred[:n],  label="Predicción",
            color="#f97316", lw=1.5, linestyle="--")
    ax.set_title(titulo, fontsize=12, fontweight="500")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Consumo (kW)")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig


def tabla_metricas_nn(resultados: list) -> pd.DataFrame:
    """Consolida métricas de todos los modelos de redes neuronales."""
    filas = []
    for r in resultados:
        fila = {"Modelo": r["nombre"]}
        if "accuracy" in r:
            fila["Accuracy"] = f"{r['accuracy']:.4f}"
        if "rmse" in r:
            fila["RMSE"] = f"{r['rmse']:.4f}"
        if "mae" in r:
            fila["MAE"] = f"{r['mae']:.4f}"
        if "r2" in r:
            fila["R²"] = f"{r['r2']:.4f}"
        if "n_anomalias" in r:
            fila["Anomalías"] = r["n_anomalias"]
            fila["% Anomalías"] = f"{r['pct_anomalias']:.1f}%"
        filas.append(fila)
    return pd.DataFrame(filas)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚡ Energy Mining Lab")
    st.markdown("---")
    st.caption("LEAD University · BCD-7213")
    st.caption("Susana Herrera & Kendra Gutiérrez")
    st.markdown("---")

    fuente = st.selectbox(
        "Fuente de datos",
        ["Dataset Local", "Dataset + EIA API"]
    )

    api_key = ""
    if fuente == "Dataset + EIA API":
        api_key = st.text_input("EIA API Key", type="password")

    st.markdown("---")
    menu = st.radio(
        "Módulo",
        [
            "Inicio",
            "Exploración del Dataset",
            "Web Mining",
            "Redes Neuronales",
            "Reglas de Asociación",
            "Series Temporales",
            "Clasificación Clásica",
            "K-Fold Validation",
            "Hiperparametrización",
        ]
    )
    st.markdown("---")
    st.caption("Caso de Estudio #2 · I Cuatrimestre 2025")


# ══════════════════════════════════════════════════════════════
# CARGA DE DATOS (cacheada)
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Cargando dataset…")
def cargar_df():
    df = pd.read_csv("data/energy.csv", sep=";")
    df["period"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df = df.dropna(subset=["period"])
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["hour"]    = df["period"].dt.hour
    df["day"]     = df["period"].dt.day
    df["month"]   = df["period"].dt.month
    df["weekday"] = df["period"].dt.weekday
    return df


df = cargar_df()

if fuente == "Dataset + EIA API" and api_key:
    with st.spinner("Conectando a API EIA…"):
        df_eia = cargar_datos_eia(api_key)
    if df_eia is not None:
        df = pd.merge(df, df_eia, on="period", how="left")
        df["eia_demand"] = df["eia_demand"].ffill()
        st.sidebar.success("✓ EIA conectado")
    else:
        st.sidebar.error("No se pudo conectar a EIA")


# ══════════════════════════════════════════════════════════════
# MÓDULO: INICIO
# ══════════════════════════════════════════════════════════════

if menu == "🏠  Inicio":
    st.markdown("""
    <div class="main-header">
      <h1>⚡ Energy Mining Lab</h1>
      <p>BCD-7213 · Minería de Datos Avanzada · Caso de Estudio #2</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
          <div class="val">{df.shape[0]:,}</div>
          <div class="lbl">Registros</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
          <div class="val">{df['global_active_power'].mean():.2f}</div>
          <div class="lbl">Consumo Medio (kW)</div></div>""", unsafe_allow_html=True)
    with col3:
        n_days = (df["period"].max() - df["period"].min()).days
        st.markdown(f"""<div class="metric-card">
          <div class="val">{n_days:,}</div>
          <div class="lbl">Días de Medición</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
          <div class="val">5</div>
          <div class="lbl">Tipos de Redes Neuronales</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-badge">01 · Web Mining</div>', unsafe_allow_html=True)
        st.markdown("""
        **Extracción y análisis de contenido web** sobre organismos energéticos
        (IEA, EIA). Incluye Content Mining, Structure Mining y Usage Mining
        mediante la API de EIA.
        """)
    with c2:
        st.markdown('<div class="section-badge">02 · Redes Neuronales</div>', unsafe_allow_html=True)
        st.markdown("""
        **Cinco arquitecturas** aplicadas al consumo energético:
        MLP, LSTM, CNN 1D, Autoencoder y Red de Regresión Profunda.
        Cada modelo aborda un aspecto distinto del problema.
        """)
    with c3:
        st.markdown('<div class="section-badge">03 · Reglas de Asociación</div>', unsafe_allow_html=True)
        st.markdown("""
        **FP-Growth** sobre el dataset discretizado. Descubre patrones del
        tipo *"cuando el submedidor 3 está activo en la tarde → consumo alto"*
        con soporte, confianza y lift.
        """)

    st.markdown('<hr class="topic-divider">', unsafe_allow_html=True)

    # Mini gráfica de consumo
    st.subheader("Consumo energético — muestra de 5000 registros")
    sample = df.sample(5000, random_state=42).sort_values("period")
    fig = px.line(sample, x="period", y="global_active_power",
                  color_discrete_sequence=["#1e3a5f"],
                  labels={"global_active_power": "kW", "period": ""})
    fig.update_layout(height=280, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# MÓDULO: EXPLORACIÓN DEL DATASET
# ══════════════════════════════════════════════════════════════

elif menu == "📊  Exploración del Dataset":
    st.header("Exploración del Dataset")
    st.markdown('<div class="section-badge">Individual Household Electric Power Consumption</div>',
                unsafe_allow_html=True)

    rows = st.slider("Registros a mostrar", 5, 50, 10)
    st.dataframe(df.head(rows), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", df.shape[1])
    c3.metric("Período", f"{df['period'].min().date()} → {df['period'].max().date()}")

    st.subheader("Estadísticas descriptivas")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    st.dataframe(df[numeric_cols].describe().round(4), use_container_width=True)

    st.subheader("Análisis visual")
    variable  = st.selectbox("Variable", numeric_cols)
    chart_type = st.radio("Tipo", ["Histograma", "Boxplot", "Tendencia diaria"])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f0f4f8")

    if chart_type == "Histograma":
        sns.histplot(df[variable].dropna(), kde=True, ax=ax, color="#1e3a5f")
    elif chart_type == "Boxplot":
        sns.boxplot(x=df[variable].dropna(), ax=ax, color="#7ec8e3")
    else:
        daily = df.groupby("hour")[variable].mean()
        ax.plot(daily.index, daily.values, color="#1e3a5f", lw=2.5, marker="o", ms=4)
        ax.set_xlabel("Hora del día")
        ax.set_ylabel(f"Promedio {variable}")

    ax.set_title(variable, fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Matriz de correlación")
    corr = df[numeric_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f",
                ax=ax2, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.tight_layout()
    st.pyplot(fig2)


# ══════════════════════════════════════════════════════════════
# MÓDULO: WEB MINING
# ══════════════════════════════════════════════════════════════

elif menu == "🌐  Web Mining":
    st.header("Web Mining")
    st.markdown('<div class="section-badge">Content · Structure · Usage Mining</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
    <strong>¿Qué es Web Mining?</strong><br>
    Web Mining es la aplicación de técnicas de minería de datos sobre el contenido,
    la estructura y el uso de la Web. En este proyecto se aplica sobre fuentes de
    información energética (IEA, EIA) para contextualizar el análisis del dataset
    doméstico.
    </div>
    """, unsafe_allow_html=True)

    subtema = st.tabs([
        "🔍 Content Mining",
        "🕸️ Structure Mining",
        "📡 Usage Mining (EIA API)"
    ])

    # ── 1. Content Mining ──────────────────────────────────────
    with subtema[0]:
        st.subheader("Web Content Mining")
        st.markdown("""
        Extrae y analiza texto de páginas de organismos energéticos internacionales.
        Se mide la frecuencia de keywords energéticos y se calcula un score de
        relevancia (menciones por mil palabras).
        """)

        if st.button("▶ Ejecutar Content Mining", key="btn_content"):
            with st.spinner("Extrayendo y analizando contenido web…"):
                df_content = ejecutar_content_mining()
            st.session_state["content_result"] = df_content

        if "content_result" in st.session_state:
            df_c = st.session_state["content_result"]
            st.dataframe(
                df_c[["fuente","tipo","status","n_palabras",
                       "score_relevancia","total_keywords","titulo"]],
                use_container_width=True
            )

            ok = df_c[df_c["status"] == "ok"]
            if not ok.empty:
                fig = px.bar(
                    ok, x="fuente", y="score_relevancia",
                    color="tipo",
                    color_discrete_sequence=["#1e3a5f", "#7ec8e3", "#f97316"],
                    labels={"score_relevancia": "Score (menciones/1000 palabras)",
                            "fuente": "Fuente"},
                    title="Score de relevancia energética por fuente"
                )
                fig.update_layout(height=350, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

                # Top keywords
                kw_cols = [c for c in ok.columns if c.startswith("kw_")]
                if kw_cols:
                    kw_totals = ok[kw_cols].sum().rename(
                        lambda x: x.replace("kw_", "")
                    ).sort_values(ascending=True)
                    fig2 = px.bar(
                        x=kw_totals.values, y=kw_totals.index,
                        orientation="h",
                        color_discrete_sequence=["#1e3a5f"],
                        labels={"x": "Total menciones", "y": "Keyword"},
                        title="Frecuencia de keywords en todas las fuentes"
                    )
                    fig2.update_layout(height=300)
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Presiona el botón para ejecutar el análisis.")

    # ── 2. Structure Mining ────────────────────────────────────
    with subtema[1]:
        st.subheader("Web Structure Mining")
        st.markdown("""
        Analiza la arquitectura de navegación de sitios energéticos:
        jerarquía de headings, tipos de enlaces, datos estructurados (JSON-LD)
        y densidad de contenido informacional.
        """)

        if st.button("▶ Ejecutar Structure Mining", key="btn_structure"):
            with st.spinner("Analizando estructura web…"):
                df_struct = ejecutar_structure_mining()
            st.session_state["struct_result"] = df_struct

        if "struct_result" in st.session_state:
            df_s = st.session_state["struct_result"]
            st.dataframe(df_s, use_container_width=True)

            ok_s = df_s[df_s["status"] == "ok"]
            if not ok_s.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Internos", x=ok_s["url"].str[-30:],
                    y=ok_s["enlaces_internos"], marker_color="#1e3a5f"
                ))
                fig.add_trace(go.Bar(
                    name="Externos", x=ok_s["url"].str[-30:],
                    y=ok_s["enlaces_externos"], marker_color="#7ec8e3"
                ))
                fig.add_trace(go.Bar(
                    name="Datos", x=ok_s["url"].str[-30:],
                    y=ok_s["enlaces_datos"], marker_color="#f97316"
                ))
                fig.update_layout(
                    barmode="group", height=350,
                    title="Estructura de enlaces por sitio",
                    xaxis_title="Sitio", yaxis_title="Cantidad"
                )
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    tiene_json = ok_s["tiene_json_ld"].sum()
                    st.metric("Sitios con JSON-LD", f"{tiene_json}/{len(ok_s)}")
                with c2:
                    avg_dens = ok_s["densidad_contenido"].mean()
                    st.metric("Densidad de contenido media", f"{avg_dens:.1f}")
        else:
            st.info("Presiona el botón para analizar.")

    # ── 3. Usage Mining (EIA) ──────────────────────────────────
    with subtema[2]:
        st.subheader("Web Usage Mining – API EIA")
        st.markdown("""
        Consume datos de demanda eléctrica en tiempo real de la
        **Energy Information Administration (EIA)** de EE.UU.
        El operador NYISO (Nueva York) provee datos diarios que
        se usan para contextualizar patrones de consumo globales.
        """)

        if api_key:
            if st.button("▶ Cargar datos EIA", key="btn_eia"):
                with st.spinner("Consultando API EIA…"):
                    df_eia = cargar_datos_eia(api_key)
                if df_eia is not None:
                    st.session_state["eia_result"] = df_eia
                    st.success(f"✓ {len(df_eia)} registros cargados de EIA")
                else:
                    st.error("No se pudieron obtener datos de la API.")

            if "eia_result" in st.session_state:
                df_e = st.session_state["eia_result"]
                st.dataframe(df_e.head(20), use_container_width=True)
                fig = px.line(df_e, x="period", y="eia_demand",
                              color_discrete_sequence=["#1e3a5f"],
                              title="Demanda eléctrica diaria NYISO – 2023",
                              labels={"eia_demand": "Demanda (MW)", "period": ""})
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="info-panel">
            Para activar el Usage Mining en tiempo real, ingresa tu EIA API Key
            en el panel lateral. Puedes obtener una llave gratuita en
            <a href="https://www.eia.gov/opendata/" target="_blank">eia.gov/opendata</a>.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **¿Cómo se relaciona con el dataset doméstico?**
            Los patrones de consumo individual capturados en el dataset de
            UCI se pueden contrastar con los datos de demanda agregada de
            la red, revelando qué porción del comportamiento doméstico
            se refleja a nivel de operador de red.
            """)


# ══════════════════════════════════════════════════════════════
# MÓDULO: REDES NEURONALES
# ══════════════════════════════════════════════════════════════

elif menu == "🧠  Redes Neuronales":
    st.header("Redes Neuronales")
    st.markdown('<div class="section-badge">5 arquitecturas · Dataset energético</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
    Se implementan cinco tipos de redes neuronales complementarias sobre el mismo
    dataset de consumo energético. Cada arquitectura aborda un aspecto distinto:
    clasificación, predicción temporal, detección de patrones locales y anomalías.
    </div>
    """, unsafe_allow_html=True)

    red_sel = st.selectbox("Seleccionar red neuronal", [
        "1 · MLP – Clasificación binaria",
        "2 · LSTM – Predicción temporal",
        "3 · CNN 1D – Patrones en ventanas",
        "4 · Autoencoder – Detección de anomalías",
        "5 · Red de Regresión Profunda",
    ])

    epochs = st.slider("Épocas de entrenamiento", 5, 30, 10)

    # ── MLP ────────────────────────────────────────────────────
    if red_sel.startswith("1"):
        st.subheader("MLP – Perceptrón Multicapa")
        st.markdown("""
        **Objetivo:** Clasificar cada instante como consumo alto o bajo.
        Arquitectura densa de dos capas ocultas con Dropout para regularización.
        """)
        st.code("""Input → Dense(128, ReLU) → Dropout(0.3)
      → Dense(64, ReLU)  → Dropout(0.2)
      → Dense(1, Sigmoid)""", language="text")

        if st.button("▶ Entrenar MLP"):
            with st.spinner("Entrenando MLP…"):
                df_clean = load_and_clean_data("data/energy.csv", sample_size=30000)
                res = entrenar_mlp(df_clean, epochs=epochs)
            st.session_state["mlp_res"] = res

        if "mlp_res" in st.session_state:
            r = st.session_state["mlp_res"]
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{r['accuracy']:.4f}")
            c2.metric("Épocas", epochs)

            st.pyplot(plot_history(r["history"], "MLP – Curvas de entrenamiento"))

            rep = pd.DataFrame(r["report"]).T
            st.subheader("Reporte de clasificación")
            st.dataframe(rep.style.format("{:.3f}"), use_container_width=True)

    # ── LSTM ───────────────────────────────────────────────────
    elif red_sel.startswith("2"):
        st.subheader("LSTM – Long Short-Term Memory")
        st.markdown("""
        **Objetivo:** Predecir el consumo de la hora siguiente dado
        un historial de 24 horas. Las celdas LSTM capturan dependencias
        de largo plazo en la serie temporal.
        """)
        st.code("""Input(24h) → LSTM(64) → Dropout(0.2)
         → Dense(32, ReLU) → Dense(1)""", language="text")

        if st.button("▶ Entrenar LSTM"):
            with st.spinner("Entrenando LSTM (puede tardar ~1-2 min)…"):
                df_clean = load_and_clean_data("data/energy.csv", sample_size=30000)
                res = entrenar_lstm(df_clean, epochs=epochs)
            st.session_state["lstm_res"] = res

        if "lstm_res" in st.session_state:
            r = st.session_state["lstm_res"]
            c1, c2 = st.columns(2)
            c1.metric("RMSE", f"{r['rmse']:.4f}")
            c2.metric("MAE",  f"{r['mae']:.4f}")
            st.pyplot(plot_history(r["history"], "LSTM – Loss"))
            st.pyplot(plot_prediccion_serie(r["y_test"], r["y_pred"],
                                             "LSTM – Real vs Predicción"))

    # ── CNN 1D ─────────────────────────────────────────────────
    elif red_sel.startswith("3"):
        st.subheader("CNN 1D – Redes Convolucionales sobre Series")
        st.markdown("""
        **Objetivo:** Clasificar ventanas de 24 horas de consumo como
        período de alto o bajo consumo. Los filtros convolucionales detectan
        patrones locales independientemente de su posición en la ventana.
        """)
        st.code("""Input(24h) → Conv1D(32, k=3) → MaxPool(2)
          → Conv1D(64, k=3) → GlobalAvgPool
          → Dense(32, ReLU) → Dense(1, Sigmoid)""", language="text")

        if st.button("▶ Entrenar CNN 1D"):
            with st.spinner("Entrenando CNN 1D…"):
                df_clean = load_and_clean_data("data/energy.csv", sample_size=30000)
                res = entrenar_cnn1d(df_clean, epochs=epochs)
            st.session_state["cnn_res"] = res

        if "cnn_res" in st.session_state:
            r = st.session_state["cnn_res"]
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{r['accuracy']:.4f}")
            c2.metric("Épocas",   epochs)
            st.pyplot(plot_history(r["history"], "CNN 1D – Curvas de entrenamiento"))
            rep = pd.DataFrame(r["report"]).T
            st.dataframe(rep.style.format("{:.3f}"), use_container_width=True)

    # ── Autoencoder ────────────────────────────────────────────
    elif red_sel.startswith("4"):
        st.subheader("Autoencoder – Detección de Anomalías")
        st.markdown("""
        **Objetivo:** Aprender la distribución normal del consumo y
        detectar instancias atípicas por error de reconstrucción elevado.
        No requiere etiquetas de anomalía; es aprendizaje no supervisado.
        """)
        st.code("""Encoder: Dense(64)→Dense(32)→Dense(16)
Decoder: Dense(32)→Dense(64)→Dense(n_features)""", language="text")

        percentil = st.slider("Percentil umbral de anomalía", 90, 99, 95)

        if st.button("▶ Entrenar Autoencoder"):
            with st.spinner("Entrenando Autoencoder…"):
                df_clean = load_and_clean_data("data/energy.csv", sample_size=30000)
                res = entrenar_autoencoder(df_clean, epochs=epochs,
                                           percentil_umbral=percentil)
            st.session_state["ae_res"] = res

        if "ae_res" in st.session_state:
            r = st.session_state["ae_res"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomalías detectadas", r["n_anomalias"])
            c2.metric("% del test set",       f"{r['pct_anomalias']:.1f}%")
            c3.metric("Umbral (error)",        f"{r['umbral']:.5f}")
            st.pyplot(plot_history(r["history"], "Autoencoder – Loss de reconstrucción"))

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.set_facecolor("#f0f4f8")
            n_show = min(1000, len(r["reconstruction_error"]))
            err    = r["reconstruction_error"][:n_show]
            colors = ["#e74c3c" if a else "#1e3a5f"
                      for a in r["anomalias"][:n_show]]
            ax.scatter(range(n_show), err, c=colors, s=6, alpha=0.7)
            ax.axhline(r["umbral"], color="#f97316", lw=1.5,
                       linestyle="--", label=f"Umbral p{percentil}")
            ax.set_title("Error de reconstrucción por muestra")
            ax.set_xlabel("Muestra")
            ax.set_ylabel("MSE reconstrucción")
            ax.legend(fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

    # ── Regresión Profunda ─────────────────────────────────────
    elif red_sel.startswith("5"):
        st.subheader("Red de Regresión Profunda")
        st.markdown("""
        **Objetivo:** Predecir el valor continuo de `global_active_power`
        a partir de voltaje, intensidad, submedidores y variables temporales.
        Usa Batch Normalization para acelerar la convergencia.
        """)
        st.code("""Input → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
      → Dense(128, ReLU) → BatchNorm → Dropout(0.2)
      → Dense(64, ReLU)  → Dense(1)""", language="text")

        if st.button("▶ Entrenar Red de Regresión"):
            with st.spinner("Entrenando red de regresión…"):
                df_clean = load_and_clean_data("data/energy.csv", sample_size=30000)
                res = entrenar_red_regresion(df_clean, epochs=epochs)
            st.session_state["reg_res"] = res

        if "reg_res" in st.session_state:
            r = st.session_state["reg_res"]
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{r['rmse']:.4f}")
            c2.metric("MAE",  f"{r['mae']:.4f}")
            c3.metric("R²",   f"{r['r2']:.4f}")
            st.pyplot(plot_history(r["history"], "Regresión – Loss"))
            st.pyplot(plot_prediccion_serie(r["y_test"], r["y_pred"],
                                             "Red Profunda – Real vs Predicción"))

    # ── Resumen comparativo ────────────────────────────────────
    st.markdown('<hr class="topic-divider">', unsafe_allow_html=True)
    st.subheader("Resumen comparativo de redes neuronales")

    resumen_data = [
        {"Modelo": "MLP",             "Tipo": "Clasificación",   "Input": "Features tabulares",   "Output": "Binario"},
        {"Modelo": "LSTM",            "Tipo": "Regresión",       "Input": "Ventana 24h",          "Output": "Valor continuo"},
        {"Modelo": "CNN 1D",          "Tipo": "Clasificación",   "Input": "Ventana 24h",          "Output": "Binario"},
        {"Modelo": "Autoencoder",     "Tipo": "No supervisado",  "Input": "Features tabulares",   "Output": "Error reconstrucción"},
        {"Modelo": "Red Regresión",   "Tipo": "Regresión",       "Input": "Features + temporales","Output": "global_active_power"},
    ]
    st.dataframe(pd.DataFrame(resumen_data), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# MÓDULO: REGLAS DE ASOCIACIÓN
# ══════════════════════════════════════════════════════════════

elif menu == "🔗  Reglas de Asociación":
    st.header("Reglas de Asociación")
    st.markdown('<div class="section-badge">FP-Growth · Consumo energético discretizado</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
    Las reglas de asociación descubren co-ocurrencias frecuentes entre condiciones
    de consumo. Se discretizan las variables continuas (bajo/medio/alto) y se
    aplica el algoritmo <strong>FP-Growth</strong> para encontrar patrones del tipo:<br>
    <em>"fin_de_semana & sub3_activo → consumo_alto"</em>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        min_sup  = st.slider("Soporte mínimo",    0.01, 0.30, 0.05, 0.01,
                              help="Fracción de transacciones que contienen el itemset")
    with col2:
        min_conf = st.slider("Confianza mínima",  0.40, 0.95, 0.60, 0.05,
                              help="P(consecuente | antecedente)")
    with col3:
        min_lift = st.slider("Lift mínimo",       1.0,  5.0,  1.0,  0.1,
                              help="Cuántas veces más probable que por azar")

    if st.button("▶ Ejecutar FP-Growth"):
        with st.spinner("Discretizando y minando reglas de asociación…"):
            resultado = pipeline_asociacion(df, min_sup, min_conf, min_lift)
        st.session_state["asoc_res"] = resultado

    if "asoc_res" in st.session_state:
        res = st.session_state["asoc_res"]
        stats  = res["stats"]
        reglas = res["reglas"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transacciones",    f"{stats['n_transacciones']:,}")
        c2.metric("Ítems únicos",     stats["n_items"])
        c3.metric("Reglas generadas", stats["n_reglas"])
        c4.metric("Lift máximo",      f"{stats['lift_max']:.2f}")

        if not reglas.empty:
            st.subheader("Reglas encontradas")
            st.dataframe(
                reglas.style.background_gradient(subset=["lift"], cmap="Blues")
                            .format({"support": "{:.3f}",
                                     "confidence": "{:.3f}",
                                     "lift": "{:.3f}"}),
                use_container_width=True
            )

            # Gráfica lift vs confidence
            fig = px.scatter(
                reglas.head(30), x="confidence", y="lift",
                size="support",
                hover_data=["antecedents", "consequents"],
                color="lift",
                color_continuous_scale="Blues",
                title="Reglas: Confianza vs Lift (tamaño = soporte)",
                labels={"confidence": "Confianza", "lift": "Lift"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Top 10 reglas por lift
            top10 = reglas.head(10).copy()
            top10["regla"] = top10["antecedents"] + " → " + top10["consequents"]
            fig2 = px.bar(
                top10, x="lift", y="regla", orientation="h",
                color="confidence",
                color_continuous_scale="Blues",
                title="Top 10 reglas por lift",
                labels={"lift": "Lift", "regla": ""}
            )
            fig2.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No se encontraron reglas con los umbrales actuales. "
                       "Prueba reducir el soporte o la confianza mínima.")
    else:
        st.info("Ajusta los parámetros y presiona el botón para ejecutar.")


# ══════════════════════════════════════════════════════════════
# MÓDULO: SERIES TEMPORALES (heredado + mejorado)
# ══════════════════════════════════════════════════════════════

elif menu == "📈  Series Temporales":
    st.header("Series Temporales")
    st.markdown('<div class="section-badge">ARIMA · Holt-Winters · Benchmarking</div>',
                unsafe_allow_html=True)

    @st.cache_data(show_spinner="Preparando serie temporal…")
    def _get_series():
        df_clean = load_and_clean_data("data/energy.csv", sample_size=50000)
        return prepare_time_series(df_clean)

    series = _get_series()
    train, test = train_test_split_time_series(series)

    model_choice = st.selectbox("Modelo", ["ARIMA", "Holt-Winters", "Comparar ambos"])

    if model_choice in ["ARIMA", "Comparar ambos"]:
        if st.button("▶ Ejecutar ARIMA") or model_choice == "Comparar ambos":
            with st.spinner("Ajustando ARIMA…"):
                _, fc_arima, rmse_a, mae_a = run_arima(train, test)
            st.session_state.update({"fc_arima": fc_arima, "rmse_a": rmse_a, "mae_a": mae_a})

    if model_choice in ["Holt-Winters", "Comparar ambos"]:
        if st.button("▶ Ejecutar Holt-Winters") or model_choice == "Comparar ambos":
            with st.spinner("Ajustando Holt-Winters…"):
                _, fc_hw, rmse_h, mae_h = run_holt_winters(train, test)
            st.session_state.update({"fc_hw": fc_hw, "rmse_h": rmse_h, "mae_h": mae_h})

    if "fc_arima" in st.session_state or "fc_hw" in st.session_state:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.set_facecolor("#f0f4f8")
        ax.plot(train[-60:], label="Entrenamiento", color="#94a3b8", lw=1.5)
        ax.plot(test,        label="Real",          color="#1e3a5f", lw=2)
        if "fc_arima" in st.session_state:
            ax.plot(test.index, st.session_state["fc_arima"],
                    label=f"ARIMA (RMSE={st.session_state['rmse_a']:.3f})",
                    color="#f97316", lw=1.8, linestyle="--")
        if "fc_hw" in st.session_state:
            ax.plot(test.index, st.session_state["fc_hw"],
                    label=f"Holt-Winters (RMSE={st.session_state['rmse_h']:.3f})",
                    color="#7ec8e3", lw=1.8, linestyle=":")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title("Predicción de consumo diario promedio", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        comp_rows = []
        if "rmse_a" in st.session_state:
            comp_rows.append({"Modelo": "ARIMA",
                              "RMSE": st.session_state["rmse_a"],
                              "MAE":  st.session_state["mae_a"]})
        if "rmse_h" in st.session_state:
            comp_rows.append({"Modelo": "Holt-Winters",
                              "RMSE": st.session_state["rmse_h"],
                              "MAE":  st.session_state["mae_h"]})
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows).set_index("Modelo").round(4),
                         use_container_width=True)


# ══════════════════════════════════════════════════════════════
# MÓDULO: CLASIFICACIÓN CLÁSICA
# ══════════════════════════════════════════════════════════════

elif menu == "⚙️  Clasificación Clásica":
    st.header("Clasificación Clásica")
    st.markdown('<div class="section-badge">Logistic Regression · Random Forest</div>',
                unsafe_allow_html=True)

    @st.cache_data(show_spinner="Entrenando modelos…")
    def _train_classifiers():
        df_clean = load_and_clean_data("data/energy.csv", sample_size=50000)
        X_train, X_test, y_train, y_test = split_data(df_clean)
        log_m = train_logistic_regression(X_train, y_train)
        rf_m  = train_random_forest(X_train, y_train)
        return log_m, rf_m, X_test, y_test

    log_model, rf_model, X_test, y_test = _train_classifiers()

    c1, c2 = st.columns(2)
    c1.metric("Logistic Regression – Accuracy",
              f"{log_model.score(X_test, y_test):.4f}")
    c2.metric("Random Forest – Accuracy",
              f"{rf_model.score(X_test, y_test):.4f}")

    y_prob_log = log_model.predict_proba(X_test)[:,1]
    y_prob_rf  = rf_model.predict_proba(X_test)[:,1]

    fpr_l, tpr_l, _ = roc_curve(y_test, y_prob_log)
    fpr_r, tpr_r, _ = roc_curve(y_test, y_prob_rf)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_l, y=tpr_l, name=f"LR AUC={auc(fpr_l,tpr_l):.3f}",
                              line=dict(color="#1e3a5f", width=2)))
    fig.add_trace(go.Scatter(x=fpr_r, y=tpr_r, name=f"RF AUC={auc(fpr_r,tpr_r):.3f}",
                              line=dict(color="#f97316", width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Aleatorio",
                              line=dict(color="#94a3b8", dash="dash")))
    fig.update_layout(
        title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR", height=380
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# MÓDULO: K-FOLD
# ══════════════════════════════════════════════════════════════

elif menu == "🔀  K-Fold Validation":
    st.header("Validación Cruzada Estratificada K-Fold")

    @st.cache_data(show_spinner="Ejecutando K-Fold…")
    def _run_kfold():
        df_clean = load_and_clean_data("data/energy.csv", sample_size=30000)
        X_train, X_test, y_train, y_test = split_data(df_clean)
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        return aplicar_kfold(X, y)

    log_auc_m, log_auc_s, rf_auc_m, rf_auc_s = _run_kfold()

    c1, c2 = st.columns(2)
    c1.metric("Logistic Regression – AUC medio", f"{log_auc_m:.4f}",
              f"± {log_auc_s:.4f}")
    c2.metric("Random Forest – AUC medio",       f"{rf_auc_m:.4f}",
              f"± {rf_auc_s:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Logistic Regression",
                          x=["AUC medio"], y=[log_auc_m],
                          error_y=dict(type="data", array=[log_auc_s]),
                          marker_color="#1e3a5f"))
    fig.add_trace(go.Bar(name="Random Forest",
                          x=["AUC medio"], y=[rf_auc_m],
                          error_y=dict(type="data", array=[rf_auc_s]),
                          marker_color="#7ec8e3"))
    fig.update_layout(title="AUC K-Fold (5 folds)", height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.info("La validación estratificada garantiza que cada fold mantiene "
            "la proporción original de clases, dando una estimación más "
            "robusta del desempeño real del modelo.")


# ══════════════════════════════════════════════════════════════
# MÓDULO: HIPERPARAMETRIZACIÓN
# ══════════════════════════════════════════════════════════════

elif menu == "🎛️  Hiperparametrización":
    st.header("Hiperparametrización")
    st.markdown('<div class="section-badge">Búsqueda Genética · Grid Search exhaustivo</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
    Compara dos estrategias de búsqueda de hiperparámetros sobre siete modelos de
    regresión: búsqueda genética (GA) y búsqueda exhaustiva (Grid Search).
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ Este módulo puede tardar varios minutos según el hardware.")

    if st.button("▶ Ejecutar Hiperparametrización"):
        df_clean = load_and_clean_data("data/energy.csv", sample_size=20000)
        X_train, X_test, y_train, y_test = split_data(df_clean)
        evaluator = ModelEvaluator(X_train, X_test, y_train, y_test)

        with st.spinner("Búsqueda genética…"):
            genetic_results = evaluator.genetic_search()
        st.session_state["genetic_results"] = genetic_results
        st.session_state["hp_Xtest"]  = X_test
        st.session_state["hp_ytest"]  = y_test

    if "genetic_results" in st.session_state:
        gr = st.session_state["genetic_results"]
        X_test_hp = st.session_state["hp_Xtest"]
        y_test_hp = st.session_state["hp_ytest"]

        rows = []
        for name, res in gr.items():
            try:
                y_pred = res["estimator"].predict(X_test_hp)
                r2   = r2_score(y_test_hp, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test_hp, y_pred))
                rows.append({"Modelo": name, "R²": round(r2, 4), "RMSE": round(rmse, 4)})
            except Exception:
                pass

        if rows:
            df_res = pd.DataFrame(rows).sort_values("R²", ascending=False)
            st.dataframe(df_res, use_container_width=True)

            fig = px.bar(df_res, x="Modelo", y="R²",
                         color="R²", color_continuous_scale="Blues",
                         title="R² por modelo (búsqueda genética)")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Mejores hiperparámetros (genético)")
        for name, res in gr.items():
            with st.expander(name):
                st.json(res["best_params"])