"""
Sintetizador y Analizador Cuantitativo de Distribuciones
========================================================
App Streamlit para un flujo de tres etapas:

  1. GENERACIÓN  -> muestrea datos sintéticos de una distribución paramétrica
                    (con ruido y outliers opcionales para estresar el ajuste).
  2. ANÁLISIS    -> estadística descriptiva completa (momentos, dispersión,
                    forma, percentiles) y diagnósticos visuales.
  3. INFERENCIA  -> ajusta un panel de distribuciones por máxima verosimilitud
                    y las rankea por AIC / BIC + prueba KS de bondad de ajuste.

Diseño: la "verdad" (distribución generadora) se mantiene oculta al motor de
inferencia; así la app funciona como banco de pruebas del propio ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as stats
import plotly.graph_objects as go

st.set_page_config(
    page_title="Analizador de Distribuciones",
    layout="wide",
    page_icon="📊",
)

# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE GENERADORES
#    Cada entrada define: si es discreta, sus parámetros (clave, etiqueta,
#    default, min, max, step) y un sampler que recibe un dict de parámetros,
#    el tamaño n y un generador rng de numpy.
# ---------------------------------------------------------------------------
GENERATORS = {
    "Normal": {
        "discrete": False,
        "params": [
            ("mu", "μ (media)", 0.0, -100.0, 100.0, 0.5),
            ("sigma", "σ (desviación)", 1.0, 0.01, 50.0, 0.1),
        ],
        "sampler": lambda p, n, rng: rng.normal(p["mu"], p["sigma"], n),
    },
    "Log-normal": {
        "discrete": False,
        "params": [
            ("mu", "μ (log-media)", 0.0, -5.0, 5.0, 0.1),
            ("sigma", "σ (log-desv.)", 0.5, 0.01, 3.0, 0.05),
        ],
        "sampler": lambda p, n, rng: rng.lognormal(p["mu"], p["sigma"], n),
    },
    "Exponencial": {
        "discrete": False,
        "params": [("scale", "escala (1/λ)", 1.0, 0.01, 20.0, 0.1)],
        "sampler": lambda p, n, rng: rng.exponential(p["scale"], n),
    },
    "Gamma": {
        "discrete": False,
        "params": [
            ("shape", "forma (k)", 2.0, 0.1, 20.0, 0.1),
            ("scale", "escala (θ)", 1.0, 0.01, 20.0, 0.1),
        ],
        "sampler": lambda p, n, rng: rng.gamma(p["shape"], p["scale"], n),
    },
    "Beta": {
        "discrete": False,
        "params": [
            ("a", "α", 2.0, 0.1, 20.0, 0.1),
            ("b", "β", 5.0, 0.1, 20.0, 0.1),
        ],
        "sampler": lambda p, n, rng: rng.beta(p["a"], p["b"], n),
    },
    "Uniforme": {
        "discrete": False,
        "params": [
            ("low", "mínimo", 0.0, -100.0, 100.0, 0.5),
            ("high", "máximo", 10.0, -100.0, 100.0, 0.5),
        ],
        "sampler": lambda p, n, rng: rng.uniform(p["low"], p["high"], n),
    },
    "Weibull": {
        "discrete": False,
        "params": [
            ("shape", "forma (k)", 1.5, 0.1, 10.0, 0.1),
            ("scale", "escala (λ)", 1.0, 0.01, 20.0, 0.1),
        ],
        "sampler": lambda p, n, rng: p["scale"] * rng.weibull(p["shape"], n),
    },
    "t-Student": {
        "discrete": False,
        "params": [("df", "grados de libertad", 5.0, 1.0, 50.0, 1.0)],
        "sampler": lambda p, n, rng: rng.standard_t(p["df"], n),
    },
    "Mixtura bimodal": {
        "discrete": False,
        "params": [
            ("mu1", "μ₁", -3.0, -20.0, 20.0, 0.5),
            ("mu2", "μ₂", 3.0, -20.0, 20.0, 0.5),
            ("sigma", "σ (común)", 1.0, 0.1, 10.0, 0.1),
            ("w", "peso componente 1", 0.5, 0.0, 1.0, 0.05),
        ],
        "sampler": lambda p, n, rng: np.where(
            rng.random(n) < p["w"],
            rng.normal(p["mu1"], p["sigma"], n),
            rng.normal(p["mu2"], p["sigma"], n),
        ),
    },
    "Poisson (discreta)": {
        "discrete": True,
        "params": [("lam", "λ (tasa)", 3.0, 0.1, 50.0, 0.1)],
        "sampler": lambda p, n, rng: rng.poisson(p["lam"], n).astype(float),
    },
}

# Panel de distribuciones candidatas para la inferencia (todas continuas).
CANDIDATES = [
    "norm", "lognorm", "expon", "gamma", "beta", "weibull_min",
    "uniform", "t", "chi2", "pareto", "rayleigh", "gumbel_r",
]


# ---------------------------------------------------------------------------
# 2. FUNCIONES DE CÁLCULO (cacheadas)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_data(dist_name, param_items, n, seed, noise_sd, n_outliers, outlier_mag):
    """Genera la muestra. param_items es una tupla de (clave, valor) por hashabilidad."""
    rng = np.random.default_rng(seed)
    p = dict(param_items)
    data = np.asarray(GENERATORS[dist_name]["sampler"](p, n, rng), dtype=float)

    if noise_sd > 0:
        data = data + rng.normal(0, noise_sd, n)

    if n_outliers > 0:
        idx = rng.choice(n, size=min(n_outliers, n), replace=False)
        signo = rng.choice([-1.0, 1.0], size=len(idx))
        data[idx] = data[idx] + signo * outlier_mag * np.std(data)

    return data


def describe(data):
    """Estadística descriptiva de primer orden."""
    media = np.mean(data)
    desv = np.std(data, ddof=1)
    return {
        "N": len(data),
        "Media": media,
        "Mediana": np.median(data),
        "Desv. estándar": desv,
        "Varianza": np.var(data, ddof=1),
        "Mínimo": np.min(data),
        "Máximo": np.max(data),
        "Rango": np.ptp(data),
        "IQR": stats.iqr(data),
        "Asimetría (skew)": stats.skew(data),
        "Curtosis (exceso)": stats.kurtosis(data),
        "CV (%)": 100 * desv / media if media != 0 else np.nan,
    }


@st.cache_data(show_spinner=False)
def fit_distributions(data, dist_names):
    """
    Ajusta cada candidata por MLE (dist.fit) y calcula:
      log-verosimilitud, AIC, BIC, estadístico KS y su p-valor.
    Filtra ajustes con log-verosimilitud no finita (soporte incompatible).
    Ranking ascendente por AIC.
    """
    n = len(data)
    filas = []
    for name in dist_names:
        dist = getattr(stats, name)
        try:
            params = dist.fit(data)
            ll = np.sum(dist.logpdf(data, *params))
            if not np.isfinite(ll):
                continue
            k = len(params)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            ks_stat, ks_p = stats.kstest(data, name, args=params)
            filas.append({
                "Distribución": name,
                "log-L": ll,
                "AIC": aic,
                "BIC": bic,
                "KS": ks_stat,
                "p-valor KS": ks_p,
                "params": params,
            })
        except Exception:
            continue

    if not filas:
        return pd.DataFrame()

    df = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    df["ΔAIC"] = df["AIC"] - df["AIC"].min()
    return df


# ---------------------------------------------------------------------------
# 3. SIDEBAR — CONTROLES DE GENERACIÓN
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Generación de datos")

dist_name = st.sidebar.selectbox("Distribución generadora", list(GENERATORS.keys()))
cfg = GENERATORS[dist_name]

st.sidebar.markdown("**Parámetros**")
param_dict = {}
for key, label, default, mn, mx, step in cfg["params"]:
    param_dict[key] = st.sidebar.slider(label, float(mn), float(mx), float(default), float(step))

n = st.sidebar.slider("Tamaño de muestra (N)", 50, 20000, 2000, 50)
seed = st.sidebar.number_input("Semilla (reproducibilidad)", value=42, step=1)

st.sidebar.markdown("**Perturbaciones** (opcionales)")
noise_sd = st.sidebar.slider("Ruido gaussiano (σ)", 0.0, 5.0, 0.0, 0.1)
n_outliers = st.sidebar.slider("Nº de outliers", 0, 100, 0, 1)
outlier_mag = st.sidebar.slider("Magnitud outlier (×σ)", 1.0, 10.0, 3.0, 0.5)

# Validación mínima para Uniforme (low < high)
if dist_name == "Uniforme" and param_dict["low"] >= param_dict["high"]:
    st.sidebar.error("El mínimo debe ser menor que el máximo.")
    st.stop()

data = generate_data(
    dist_name, tuple(param_dict.items()), n, int(seed),
    noise_sd, n_outliers, outlier_mag,
)


# ---------------------------------------------------------------------------
# 4. CUERPO PRINCIPAL
# ---------------------------------------------------------------------------
st.title("📊 Analizador Cuantitativo de Distribuciones")
st.caption(
    "Genera datos sintéticos, caracterízalos estadísticamente e infiere su "
    "distribución subyacente por ajuste de máxima verosimilitud."
)

tab_datos, tab_desc, tab_viz, tab_fit = st.tabs(
    ["🔢 Datos", "📐 Descriptiva", "📈 Visualización", "🎯 Inferencia"]
)

# --- Tab 1: Datos --------------------------------------------------------
with tab_datos:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Muestra generada")
        st.write(f"**Distribución:** {dist_name}")
        st.write(f"**N:** {len(data):,}")
        st.dataframe(
            pd.DataFrame({"valor": data}).head(200),
            use_container_width=True, height=360,
        )
        st.download_button(
            "⬇️ Descargar CSV",
            pd.DataFrame({"valor": data}).to_csv(index=False).encode(),
            file_name="datos_sinteticos.csv",
            mime="text/csv",
        )
    with c2:
        st.subheader("Vista rápida")
        fig = go.Figure(go.Histogram(x=data, nbinsx=60, marker_color="#4C78A8"))
        fig.update_layout(
            height=360, bargap=0.02,
            xaxis_title="valor", yaxis_title="frecuencia",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Descriptiva --------------------------------------------------
with tab_desc:
    st.subheader("Estadística descriptiva")
    stats_dict = describe(data)

    cols = st.columns(4)
    orden = list(stats_dict.items())
    for i, (k, v) in enumerate(orden):
        with cols[i % 4]:
            valor = f"{v:,.4g}" if isinstance(v, (int, float, np.floating)) else str(v)
            st.metric(k, valor)

    st.markdown("---")
    st.subheader("Percentiles")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    perc = pd.DataFrame({
        "Percentil": [f"P{q}" for q in qs],
        "Valor": [np.percentile(data, q) for q in qs],
    })
    st.dataframe(perc, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Lectura de forma")
    skew = stats_dict["Asimetría (skew)"]
    kurt = stats_dict["Curtosis (exceso)"]
    lect_skew = (
        "aproximadamente simétrica" if abs(skew) < 0.5
        else ("sesgo a la derecha (cola larga positiva)" if skew > 0
              else "sesgo a la izquierda (cola larga negativa)")
    )
    lect_kurt = (
        "colas pesadas / leptocúrtica (más outliers que la normal)" if kurt > 0.5
        else ("colas ligeras / platicúrtica" if kurt < -0.5
              else "curtosis cercana a la normal")
    )
    st.info(f"**Asimetría = {skew:.3f}** → {lect_skew}.  \n"
            f"**Curtosis (exceso) = {kurt:.3f}** → {lect_kurt}.")

# --- Tab 3: Visualización ------------------------------------------------
with tab_viz:
    st.subheader("Diagnósticos visuales")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Histograma + densidad (KDE)**")
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=data, nbinsx=60, histnorm="probability density",
            name="datos", marker_color="#4C78A8", opacity=0.7,
        ))
        try:
            kde = stats.gaussian_kde(data)
            xs = np.linspace(np.min(data), np.max(data), 400)
            fig_h.add_trace(go.Scatter(
                x=xs, y=kde(xs), mode="lines", name="KDE",
                line=dict(color="#E45756", width=2),
            ))
        except Exception:
            pass
        fig_h.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="valor", yaxis_title="densidad")
        st.plotly_chart(fig_h, use_container_width=True)

    with col_b:
        st.markdown("**Boxplot (dispersión y outliers)**")
        fig_b = go.Figure(go.Box(y=data, boxpoints="outliers",
                                 marker_color="#54A24B", name=""))
        fig_b.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                            yaxis_title="valor")
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("**Función de distribución acumulada empírica (ECDF)**")
    xs_sorted = np.sort(data)
    ecdf = np.arange(1, len(data) + 1) / len(data)
    fig_e = go.Figure(go.Scatter(x=xs_sorted, y=ecdf, mode="lines",
                                 line=dict(color="#4C78A8", width=2)))
    fig_e.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="valor", yaxis_title="F(x)")
    st.plotly_chart(fig_e, use_container_width=True)

# --- Tab 4: Inferencia ---------------------------------------------------
with tab_fit:
    st.subheader("Inferencia de la distribución subyacente")

    if cfg["discrete"]:
        st.warning(
            "Los datos generados son **discretos** (Poisson). El panel de "
            "candidatas es continuo, por lo que el ajuste es sólo aproximado: "
            "interprétalo como la mejor envolvente continua, no como el modelo real."
        )

    seleccion = st.multiselect(
        "Distribuciones candidatas a evaluar",
        CANDIDATES, default=CANDIDATES,
    )

    if not seleccion:
        st.info("Selecciona al menos una distribución candidata.")
        st.stop()

    with st.spinner("Ajustando por máxima verosimilitud..."):
        ranking = fit_distributions(data, tuple(seleccion))

    if ranking.empty:
        st.error("Ninguna candidata pudo ajustarse a estos datos "
                 "(soporte incompatible). Prueba con otras distribuciones.")
        st.stop()

    st.markdown("**Ranking por AIC** (menor es mejor). "
                "ΔAIC < 2 ≈ soporte equivalente; ΔAIC > 10 ≈ diferencia fuerte.")
    tabla = ranking.drop(columns=["params"]).copy()
    st.dataframe(
        tabla.style.format({
            "log-L": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}",
            "KS": "{:.4f}", "p-valor KS": "{:.4f}", "ΔAIC": "{:.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    mejor = ranking.iloc[0]
    dist_obj = getattr(stats, mejor["Distribución"])
    params = mejor["params"]

    verdad = None
    map_verdad = {
        "Normal": "norm", "Log-normal": "lognorm", "Exponencial": "expon",
        "Gamma": "gamma", "Beta": "beta", "Uniforme": "uniform",
        "Weibull": "weibull_min", "t-Student": "t",
    }
    verdad = map_verdad.get(dist_name)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mejor ajuste (AIC)", mejor["Distribución"])
    c2.metric("Estadístico KS", f"{mejor['KS']:.4f}")
    c3.metric("p-valor KS", f"{mejor['p-valor KS']:.4f}")

    if verdad:
        acierto = "✅ coincide" if verdad == mejor["Distribución"] else "⚠️ difiere"
        st.caption(f"Distribución generadora real: **{verdad}** → {acierto} "
                   f"con el mejor ajuste. (En muestras finitas y con ruido, "
                   f"el ranking puede favorecer una familia vecina.)")

    # Interpretación del p-valor KS
    if mejor["p-valor KS"] < 0.05:
        st.warning(f"KS rechaza el ajuste de **{mejor['Distribución']}** "
                   f"(p < 0.05): aun siendo la mejor candidata, no reproduce "
                   f"bien la muestra. Frecuente con N grande, donde KS detecta "
                   f"desviaciones mínimas.")
    else:
        st.success(f"KS no rechaza **{mejor['Distribución']}** (p ≥ 0.05): "
                   f"ajuste estadísticamente compatible con los datos.")

    # Overlay: histograma + top-3 PDFs
    st.markdown("**Ajuste sobre el histograma (top 3)**")
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Histogram(
        x=data, nbinsx=60, histnorm="probability density",
        name="datos", marker_color="#BAB0AC", opacity=0.6,
    ))
    xs = np.linspace(np.min(data), np.max(data), 400)
    colores = ["#E45756", "#4C78A8", "#54A24B"]
    for i in range(min(3, len(ranking))):
        fila = ranking.iloc[i]
        d = getattr(stats, fila["Distribución"])
        fig_fit.add_trace(go.Scatter(
            x=xs, y=d.pdf(xs, *fila["params"]), mode="lines",
            name=f"{fila['Distribución']} (AIC {fila['AIC']:.0f})",
            line=dict(color=colores[i], width=2),
        ))
    fig_fit.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="valor", yaxis_title="densidad")
    st.plotly_chart(fig_fit, use_container_width=True)

    # Q-Q plot del mejor ajuste
    st.markdown(f"**Q-Q plot vs {mejor['Distribución']}** "
                f"(puntos sobre la diagonal ⇒ buen ajuste)")
    probs = (np.arange(1, len(data) + 1) - 0.5) / len(data)
    teoricos = dist_obj.ppf(probs, *params)
    muestrales = np.sort(data)
    lim = [np.nanmin([teoricos, muestrales]), np.nanmax([teoricos, muestrales])]
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=teoricos, y=muestrales, mode="markers",
                                marker=dict(color="#4C78A8", size=4), name="Q-Q"))
    fig_qq.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                line=dict(color="#E45756", dash="dash"),
                                name="diagonal ideal"))
    fig_qq.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title="cuantiles teóricos",
                         yaxis_title="cuantiles muestrales")
    st.plotly_chart(fig_qq, use_container_width=True)
