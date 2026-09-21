"""Front deportivo minimalista para el Analista de Apuestas (Streamlit).

Reutiliza la lógica de ChatApuestas.py (RAG + agente).

Ejecutar:
    streamlit run app_apuestas.py
"""

import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from ChatApuestas import (
    load_config, Recuperador, construir_llms, responder_stream, ENV_FILE, PROJECT_ROOT,
)
from apuestas import simulador

CONFIG_PATH = PROJECT_ROOT / "config_apuestas.json"

st.set_page_config(page_title="Analista de Apuestas", page_icon="⚽", layout="centered")

# ------------------------------------------------------------
# ESTILO DEPORTIVO MINIMALISTA
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500&display=swap');

    :root {
        --verde: #21e065;
        --verde-osc: #0d7a37;
        --fondo: #0b1014;
        --panel: #131b21;
        --texto: #e8f0ea;
        --tenue: #7d8b83;
    }
    .stApp { background: var(--fondo); color: var(--texto); }
    #MainMenu, footer { visibility: hidden; }
    /* Recupera el espacio de arriba: oculta el header y sube el contenido. */
    header[data-testid="stHeader"] { height: 0; background: transparent; }
    .block-container { padding-top: 1.4rem !important; padding-bottom: 5.5rem !important; }

    .hero {
        background: linear-gradient(120deg, var(--verde-osc) 0%, #06231a 60%, var(--fondo) 100%);
        border-radius: 14px; padding: 14px 20px; margin-bottom: 8px;
        border: 1px solid rgba(33,224,101,.25);
    }
    .hero h1 {
        font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 1.7rem;
        margin: 0; letter-spacing: .5px; color: #fff; text-transform: uppercase;
    }
    .hero .sub { font-family: 'Inter'; color: #cfe9d8; font-size: .86rem; margin-top: 2px; }
    .hero .tag {
        display:inline-block; margin-top:8px; background: var(--verde); color:#04140a;
        font-family:'Rajdhani'; font-weight:700; font-size:.75rem; letter-spacing:1px;
        padding:2px 11px; border-radius:20px; text-transform:uppercase;
    }

    .pills { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 4px; }
    .pill {
        background: var(--panel); border:1px solid rgba(255,255,255,.06);
        border-left:3px solid var(--verde); border-radius:10px;
        padding:8px 12px; font-family:'Rajdhani'; font-weight:600; font-size:.92rem;
    }
    .pill .vs { color: var(--tenue); font-weight:500; margin:0 6px; }
    .pill .fav { color: var(--verde); font-size:.78rem; display:block; font-weight:600; }

    h3, .seccion {
        font-family:'Rajdhani'; text-transform:uppercase; letter-spacing:1px;
        color: var(--verde); font-weight:700;
    }
    section[data-testid="stSidebar"] { background: #0a0f13; border-right:1px solid rgba(33,224,101,.12); }
    .marca {
        font-family:'Rajdhani'; font-weight:700; font-size:1.15rem; color:#fff;
        text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;
    }
    /* Navegación (secciones) tipo menú en el panel izquierdo */
    section[data-testid="stSidebar"] div[role="radiogroup"] { gap:4px; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding:8px 10px; border-radius:9px; font-family:'Rajdhani'; font-weight:600;
        transition: background .15s;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(33,224,101,.10);
    }

    .stChatMessage { background: var(--panel); border-radius:14px; border:1px solid rgba(255,255,255,.05); }
    .stButton>button {
        background: transparent; color: var(--texto); border:1px solid rgba(33,224,101,.35);
        border-radius:20px; font-family:'Rajdhani'; font-weight:600; font-size:.85rem;
    }
    .stButton>button:hover { border-color: var(--verde); color: var(--verde); }

    /* Pie de página estilo ChatGPT: línea fina fija abajo, y el input se eleva. */
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { bottom: 26px !important; }
    .gpt-footer {
        position: fixed; left:0; right:0; bottom:0; height:26px; z-index:1001;
        display:flex; align-items:center; justify-content:center;
        background: var(--fondo); border-top:1px solid rgba(255,255,255,.06);
        color: var(--tenue); font-size:.7rem; text-align:center; padding:0 12px;
    }

    /* Indicador "pensando" con frases rotativas (animado por CSS en el navegador). */
    .pensando {
        font-family:'Rajdhani'; font-weight:600; font-size:1.02rem;
        color: var(--verde); padding:4px 0; animation: pulso 1.4s ease-in-out infinite;
    }
    .pensando::after {
        content: "⚽ Calentando en la banda…";
        animation: frases 16s linear infinite;
    }
    @keyframes pulso { 0%,100%{opacity:.5;} 50%{opacity:1;} }
    @keyframes frases {
        0%,12%    { content: "⚽ Calentando en la banda…"; }
        14%,26%   { content: "📊 Analizando partidos previos…"; }
        28%,40%   { content: "🧮 Calculando probabilidades implícitas…"; }
        42%,54%   { content: "🔎 Revisando lesiones y bajas…"; }
        56%,68%   { content: "💧 Bebiendo agua…"; }
        70%,82%   { content: "📈 Comparando cuotas de las casas…"; }
        84%,98%   { content: "🧠 Pensando el mejor pick…"; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# CARGA (cacheada) DE MODELOS Y BASE
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos y base de partidos…")
def iniciar():
    cfg = load_config(CONFIG_PATH)
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No se encontró OPENAI_API_KEY en secrets/.env")
        st.stop()
    recuperador = Recuperador(cfg)
    llms = construir_llms(cfg, api_key)
    return cfg, recuperador, llms


try:
    cfg, recuperador, llms = iniciar()
except FileNotFoundError:
    st.error("No existe la base. Ejecuta primero: `python -m apuestas.construye_corpus`")
    st.stop()

sys_inst = cfg.get("system_instruction", "Eres un analista de apuestas responsable.")
liga = cfg["api"].get("league_name", "—")
fecha = recuperador.metas[0].get("fecha", "") if recuperador.metas else ""
es_hoy = fecha == date.today().isoformat()
etiqueta = "● En juego hoy" if es_hoy else "● Última jornada"
sub_txt = "Jornada de hoy" if es_hoy else "Última jornada disponible"

# ------------------------------------------------------------
# SIDEBAR: navegación (secciones) + panel + actualizar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="marca">⚽ Analista</div>', unsafe_allow_html=True)
    seccion = st.radio(
        "Secciones",
        ["💬 Analista", "📅 Jornadas pasadas", "📈 Histórico (bankroll)"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Liga: **{liga}**")
    st.caption(f"Modelo: `{cfg['llm']['model_name']}`")
    st.caption(f"Fichas en base: **{len(recuperador.textos)}**")
    if st.button("🔄 Actualizar partidos", use_container_width=True):
        from apuestas.construye_corpus import build_base
        with st.spinner("Consultando la API y reconstruyendo la base…"):
            fecha_cargada = build_base(cfg)
        if fecha_cargada:
            st.cache_resource.clear()
            st.session_state.pop("mensajes", None)
            st.session_state["aviso"] = f"Base actualizada con la jornada {fecha_cargada}."
            st.rerun()
        else:
            st.warning("No hay partidos de la liga en la ventana disponible "
                       "(hoy ±1 día). Se mantiene la última jornada cargada.")

_aviso = st.session_state.pop("aviso", None)
if _aviso:
    st.toast(_aviso, icon="✅")

# ------------------------------------------------------------
# HERO (compacto)
# ------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>⚽ Analista de Apuestas</h1>
        <div class="sub">{liga} · {sub_txt} · Mercado 1X2</div>
        <span class="tag">{etiqueta} · {fecha}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SECCIÓN: ANALISTA
# ============================================================
if seccion.startswith("💬"):
    if not es_hoy:
        st.warning(
            f"📅 Hoy ({date.today().isoformat()}) no hay partidos de {liga}. "
            f"Se muestra la última jornada disponible ({fecha}). "
            "Míralas en **📅 Jornadas pasadas** (panel izquierdo)."
        )

    if recuperador.metas:
        pills = ""
        for m in recuperador.metas:
            fav = m.get("favorito")
            fav_txt = {"Home": m.get("home"), "Away": m.get("away"), "Draw": "Empate"}.get(fav, "")
            pills += (
                f'<div class="pill">{m.get("home")}<span class="vs">vs</span>{m.get("away")}'
                f'<span class="fav">★ favorito: {fav_txt}</span></div>'
            )
        st.markdown(f'<div class="pills">{pills}</div>', unsafe_allow_html=True)

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    pregunta = st.chat_input("Pregúntale al analista…")
    if st.session_state.get("pendiente"):
        pregunta = st.session_state.pop("pendiente")

    # El recuadro con scroll propio solo aparece cuando hay conversación; así no
    # se ve un cuadro negro vacío ni sobra scroll al inicio.
    if st.session_state.mensajes or pregunta:
        chat_box = st.container(height=430)
        with chat_box:
            for msg in st.session_state.mensajes:
                avatar = "⚽" if msg["role"] == "assistant" else "🧑"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        if pregunta:
            st.session_state.mensajes.append({"role": "user", "content": pregunta})
            historial = []
            for m in st.session_state.mensajes[:-1][-6:]:
                cls = HumanMessage if m["role"] == "user" else AIMessage
                historial.append(cls(content=m["content"]))

            with chat_box:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(pregunta)
                with st.chat_message("assistant", avatar="⚽"):
                    ph = st.empty()
                    ph.markdown('<div class="pensando"></div>', unsafe_allow_html=True)
                    respuesta = ""
                    try:
                        for trozo in responder_stream(llms, recuperador, sys_inst, pregunta, historial):
                            respuesta += trozo
                            ph.markdown(respuesta)
                        if not respuesta:
                            respuesta = "No obtuve respuesta del modelo. Intenta de nuevo."
                            ph.markdown(respuesta)
                    except Exception as e:  # noqa: BLE001
                        respuesta = f"Ups, el modelo falló (posible rate-limit del free tier): {e}"
                        ph.markdown(respuesta)

            st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
    else:
        st.caption("💬 Escribe tu pregunta abajo o toca una sugerencia para empezar.")

    # Preguntas rápidas DEBAJO del chat (chips)
    st.caption("Preguntas rápidas")
    chips = [
        ("🔥 Mejores apuestas", "¿Cuáles son las mejores apuestas de hoy?"),
        ("🛡️ La más segura", "¿Cuál es la apuesta más segura y por qué?"),
        ("🎲 Pick arriesgado", "Dame un pick arriesgado con buena cuota."),
    ]
    cols = st.columns(len(chips))
    for col, (label, q) in zip(cols, chips):
        if col.button(label, use_container_width=True):
            st.session_state.pendiente = q
            st.rerun()

# ============================================================
# SECCIÓN: JORNADAS PASADAS
# ============================================================
elif seccion.startswith("📅"):
    from apuestas.construye_corpus import cargar_jornadas
    jornadas = cargar_jornadas(cfg)
    if not jornadas:
        st.info("Aún no hay jornadas archivadas. Se guardan cada vez que actualizas "
                "los partidos.")
    else:
        st.caption("Histórico de jornadas cargadas (partidos, favorito del mercado "
                   "y resultado real).")
        for f in sorted(jornadas.keys(), reverse=True):
            partidos = jornadas[f]
            marca = " · hoy" if f == date.today().isoformat() else ""
            with st.expander(f"📅 {f}{marca} · {len(partidos)} partidos",
                             expanded=(f == fecha)):
                filas = []
                for p in partidos:
                    cuotas = p.get("cuotas") or {}
                    fav = p.get("favorito")
                    res = p.get("resultado")
                    nombre = {"Home": p["home"], "Away": p["away"], "Draw": "Empate"}
                    filas.append({
                        "Partido": f"{p['home']} vs {p['away']}",
                        "Favorito": nombre.get(fav, "—"),
                        "Cuota fav": cuotas.get(fav) if fav else None,
                        "Resultado": nombre.get(res, "por jugar" if res is None else "—"),
                        "Fav OK": ("✓" if res == fav else "✗") if res else "—",
                    })
                st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

# ============================================================
# SECCIÓN: HISTÓRICO (BANKROLL)
# ============================================================
else:
    ledger = simulador.cargar_ledger(cfg)
    inicial = ledger["bankroll_inicial"]
    actual = ledger["bankroll"]
    hist = ledger["historial"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Bankroll inicial", f"${inicial:,.0f}")
    m2.metric("Bankroll actual", f"${actual:,.0f}", f"{actual - inicial:+,.0f}")
    m3.metric("Jornadas jugadas", len(hist))
    if ledger.get("quebrado"):
        st.error("💀 El agente quebró: bankroll en 0.")

    if hist:
        puntos = [{"jornada": 0, "bankroll": inicial}]
        for i, e in enumerate(hist, start=1):
            puntos.append({"jornada": i, "bankroll": e["bankroll_final"]})
        st.line_chart(pd.DataFrame(puntos).set_index("jornada")["bankroll"], height=240)
        st.caption("Eje X: número de jornada (0 = inicio con el bankroll base).")

        for e in reversed(hist):
            signo = "🟢" if e["ganancia_total"] >= 0 else "🔴"
            with st.expander(
                f"{signo} {e['fecha']} · {e['bankroll_inicial']:,.0f} → "
                f"{e['bankroll_final']:,.0f}  ({e['ganancia_total']:+,.0f})"
            ):
                filas = [{
                    "Partido": a["partido"],
                    "Pick": a["pick"],
                    "Cuota": a["cuota"],
                    "Apostado": round(a["stake"]),
                    "Resultado": a["resultado_real"],
                    "OK": "✓" if a["acierto"] else "✗",
                    "Ganancia": round(a["ganancia"]),
                } for a in e["apuestas"]]
                st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
    else:
        st.info("Aún no hay jornadas simuladas. Pulsa **Simular jornada de hoy**.")

    st.caption("El agente apuesta entre el 50% y el 100% del bankroll por jornada. "
               "El presupuesto es acumulable hacia adelante.")

    b1, b2 = st.columns(2)
    if b1.button("▶️ Simular jornada de hoy", use_container_width=True):
        api_key = os.getenv("OPENAI_API_KEY")
        with st.spinner("El agente está analizando y apostando…"):
            simulador.simular_dia(cfg, api_key, date.today().isoformat())
        st.rerun()
    if b2.button("♻️ Reiniciar simulación", use_container_width=True):
        simulador.resetear(cfg)
        st.rerun()

# ------------------------------------------------------------
# PIE DE PÁGINA (disclaimer fijo, estilo ChatGPT)
# ------------------------------------------------------------
st.markdown(
    '<div class="gpt-footer">Solo para mayores de 18 años · Apostar implica riesgo real '
    "de pérdida · Ejercicio académico, no asesoría ni garantía de resultados · Juega con "
    "responsabilidad.</div>",
    unsafe_allow_html=True,
)
