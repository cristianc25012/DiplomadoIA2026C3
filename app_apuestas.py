"""Front deportivo minimalista para el Analista de Apuestas (Streamlit).

Reutiliza la lógica de ChatApuestas.py (RAG + agente).

Ejecutar:
    streamlit run app_apuestas.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from ChatApuestas import (
    load_config, Recuperador, construir_llms, responder_stream, ENV_FILE, PROJECT_ROOT,
)

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
    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: linear-gradient(120deg, var(--verde-osc) 0%, #06231a 60%, var(--fondo) 100%);
        border-radius: 18px; padding: 22px 26px; margin-bottom: 8px;
        border: 1px solid rgba(33,224,101,.25);
    }
    .hero h1 {
        font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 2.2rem;
        margin: 0; letter-spacing: .5px; color: #fff; text-transform: uppercase;
    }
    .hero .sub { font-family: 'Inter'; color: #cfe9d8; font-size: .95rem; margin-top: 2px; }
    .hero .tag {
        display:inline-block; margin-top:12px; background: var(--verde); color:#04140a;
        font-family:'Rajdhani'; font-weight:700; font-size:.8rem; letter-spacing:1px;
        padding:3px 12px; border-radius:20px; text-transform:uppercase;
    }

    .pills { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 4px; }
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
    .stChatMessage { background: var(--panel); border-radius:14px; border:1px solid rgba(255,255,255,.05); }
    .stButton>button {
        background: transparent; color: var(--texto); border:1px solid rgba(33,224,101,.35);
        border-radius:10px; font-family:'Rajdhani'; font-weight:600; text-align:left;
    }
    .stButton>button:hover { border-color: var(--verde); color: var(--verde); }
    .disc { color: var(--tenue); font-size:.78rem; line-height:1.4; }

    /* Indicador "pensando" con frases rotativas (animado por CSS en el navegador,
       sigue cambiando aunque el servidor espere la respuesta del modelo). */
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
fecha = recuperador.metas[0].get("fecha", "hoy") if recuperador.metas else "hoy"

# ------------------------------------------------------------
# HERO + PARTIDOS DE HOY
# ------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>⚽ Analista de Apuestas</h1>
        <div class="sub">{liga} · Jornada de hoy · Mercado 1X2</div>
        <span class="tag">● En juego hoy · {fecha}</span>
    </div>
    """,
    unsafe_allow_html=True,
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

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### Panel")
    st.caption(f"Liga: **{liga}**")
    st.caption(f"Modelo: `{cfg['llm']['model_name']}`")
    st.caption(f"Fichas en base: **{len(recuperador.textos)}**")
    st.divider()
    st.markdown("### Preguntas rápidas")
    ejemplos = [
        "¿Cuáles son las mejores apuestas de hoy?",
        "¿Cuál es la apuesta más segura y por qué?",
        "Dame un pick arriesgado con buena cuota.",
    ]
    for e in ejemplos:
        if st.button(e, use_container_width=True):
            st.session_state.pendiente = e
    st.divider()
    if st.button("🔄 Actualizar partidos", use_container_width=True):
        from apuestas.construye_corpus import build_base
        with st.spinner("Consultando la API y reconstruyendo la base…"):
            build_base(cfg)
        st.cache_resource.clear()
        st.session_state.pop("mensajes", None)
        st.rerun()
    st.divider()
    st.markdown(
        '<p class="disc">⚠️ Solo para mayores de 18 años. Apostar implica riesgo '
        "real de pérdida. Esto es un ejercicio académico, no asesoría ni garantía "
        "de resultados. Juega con responsabilidad.</p>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# CHAT
# ------------------------------------------------------------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    avatar = "⚽" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

pregunta = st.chat_input("Pregúntale al analista…")
if st.session_state.get("pendiente"):
    pregunta = st.session_state.pop("pendiente")

if pregunta:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(pregunta)

    # memoria corta: últimas 3 parejas previas -> mensajes LangChain
    historial = []
    for m in st.session_state.mensajes[:-1][-6:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        historial.append(cls(content=m["content"]))

    with st.chat_message("assistant", avatar="⚽"):
        ph = st.empty()
        # Indicador animado mientras el modelo "piensa" (antes del primer token).
        ph.markdown('<div class="pensando"></div>', unsafe_allow_html=True)
        respuesta = ""
        try:
            for trozo in responder_stream(llms, recuperador, sys_inst, pregunta, historial):
                respuesta += trozo
                ph.markdown(respuesta)  # el primer token reemplaza el "pensando…"
            if not respuesta:
                respuesta = "No obtuve respuesta del modelo. Intenta de nuevo."
                ph.markdown(respuesta)
        except Exception as e:  # noqa: BLE001
            respuesta = f"Ups, el modelo falló (posible rate-limit del free tier): {e}"
            ph.markdown(respuesta)

    st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
