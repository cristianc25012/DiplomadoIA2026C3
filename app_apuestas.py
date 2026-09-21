"""Front del Analista de Apuestas (Streamlit) — diseño portado de Claude Design.

Dashboard deportivo oscuro: sidebar con navegación, top-bar, layout de 2 columnas
en Analista (chat + panel de partidos), KPIs, gráfica de bankroll y tablas.
Reutiliza la lógica de ChatApuestas.py (RAG + agente) y del simulador.

Ejecutar:
    streamlit run app_apuestas.py
"""

import os
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from ChatApuestas import (
    load_config, Recuperador, construir_llms, responder_stream, ENV_FILE, PROJECT_ROOT,
)
from apuestas import simulador

CONFIG_PATH = PROJECT_ROOT / "config_apuestas.json"

st.set_page_config(page_title="Analista de Apuestas", page_icon="⚽", layout="wide")

# ------------------------------------------------------------
# ESTILO (portado del diseño de Claude Design)
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg:#0b0d10; --panel:#101318; --panel2:#0d1014; --line:#1d232b; --line2:#171c22;
        --text:#e8ecf1; --muted:#8a94a3; --muted2:#69727f;
        --lime:#b8ff3c; --lime-soft:#c9ff6e; --pos:#35d07f; --neg:#ff5a4d;
    }
    .stApp { background: var(--bg); color: var(--text);
             font-family:'IBM Plex Sans', system-ui, sans-serif; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stStatusWidget"] { display:none !important; }
    .block-container { padding:0.7rem 1.6rem 1.6rem !important; max-width:100% !important; }
    [data-testid="stMainBlockContainer"] { padding-top:0.7rem !important; }
    h1,h2,h3 { font-family:'Space Grotesk', sans-serif; }

    /* barra superior */
    .topbar { display:flex; align-items:center; gap:18px; padding:11px 18px; margin-bottom:10px;
        border:1px solid var(--line); border-radius:14px;
        background:linear-gradient(90deg, rgba(184,255,60,.06), rgba(11,13,16,0) 60%); }
    .topbar h1 { margin:0; font-size:20px; font-weight:700; color:#f4f7fa; letter-spacing:.01em; }
    .topbar .sub { font-size:12.5px; color:var(--muted); margin-top:2px; }
    .jornada-pill { margin-left:auto; display:inline-flex; align-items:center; gap:7px;
        background:rgba(184,255,60,.1); border:1px solid rgba(184,255,60,.28); color:#c9ff6e;
        border-radius:999px; padding:6px 13px; font-family:'IBM Plex Mono',monospace;
        font-size:11.5px; letter-spacing:.04em; }
    .dot { width:7px; height:7px; border-radius:50%; background:var(--lime);
        box-shadow:0 0 8px var(--lime); animation:pulse 1.8s infinite; }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.35;} }

    /* sidebar */
    section[data-testid="stSidebar"] { background: var(--panel2); border-right:1px solid var(--line); }
    .marca { display:flex; align-items:center; gap:11px; padding:2px 4px 14px;
        font-family:'Space Grotesk'; font-weight:700; letter-spacing:.14em; font-size:15px; color:#f4f7fa; }
    .marca .logo { width:34px; height:34px; border-radius:9px;
        background:linear-gradient(135deg,#b8ff3c,#5fe08a); display:flex; align-items:center;
        justify-content:center; font-size:18px; }
    .info-lbl { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted2); }
    .modelo-chip { font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8fd94f;
        background:#14181d; border:1px solid #22282f; border-radius:6px; padding:5px 7px;
        word-break:break-all; line-height:1.5; display:block; margin-top:3px; }
    .fichas-box { display:flex; align-items:center; justify-content:space-between; background:#14181d;
        border:1px solid #22282f; border-radius:8px; padding:9px 12px; margin-top:12px; }
    .fichas-box b { font-family:'Space Grotesk'; font-size:15px; color:var(--text); }

    /* navegación (radio como menú) */
    section[data-testid="stSidebar"] div[role="radiogroup"] { gap:6px; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        display:flex; align-items:center; gap:8px; padding:10px 12px; border-radius:10px;
        border:1px solid transparent; color:var(--muted); font-weight:500; font-size:13.5px;
        cursor:pointer; transition:all .15s; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:rgba(184,255,60,.06); }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background:rgba(184,255,60,.10); border-color:rgba(184,255,60,.22); color:#eaffc9; font-weight:600; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display:none; }

    /* paneles y tarjetas */
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:16px; }
    .sec-title { font-family:'Space Grotesk'; font-size:13px; font-weight:600; letter-spacing:.06em;
        text-transform:uppercase; color:var(--muted); margin:0 0 10px; }
    .match { background:var(--panel); border:1px solid var(--line); border-radius:13px;
        padding:13px 15px; margin-bottom:10px; }
    .match .row1 { display:flex; align-items:center; justify-content:space-between; gap:8px;
        font-family:'Space Grotesk'; font-weight:600; font-size:13.5px; color:#eef2f6; }
    .match .vs { font-size:10px; color:var(--muted2); letter-spacing:.1em; }
    .match .row2 { display:flex; align-items:center; justify-content:space-between; margin-top:11px; }
    .match .fav { font-size:12px; color:var(--lime-soft); font-weight:500; }
    .match .odd { font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:var(--text);
        background:#171c22; border:1px solid #262c34; border-radius:6px; padding:3px 9px; }

    .kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:16px; }
    .kpi { background:var(--panel); border:1px solid var(--line); border-radius:14px;
        padding:16px 18px; display:flex; flex-direction:column; gap:6px; }
    .kpi .lbl { font-size:11.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted2); }
    .kpi .val { font-family:'Space Grotesk'; font-weight:700; font-size:26px; color:#f4f7fa; line-height:1; }
    .kpi .dl { font-size:12px; font-weight:600; }

    table.dc { width:100%; border-collapse:collapse; font-size:13px; }
    table.dc th { color:var(--muted2); text-align:left; font-size:11px; letter-spacing:.06em;
        text-transform:uppercase; font-weight:500; padding:11px 14px; }
    table.dc td { padding:11px 14px; border-top:1px solid var(--line2); color:#e2e7ee; }
    table.dc td.mono, table.dc th.r { text-align:right; }
    table.dc td.mono { font-family:'IBM Plex Mono',monospace; }
    .jcard { background:var(--panel); border:1px solid var(--line); border-radius:14px;
        overflow:hidden; margin-bottom:14px; }
    .jcard .head { display:flex; align-items:center; gap:12px; padding:14px 16px; border-bottom:1px solid var(--line2); }
    .jcard .head .d { font-family:'Space Grotesk'; font-weight:600; font-size:14.5px; color:#eef2f6; }
    .jcard .head .c { font-size:12.5px; color:var(--muted2); }
    .badge { margin-left:auto; font-size:12px; font-weight:600; }

    /* chips y botones */
    .stButton>button, .stFormSubmitButton>button {
        background:#14181d; color:var(--text); border:1px solid #2a3138; border-radius:999px;
        font-family:'IBM Plex Sans'; font-weight:500; font-size:12.5px; }
    .stButton>button:hover { border-color:var(--lime); color:#eaffc9; }
    button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
        background:var(--lime) !important; color:#0b0d10 !important; border:none !important;
        font-weight:700 !important; }
    button[kind="primary"]:hover { background:var(--lime-soft) !important; }
    .stTextInput input { background:#14181d !important; border:1px solid #262c34 !important;
        color:var(--text) !important; border-radius:12px !important; }
    .stChatMessage { background:transparent; }

    /* pie de página fijo */
    .gpt-footer { position:fixed; left:0; right:0; bottom:0; height:26px; z-index:1001;
        display:flex; align-items:center; justify-content:center; background:var(--bg);
        border-top:1px solid var(--line); color:var(--muted2); font-size:11px; text-align:center; padding:0 12px; }

    /* estados de carga: skeletons + balón de fútbol girando (CSS) */
    @keyframes rot { to { transform:rotate(360deg); } }
    @keyframes shimmer { 0%{background-position:-600px 0;} 100%{background-position:600px 0;} }
    .ball { width:48px; height:48px; border-radius:50%; animation:rot .9s linear infinite;
        background:
          radial-gradient(circle at 50% 30%, #0b0d10 0 6px, transparent 7px),
          radial-gradient(circle at 24% 56%, #0b0d10 0 4.5px, transparent 5.5px),
          radial-gradient(circle at 76% 56%, #0b0d10 0 4.5px, transparent 5.5px),
          radial-gradient(circle at 37% 83%, #0b0d10 0 4px, transparent 5px),
          radial-gradient(circle at 63% 83%, #0b0d10 0 4px, transparent 5px),
          #ffffff;
        box-shadow:0 0 0 2px #2a333d, 0 0 16px rgba(184,255,60,.30); }
    .sk { background:#171d25; border:1px solid #2a333d; border-radius:14px;
        position:relative; overflow:hidden; }
    .sk::after { content:''; position:absolute; inset:0; background-size:600px 100%;
        background-image:linear-gradient(90deg, transparent, rgba(184,255,60,.14), transparent);
        animation:shimmer 1.2s infinite; }

    .pensando { font-family:'Space Grotesk'; font-weight:600; font-size:1rem; color:var(--lime);
        animation:pulso 1.4s ease-in-out infinite; }
    .pensando::after { content:"⚽ Calentando en la banda…"; animation:frases 16s linear infinite; }
    @keyframes pulso { 0%,100%{opacity:.5;} 50%{opacity:1;} }
    @keyframes frases {
        0%,12%{content:"⚽ Calentando en la banda…";} 14%,26%{content:"📊 Analizando partidos previos…";}
        28%,40%{content:"🧮 Calculando probabilidades…";} 42%,54%{content:"🔎 Revisando lesiones…";}
        56%,68%{content:"💧 Bebiendo agua…";} 70%,82%{content:"📈 Comparando cuotas…";}
        84%,98%{content:"🧠 Pensando el mejor pick…";} }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# CARGA (cacheada)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def iniciar():
    cfg = load_config(CONFIG_PATH)
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No se encontró OPENAI_API_KEY en secrets/.env")
    return cfg, Recuperador(cfg), construir_llms(cfg, api_key)


# Estado de carga: el primer render de la sesión muestra el skeleton mientras se
# cargan modelos y base; al terminar se hace rerun y se pinta la UI real. Se evita
# st.empty()+.empty() (que dejaba la pantalla en blanco al refrescar con caché).
SKELETON = """
<div class="sk" style="height:52px;margin-bottom:12px;"></div>
<div style="display:grid;grid-template-columns:1.7fr 1fr;gap:16px;">
  <div class="sk" style="height:330px;display:flex;flex-direction:column;align-items:center;
       justify-content:center;gap:16px;">
    <div class="ball"></div>
    <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:15px;color:#e8ecf1;">
      ⚽ Cargando el analista…</div>
    <div style="font-size:12.5px;color:#8a94a3;">Modelos de IA + base de partidos · unos segundos</div>
  </div>
  <div style="display:flex;flex-direction:column;gap:10px;">
    <div class="sk" style="height:74px;"></div><div class="sk" style="height:74px;"></div>
    <div class="sk" style="height:74px;"></div><div class="sk" style="height:74px;"></div>
  </div>
</div>
"""

if not st.session_state.get("_ready"):
    st.markdown(SKELETON, unsafe_allow_html=True)
    try:
        iniciar()
    except FileNotFoundError:
        st.error("No existe la base. Ejecuta primero: `python -m apuestas.construye_corpus`")
        st.stop()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    st.session_state["_ready"] = True
    st.rerun()

cfg, recuperador, llms = iniciar()

from apuestas.construye_corpus import cargar_jornadas  # noqa: E402

sys_inst = cfg.get("system_instruction", "Eres un analista de apuestas responsable.")
liga = cfg["api"].get("league_name", "—")
fecha = recuperador.metas[0].get("fecha", "") if recuperador.metas else ""
es_hoy = fecha == date.today().isoformat()
NOMBRE = {"Home": "local", "Away": "visitante", "Draw": "Empate"}


def fmt(n):
    return "$" + f"{round(n):,}"


def signed(n):
    return ("+" if n >= 0 else "−") + "$" + f"{abs(round(n)):,}"


def svg_bankroll(vals):
    if len(vals) < 2:
        vals = [vals[0], vals[0]] if vals else [0, 0]
    W, H, padL = 1000, 250, 66
    maxY = (max(vals) * 1.15) or 1
    n = len(vals)
    px = lambda i: padL + (W - padL - 8) * (i / (n - 1))
    py = lambda v: H - 14 - (H - 28) * (v / maxY)
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
    area = (f"M{px(0):.1f},{py(vals[0]):.1f} L" +
            " L".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals)) +
            f" L{px(n-1):.1f},{H-14} L{px(0):.1f},{H-14} Z")
    dots = "".join(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4.5" fill="#0b0d10" '
                   f'stroke="#b8ff3c" stroke-width="2.5"/>' for i, v in enumerate(vals))
    grid = ""
    for frac in (1.0, 0.66, 0.33):
        v = maxY * frac
        y = py(v)
        grid += (f'<line x1="0" x2="1000" y1="{y:.1f}" y2="{y:.1f}" stroke="#1c222a" stroke-width="1"/>'
                 f'<text x="6" y="{y-5:.1f}" fill="#5c6572" font-size="13" '
                 f'font-family="IBM Plex Mono, monospace">{v/1000:.0f}k</text>')
    return (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" style="width:100%;height:190px;">'
            '<defs><linearGradient id="bkfill" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0%" stop-color="#b8ff3c" stop-opacity="0.22"/>'
            '<stop offset="100%" stop-color="#b8ff3c" stop-opacity="0"/></linearGradient></defs>'
            f'{grid}<path d="{area}" fill="url(#bkfill)"/>'
            f'<polyline points="{pts}" fill="none" stroke="#b8ff3c" stroke-width="3" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
            f'{dots}</svg>')


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="marca"><span class="logo">⚽</span>ANALISTA</div>',
                unsafe_allow_html=True)
    seccion = st.radio("Secciones",
                       ["💬  Analista", "🗓  Jornadas pasadas", "📈  Histórico"],
                       label_visibility="collapsed")
    st.divider()
    st.markdown(f'<div class="info-lbl">Liga</div>'
                f'<div style="font-size:13px;font-weight:600;margin-bottom:12px;">{liga}</div>'
                f'<div class="info-lbl">Modelo</div>'
                f'<span class="modelo-chip">{cfg["llm"]["model_name"]}</span>'
                f'<div class="fichas-box"><span style="font-size:12px;color:#8a94a3;">Fichas en base</span>'
                f'<b>{len(recuperador.textos)}</b></div>',
                unsafe_allow_html=True)
    if st.button("↻  Actualizar partidos", use_container_width=True):
        from apuestas.construye_corpus import build_base
        with st.spinner("Consultando la API y reconstruyendo la base…"):
            fecha_cargada = build_base(cfg)
        if fecha_cargada:
            st.cache_resource.clear()
            st.session_state.pop("mensajes", None)
            st.rerun()
        else:
            st.warning("No hay partidos en la ventana disponible (hoy ±1 día).")

# ------------------------------------------------------------
# TOP BAR
# ------------------------------------------------------------
titulos = {"💬": "Analista", "🗓": "Jornadas pasadas", "📈": "Histórico · Bankroll"}
titulo = titulos.get(seccion[:1], "Analista")
st.markdown(
    f'<div class="topbar"><div><h1>{titulo}</h1>'
    f'<div class="sub">{liga} · Mercado 1X2 · '
    f'{"jornada de hoy" if es_hoy else "última jornada disponible"}</div></div>'
    f'<span class="jornada-pill"><span class="dot"></span>JORNADA {fecha}</span></div>',
    unsafe_allow_html=True,
)

# ============================================================
# ANALISTA
# ============================================================
if seccion.startswith("💬"):
    col_chat, col_matches = st.columns([1.7, 1], gap="medium")

    with col_chat:
        if "mensajes" not in st.session_state:
            st.session_state.mensajes = [{
                "role": "assistant",
                "content": (f"Hola. Tengo cargada la jornada del {fecha} de {liga}. "
                            "Pregúntame por las mejores apuestas, el pick más seguro o uno arriesgado."),
            }]

        chat_box = st.container(height=330)
        with chat_box:
            for msg in st.session_state.mensajes:
                with st.chat_message(msg["role"], avatar="⚽" if msg["role"] == "assistant" else "🧑"):
                    st.markdown(msg["content"])

        # chips
        chips = [("🔥 Mejores apuestas", "¿Cuáles son las mejores apuestas de hoy?"),
                 ("🛡 La más segura", "¿Cuál es la apuesta más segura y por qué?"),
                 ("🎲 Pick arriesgado", "Dame un pick arriesgado con buena cuota.")]
        ccols = st.columns(len(chips))
        for c, (lbl, q) in zip(ccols, chips):
            if c.button(lbl, use_container_width=True):
                st.session_state.pendiente = q
                st.rerun()

        # input (form contenido, estilo del diseño)
        with st.form("form_chat", clear_on_submit=True):
            fi = st.columns([1, 0.08])
            texto = fi[0].text_input("m", label_visibility="collapsed",
                                     placeholder="Pregúntale al analista…")
            enviar = fi[1].form_submit_button("↑", type="primary")

        pregunta = texto.strip() if (enviar and texto.strip()) else None
        if st.session_state.get("pendiente"):
            pregunta = st.session_state.pop("pendiente")

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

    with col_matches:
        partidos = cargar_jornadas(cfg).get(fecha, [])
        st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;">'
                    f'<span class="sec-title">Jornada actual</span>'
                    f'<span style="font-size:12px;color:#69727f;">{len(partidos)} partidos</span></div>',
                    unsafe_allow_html=True)
        if not es_hoy:
            st.markdown(f'<div style="font-size:12px;color:#c9ff6e;background:rgba(184,255,60,.07);'
                        f'border:1px solid rgba(184,255,60,.2);border-radius:9px;padding:8px 11px;'
                        f'margin-bottom:10px;">📅 Hoy no hay partidos de {liga}. Mostrando la última '
                        f'jornada ({fecha}).</div>', unsafe_allow_html=True)
        tarjetas = ""
        for p in partidos:
            fav = p.get("favorito")
            fav_nom = {"Home": p["home"], "Away": p["away"], "Draw": "Empate"}.get(fav, "—")
            odd = (p.get("cuotas") or {}).get(fav)
            tarjetas += (f'<div class="match"><div class="row1"><span>{p["home"]}</span>'
                         f'<span class="vs">VS</span><span>{p["away"]}</span></div>'
                         f'<div class="row2"><span class="fav">★ {fav_nom}</span>'
                         f'<span class="odd">{odd:.2f}</span></div></div>') if odd else (
                         f'<div class="match"><div class="row1"><span>{p["home"]}</span>'
                         f'<span class="vs">VS</span><span>{p["away"]}</span></div></div>')
        st.markdown(f'<div>{tarjetas}</div>', unsafe_allow_html=True)

# ============================================================
# JORNADAS PASADAS
# ============================================================
elif seccion.startswith("🗓"):
    jornadas = cargar_jornadas(cfg)
    st.markdown('<p style="font-size:13.5px;color:#8a94a3;margin:0 0 6px;">Histórico de jornadas '
                'cargadas · partido, favorito del mercado y resultado real.</p>', unsafe_allow_html=True)
    if not jornadas:
        st.info("Aún no hay jornadas archivadas. Se guardan al actualizar los partidos.")
    for f in sorted(jornadas.keys(), reverse=True):
        partidos = jornadas[f]
        aciertos = sum(1 for p in partidos if p.get("resultado") and p.get("resultado") == p.get("favorito"))
        con_res = sum(1 for p in partidos if p.get("resultado"))
        filas = ""
        for p in partidos:
            fav = p.get("favorito"); res = p.get("resultado")
            fav_nom = {"Home": p["home"], "Away": p["away"], "Draw": "Empate"}.get(fav, "—")
            res_nom = {"Home": p["home"], "Away": p["away"], "Draw": "Empate"}.get(res, "por jugar")
            odd = (p.get("cuotas") or {}).get(fav)
            ok = "✓" if (res and res == fav) else ("✕" if res else "·")
            okc = "color:#35d07f;" if (res and res == fav) else ("color:#ff5a4d;" if res else "color:#69727f;")
            filas += (f'<tr><td>{p["home"]} vs {p["away"]}</td><td>{fav_nom}</td>'
                      f'<td class="mono">{odd:.2f}</td><td>{res_nom}</td>'
                      f'<td class="mono" style="text-align:center;font-weight:700;{okc}">{ok}</td></tr>'
                      if odd else
                      f'<tr><td>{p["home"]} vs {p["away"]}</td><td>{fav_nom}</td>'
                      f'<td class="mono">—</td><td>{res_nom}</td>'
                      f'<td class="mono" style="text-align:center;{okc}">{ok}</td></tr>')
        st.markdown(
            f'<div class="jcard"><div class="head"><span>🗓</span>'
            f'<span class="d">{f}</span><span class="c">· {len(partidos)} partidos</span>'
            f'<span class="badge" style="color:#c9ff6e;">favorito {aciertos}/{con_res}</span></div>'
            f'<table class="dc"><thead><tr><th>Partido</th><th>Favorito</th>'
            f'<th class="r">Cuota</th><th>Resultado</th><th style="text-align:center;">Fav</th></tr></thead>'
            f'<tbody>{filas}</tbody></table></div>',
            unsafe_allow_html=True)

# ============================================================
# HISTÓRICO · BANKROLL
# ============================================================
else:
    ledger = simulador.cargar_ledger(cfg)
    inicial = ledger["bankroll_inicial"]
    actual = ledger["bankroll"]
    hist = ledger["historial"]
    diff = actual - inicial
    wins = sum(1 for e in hist if e["ganancia_total"] > 0)
    roi = (diff / inicial * 100) if inicial else 0
    pc = "color:#35d07f;" if diff >= 0 else "color:#ff5a4d;"

    st.markdown(
        f'<div class="kpis">'
        f'<div class="kpi"><span class="lbl">Bankroll inicial</span>'
        f'<span class="val">{fmt(inicial)}</span><span class="dl" style="color:#69727f;">base</span></div>'
        f'<div class="kpi"><span class="lbl">Bankroll actual</span>'
        f'<span class="val">{fmt(actual)}</span><span class="dl" style="{pc}">{signed(diff)}</span></div>'
        f'<div class="kpi"><span class="lbl">Jornadas jugadas</span>'
        f'<span class="val">{len(hist)}</span><span class="dl" style="color:#69727f;">{wins} ganadas</span></div>'
        f'<div class="kpi"><span class="lbl">ROI acumulado</span>'
        f'<span class="val">{roi:.1f}%</span><span class="dl" style="{pc}">sobre inicial</span></div>'
        f'</div>', unsafe_allow_html=True)

    if ledger.get("quebrado"):
        st.error("💀 El agente quebró: bankroll en 0.")

    vals = [inicial] + [e["bankroll_final"] for e in hist]
    st.markdown(
        f'<div class="panel" style="padding:20px 22px 14px;margin-bottom:16px;">'
        f'<div style="display:flex;align-items:baseline;justify-content:space-between;">'
        f'<h2 style="margin:0;font-size:14px;font-weight:600;color:#cdd5df;">Evolución del bankroll</h2>'
        f'<span style="font-size:11.5px;color:#69727f;">eje X · número de jornada</span></div>'
        f'{svg_bankroll(vals)}</div>', unsafe_allow_html=True)

    if hist:
        filas = ""
        for e in reversed(hist):
            pcl = "color:#35d07f;" if e["ganancia_total"] >= 0 else "color:#ff5a4d;"
            filas += (f'<tr><td class="mono">{e["fecha"]}</td>'
                      f'<td class="mono">{fmt(e["bankroll_inicial"])}</td>'
                      f'<td class="mono">{fmt(e["bankroll_final"])}</td>'
                      f'<td class="mono" style="color:#8a94a3;">{fmt(e["stake_total"])}</td>'
                      f'<td class="mono" style="font-weight:600;{pcl}">{signed(e["ganancia_total"])}</td></tr>')
        st.markdown(
            f'<div class="panel" style="overflow:hidden;margin-bottom:16px;"><table class="dc">'
            f'<thead><tr><th>Jornada</th><th class="r">Inicio</th><th class="r">Cierre</th>'
            f'<th class="r">Apostado</th><th class="r">Resultado</th></tr></thead>'
            f'<tbody>{filas}</tbody></table></div>', unsafe_allow_html=True)
    else:
        st.info("Aún no hay jornadas simuladas. Pulsa **Simular jornada de hoy**.")

    st.markdown('<span style="font-size:12.5px;color:#69727f;">El agente apuesta entre 50% y 100% '
                'del bankroll por jornada. Presupuesto acumulable.</span>', unsafe_allow_html=True)
    b1, b2, _ = st.columns([0.32, 0.22, 1])
    if b1.button("▶  Simular jornada de hoy", use_container_width=True, type="primary"):
        with st.spinner("El agente está analizando y apostando…"):
            simulador.simular_dia(cfg, os.getenv("OPENAI_API_KEY"), date.today().isoformat())
        st.rerun()
    if b2.button("↺  Reiniciar", use_container_width=True):
        simulador.resetear(cfg)
        st.rerun()

# ------------------------------------------------------------
# PIE DE PÁGINA
# ------------------------------------------------------------
st.markdown(
    '<div class="gpt-footer">Solo para mayores de 18 años · Apostar implica riesgo real de pérdida · '
    "Ejercicio académico, no asesoría · Juega con responsabilidad.</div>",
    unsafe_allow_html=True,
)
