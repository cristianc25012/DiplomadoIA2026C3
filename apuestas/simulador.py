"""Simulador de bankroll: apuestas ficticias acumulables.

El agente LLM arranca con un presupuesto (100.000) el día configurado y, por
cada jornada (fecha), decide en qué partidos apostar y qué porcentaje del
bankroll arriesgar (entre 50% y 100% en total). Como los partidos de la ventana
actual ya están finalizados, las apuestas se liquidan de inmediato contra el
resultado real y el bankroll se acumula hacia adelante.

El estado vive en un JSON (config: simulacion.ledger_path) y es idempotente por
fecha: una fecha ya jugada no se vuelve a apostar.

Uso:
    python -m apuestas.simulador            # simula la jornada de HOY
    python -m apuestas.simulador 2026-09-20 # simula una fecha concreta
    python -m apuestas.simulador --reset    # reinicia el bankroll a 100.000
"""

import os
import re
import sys
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from apuestas.api_football import ApiFootball, ApiFootballError, ligas_config
from apuestas.fichas import construir_ficha, agregar_cuotas
from ChatApuestas import load_config, _crear_chat, PROJECT_ROOT, ENV_FILE

CONFIG_PATH = PROJECT_ROOT / "config_apuestas.json"
_FINALIZADOS = {"FT", "AET", "PEN"}


# ------------------------------------------------------------
# LEDGER (estado persistente)
# ------------------------------------------------------------
def _ledger_path(cfg: Dict) -> Path:
    p = Path(cfg["simulacion"]["ledger_path"])
    return p if p.is_absolute() else PROJECT_ROOT / p


def cargar_ledger(cfg: Dict) -> Dict:
    path = _ledger_path(cfg)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    sim = cfg["simulacion"]
    return {
        "bankroll_inicial": sim["bankroll_inicial"],
        "bankroll": sim["bankroll_inicial"],
        "quebrado": False,
        "historial": [],
    }


def guardar_ledger(cfg: Dict, ledger: Dict) -> None:
    with open(_ledger_path(cfg), "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)


def resetear(cfg: Dict) -> Dict:
    ledger = {
        "bankroll_inicial": cfg["simulacion"]["bankroll_inicial"],
        "bankroll": cfg["simulacion"]["bankroll_inicial"],
        "quebrado": False,
        "historial": [],
    }
    guardar_ledger(cfg, ledger)
    return ledger


# ------------------------------------------------------------
# DATOS DE LA JORNADA
# ------------------------------------------------------------
def _resultado_real(fx: Dict) -> Optional[str]:
    """Devuelve 'Home' | 'Draw' | 'Away' si el partido terminó; None si no."""
    if fx["fixture"]["status"]["short"] not in _FINALIZADOS:
        return None
    if fx["teams"]["home"].get("winner"):
        return "Home"
    if fx["teams"]["away"].get("winner"):
        return "Away"
    return "Draw"


def datos_de_jornada(api: ApiFootball, cfg: Dict, fecha: str) -> List[Dict]:
    """Partidos FINALIZADOS de las ligas en la fecha, con cuotas y resultado real."""
    ids, _ = ligas_config(cfg["api"])
    tz = cfg["api"].get("timezone", "America/Bogota")
    fixtures = api.partidos_de_hoy(fecha, leagues=ids, timezone=tz)

    datos = []
    for fx in fixtures:
        resultado = _resultado_real(fx)
        if resultado is None:
            continue  # aún no se juega: no se puede liquidar
        fid = fx["fixture"]["id"]
        try:
            odds = api.cuotas_1x2(fid)
        except ApiFootballError:
            odds = []
        ag = agregar_cuotas(odds)
        if not ag:
            continue  # sin cuotas: no se puede calcular ganancia
        try:
            inj = api.lesiones(fid)
        except ApiFootballError:
            inj = []
        texto, meta = construir_ficha(fx, odds, inj)
        datos.append({
            "fixture_id": fid,
            "home": fx["teams"]["home"]["name"],
            "away": fx["teams"]["away"]["name"],
            "cuotas": ag["cuotas"],      # {'Home','Draw','Away'}
            "favorito": ag["favorito"],
            "resultado": resultado,       # 'Home' | 'Draw' | 'Away'
            "ficha": texto,
        })
    return datos


# ------------------------------------------------------------
# DECISIÓN DEL AGENTE (LLM -> JSON)
# ------------------------------------------------------------
def _extraer_json(texto: str) -> Optional[Dict]:
    """Extrae el primer objeto JSON del texto (tolera ```json ...``` y ruido)."""
    if not texto:
        return None
    limpio = re.sub(r"```(?:json)?|```", "", texto)
    m = re.search(r"\{.*\}", limpio, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _decision_respaldo(datos: List[Dict], pct_min: int) -> Dict:
    """Si el LLM falla: apuesta pct_min% al favorito más probable del día."""
    mejor = min(datos, key=lambda d: d["cuotas"][d["favorito"]])  # cuota más baja
    return {"apuestas": [{
        "fixture_id": mejor["fixture_id"], "pick": mejor["favorito"],
        "porcentaje": pct_min, "razon": "Respaldo: favorito más claro del día.",
    }]}


def decidir_apuestas(cfg: Dict, api_key: str, bankroll: float,
                     datos: List[Dict]) -> Dict:
    """Pide al LLM las apuestas de la jornada en JSON; valida y normaliza %."""
    sim = cfg["simulacion"]
    pct_min, pct_max = sim["stake_min_pct"], sim["stake_max_pct"]

    fichas_txt = "\n\n".join(
        f"[fixture_id={d['fixture_id']}] {d['home']} vs {d['away']} | "
        f"cuotas: Home {d['cuotas']['Home']:.2f}, Draw {d['cuotas']['Draw']:.2f}, "
        f"Away {d['cuotas']['Away']:.2f} | favorito: {d['favorito']}"
        for d in datos
    )
    system = (
        "Eres un agente apostador de fútbol. Recibes los partidos de una jornada "
        f"y un bankroll de {bankroll:.0f}. Elige apuestas 1X2 arriesgando EN TOTAL "
        f"entre {pct_min}% y {pct_max}% del bankroll (la suma de los porcentajes de "
        "todas tus apuestas debe caer en ese rango). Para cada apuesta indica el "
        "fixture_id, el pick ('Home', 'Draw' o 'Away') y el porcentaje del bankroll. "
        "Decide con base en las cuotas (menor cuota = más probable). Responde "
        "ÚNICAMENTE con JSON válido, sin texto extra, con esta forma:\n"
        '{"apuestas":[{"fixture_id":123,"pick":"Home","porcentaje":60,"razon":"..."}]}'
    )
    llm = _crear_chat(cfg, api_key, cfg["llm"]["model_name"])
    try:
        resp = llm.invoke([SystemMessage(content=system),
                           HumanMessage(content="Partidos:\n" + fichas_txt)])
        decision = _extraer_json(resp.content)
    except Exception:  # noqa: BLE001
        decision = None

    if not decision or not decision.get("apuestas"):
        return _decision_respaldo(datos, pct_min)

    # Validar apuestas contra los partidos reales
    validos = {d["fixture_id"]: d for d in datos}
    apuestas = []
    for a in decision["apuestas"]:
        fid = a.get("fixture_id")
        pick = a.get("pick")
        if fid in validos and pick in ("Home", "Draw", "Away"):
            try:
                pct = float(a.get("porcentaje", 0))
            except (TypeError, ValueError):
                pct = 0
            if pct > 0:
                apuestas.append({"fixture_id": fid, "pick": pick,
                                 "porcentaje": pct, "razon": a.get("razon", "")})
    if not apuestas:
        return _decision_respaldo(datos, pct_min)

    # Normalizar el total al rango [pct_min, pct_max]
    total = sum(a["porcentaje"] for a in apuestas)
    if total > pct_max:
        factor = pct_max / total
        for a in apuestas:
            a["porcentaje"] *= factor
    elif total < pct_min:
        factor = pct_min / total
        for a in apuestas:
            a["porcentaje"] *= factor
    return {"apuestas": apuestas}


# ------------------------------------------------------------
# SIMULACIÓN DE UNA JORNADA
# ------------------------------------------------------------
def simular_dia(cfg: Dict, api_key: str, fecha: str,
                ledger: Optional[Dict] = None) -> Dict:
    """Simula la jornada `fecha` y actualiza el ledger. Idempotente por fecha."""
    if ledger is None:
        ledger = cargar_ledger(cfg)
    if ledger.get("quebrado"):
        return ledger
    if any(e["fecha"] == fecha for e in ledger["historial"]):
        print(f"La fecha {fecha} ya está en el histórico; no se repite.")
        return ledger

    api = ApiFootball()
    datos = datos_de_jornada(api, cfg, fecha)
    if not datos:
        print(f"No hay partidos finalizados con cuotas para {fecha}.")
        return ledger

    bankroll_ini = ledger["bankroll"]
    decision = decidir_apuestas(cfg, api_key, bankroll_ini, datos)
    por_id = {d["fixture_id"]: d for d in datos}

    apuestas_res = []
    ganancia_total = 0.0
    for a in decision["apuestas"]:
        d = por_id[a["fixture_id"]]
        stake = round(bankroll_ini * a["porcentaje"] / 100.0, 2)
        cuota = d["cuotas"][a["pick"]]
        gano = (a["pick"] == d["resultado"])
        ganancia = round(stake * (cuota - 1), 2) if gano else -stake
        ganancia_total += ganancia
        apuestas_res.append({
            "fixture_id": a["fixture_id"],
            "partido": f"{d['home']} vs {d['away']}",
            "pick": a["pick"],
            "cuota": cuota,
            "stake": stake,
            "resultado_real": d["resultado"],
            "acierto": gano,
            "ganancia": ganancia,
            "razon": a.get("razon", ""),
        })

    bankroll_fin = round(bankroll_ini + ganancia_total, 2)
    quebrado = bankroll_fin <= 0
    ledger["bankroll"] = max(bankroll_fin, 0)
    ledger["quebrado"] = quebrado
    ledger["historial"].append({
        "fecha": fecha,
        "bankroll_inicial": bankroll_ini,
        "stake_total": round(sum(x["stake"] for x in apuestas_res), 2),
        "ganancia_total": round(ganancia_total, 2),
        "bankroll_final": ledger["bankroll"],
        "apuestas": apuestas_res,
    })
    guardar_ledger(cfg, ledger)
    return ledger


def _imprimir_resumen(ledger: Dict) -> None:
    print(f"\nBankroll actual: {ledger['bankroll']:,.0f} "
          f"(inicial {ledger['bankroll_inicial']:,.0f})"
          + ("  [QUEBRADO]" if ledger.get("quebrado") else ""))
    for e in ledger["historial"]:
        signo = "+" if e["ganancia_total"] >= 0 else ""
        print(f"  {e['fecha']}: apostó {e['stake_total']:,.0f}, "
              f"{signo}{e['ganancia_total']:,.0f} -> {e['bankroll_final']:,.0f}")
        for a in e["apuestas"]:
            ok = "✓" if a["acierto"] else "✗"
            print(f"      {ok} {a['partido']} | {a['pick']} @ {a['cuota']:.2f} | "
                  f"stake {a['stake']:,.0f} | {a['ganancia']:+,.0f}")


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")

    if "--reset" in sys.argv:
        resetear(cfg)
        print("Simulación reiniciada.")
        _imprimir_resumen(cargar_ledger(cfg))
        sys.exit(0)

    fechas = [a for a in sys.argv[1:] if not a.startswith("--")]
    fecha = fechas[0] if fechas else date.today().isoformat()
    ledger = simular_dia(cfg, api_key, fecha)
    _imprimir_resumen(ledger)
