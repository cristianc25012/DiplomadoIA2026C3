"""Construcción de 'fichas' de texto por partido para el corpus RAG.

Cada ficha resume, en lenguaje natural, la información de un partido:
cuotas 1X2 (promedio de casas), probabilidades implícitas sin margen,
favorito del mercado, lesiones/bajas y (si ya se jugó) el resultado real.

Las funciones son puras: reciben las respuestas crudas de la API y
devuelven texto + metadatos, para poder probarlas sin llamar a la red.
"""

from typing import Dict, List, Optional, Tuple

_ETIQUETAS = {"Home": "Local", "Draw": "Empate", "Away": "Visita"}


def agregar_cuotas(odds_response: List[Dict]) -> Optional[Dict]:
    """Promedia las cuotas 1X2 de todas las casas y calcula prob. implícitas.

    Devuelve None si no hay cuotas. Estructura de salida:
        {
          "n_casas": int,
          "cuotas": {"Home": float, "Draw": float, "Away": float},
          "prob": {"Home": float, "Draw": float, "Away": float},  # normalizadas 0-1
          "favorito": "Home" | "Draw" | "Away",
        }
    """
    if not odds_response:
        return None
    bookmakers = odds_response[0].get("bookmakers", [])
    if not bookmakers:
        return None

    sumas = {"Home": 0.0, "Draw": 0.0, "Away": 0.0}
    conteo = {"Home": 0, "Draw": 0, "Away": 0}
    for bm in bookmakers:
        for bet in bm.get("bets", []):
            if bet.get("id") == 1 or bet.get("name") == "Match Winner":
                for v in bet.get("values", []):
                    etq = v.get("value")
                    if etq in sumas:
                        try:
                            sumas[etq] += float(v["odd"])
                            conteo[etq] += 1
                        except (ValueError, KeyError):
                            pass

    if min(conteo.values()) == 0:
        return None

    cuotas = {k: round(sumas[k] / conteo[k], 2) for k in sumas}
    # Probabilidad implícita = 1/cuota, normalizada para quitar el margen (vig).
    inv = {k: 1.0 / cuotas[k] for k in cuotas}
    total = sum(inv.values())
    prob = {k: round(inv[k] / total, 3) for k in inv}
    favorito = min(cuotas, key=cuotas.get)  # menor cuota = favorito

    return {
        "n_casas": max(conteo.values()),
        "cuotas": cuotas,
        "prob": prob,
        "favorito": favorito,
    }


def resumir_lesiones(injuries_response: List[Dict], home_id: int, away_id: int) -> Dict[int, List[str]]:
    """Agrupa las bajas por equipo -> {team_id: ['Jugador (motivo)', ...]}.

    La API suele repetir cada jugador; se deduplica por id de jugador.
    """
    por_equipo: Dict[int, List[str]] = {home_id: [], away_id: []}
    vistos = set()
    for item in injuries_response or []:
        tid = item.get("team", {}).get("id")
        if tid not in por_equipo:
            continue
        pl = item.get("player", {})
        clave = (tid, pl.get("id") or pl.get("name"))
        if clave in vistos:
            continue
        vistos.add(clave)
        nombre = pl.get("name", "?")
        motivo = pl.get("reason") or pl.get("type") or "baja"
        por_equipo[tid].append(f"{nombre} ({motivo})")
    return por_equipo


def construir_ficha(fixture: Dict, odds_response: List[Dict],
                    injuries_response: List[Dict]) -> Tuple[str, Dict]:
    """Devuelve (texto_ficha, metadatos) para un partido."""
    fx = fixture["fixture"]
    lg = fixture["league"]
    home = fixture["teams"]["home"]
    away = fixture["teams"]["away"]
    goals = fixture.get("goals", {})
    status = fx.get("status", {}).get("short", "NS")
    venue = fx.get("venue", {}) or {}

    lineas = [
        f"PARTIDO: {home['name']} (local) vs {away['name']} (visitante)",
        f"Liga: {lg['name']} ({lg['country']}) — {lg.get('round', '')}".rstrip(" —"),
        f"Fecha: {fx['date']}",
    ]
    if venue.get("name"):
        lineas.append(f"Estadio: {venue['name']}, {venue.get('city', '')}".rstrip(", "))

    # Cuotas y probabilidades
    ag = agregar_cuotas(odds_response)
    if ag:
        c, p = ag["cuotas"], ag["prob"]
        lineas.append("")
        lineas.append(f"CUOTAS 1X2 (promedio de {ag['n_casas']} casas):")
        for k in ("Home", "Draw", "Away"):
            if k == "Draw":
                etiqueta = "Empate"
            else:
                equipo = home["name"] if k == "Home" else away["name"]
                etiqueta = f"{_ETIQUETAS[k]} ({equipo})"
            lineas.append(f"  {etiqueta}: cuota {c[k]:.2f} "
                          f"-> prob. implícita {p[k]*100:.0f}%")
        fav_txt = (home["name"] if ag["favorito"] == "Home"
                   else away["name"] if ag["favorito"] == "Away" else "el empate")
        lineas.append(f"Favorito del mercado: {fav_txt} "
                      f"({_ETIQUETAS[ag['favorito']]}, cuota más baja).")
    else:
        lineas.append("")
        lineas.append("CUOTAS 1X2: no disponibles para este partido.")

    # Lesiones
    les = resumir_lesiones(injuries_response, home["id"], away["id"])
    lineas.append("")
    lineas.append("LESIONES / BAJAS:")
    for equipo, tid in ((home, home["id"]), (away, away["id"])):
        jugadores = les.get(tid, [])
        if jugadores:
            muestra = ", ".join(jugadores[:6])
            extra = f" (+{len(jugadores) - 6} más)" if len(jugadores) > 6 else ""
            lineas.append(f"  {equipo['name']} ({len(jugadores)}): {muestra}{extra}")
        else:
            lineas.append(f"  {equipo['name']}: sin bajas reportadas.")

    # Resultado real (si ya se jugó) — útil para evaluar los picks
    if status in ("FT", "AET", "PEN"):
        gh, ga = goals.get("home"), goals.get("away")
        ganador = (home["name"] if home.get("winner") else
                   away["name"] if away.get("winner") else "Empate")
        lineas.append("")
        lineas.append(f"RESULTADO REAL: {home['name']} {gh}-{ga} {away['name']} "
                      f"(resultado: {ganador}).")

    texto = "\n".join(lineas)
    meta = {
        "fixture_id": fx["id"],
        "league_id": lg["id"],
        "league": lg["name"],
        "fecha": fx["date"][:10],
        "home": home["name"],
        "away": away["name"],
        "status": status,
        "favorito": ag["favorito"] if ag else None,
    }
    return texto, meta
