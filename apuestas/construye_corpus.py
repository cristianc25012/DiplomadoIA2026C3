"""Construye la base vectorial Deep Lake del RAG de apuestas.

Flujo:
    partidos de hoy (API-Football) -> ficha por partido (cuotas + lesiones)
    -> embeddings -> Deep Lake (deeplake_db/apuestas)

Es el equivalente de TensorialBase.py, pero con un corpus dinámico generado
desde la API en vez de un PDF estático.

Ejecutar:
    python apuestas/construye_corpus.py
"""

import os
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake

from apuestas.api_football import ApiFootball, ApiFootballError, ligas_config
from apuestas.fichas import construir_ficha, agregar_cuotas

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitizar_meta(meta: Dict) -> Dict:
    """Deep Lake no admite None en metadatos; se reemplaza por cadena vacía."""
    return {k: ("" if v is None else v) for k, v in meta.items()}


def resolver_jornada(api: ApiFootball, cfg: Dict):
    """Devuelve (fecha, fixtures) de las ligas seguidas.

    Si cfg.api.fecha == "hoy", busca en la ventana que permite el plan free
    (hoy, ayer, mañana) y usa la primera fecha con partidos en CUALQUIERA de las
    ligas, para no quedarse con una base vacía los días que no juegan. Si es una
    fecha explícita, usa esa.
    """
    api_cfg = cfg["api"]
    ids, _ = ligas_config(api_cfg)
    tz = api_cfg.get("timezone", "America/Bogota")
    fecha_cfg = api_cfg.get("fecha", "hoy")

    if fecha_cfg != "hoy":
        return fecha_cfg, api.partidos_de_hoy(fecha_cfg, leagues=ids, timezone=tz)

    hoy = date.today()
    for cand in (hoy, hoy - timedelta(days=1), hoy + timedelta(days=1)):
        f = cand.isoformat()
        fixtures = api.partidos_de_hoy(f, leagues=ids, timezone=tz)
        if fixtures:
            return f, fixtures
    return hoy.isoformat(), []


def _outcome(fx: Dict):
    """'Home' | 'Draw' | 'Away' si el partido terminó; None si no."""
    if fx["fixture"]["status"]["short"] not in ("FT", "AET", "PEN"):
        return None
    if fx["teams"]["home"].get("winner"):
        return "Home"
    if fx["teams"]["away"].get("winner"):
        return "Away"
    return "Draw"


def construir_jornada(api: ApiFootball, cfg: Dict):
    """Devuelve (fecha, documentos, snapshot) de la jornada resuelta.

    - documentos: para la base Deep Lake (RAG).
    - snapshot: resumen ligero de cada partido para el archivo de jornadas.
    """
    api_cfg = cfg["api"]
    _, nombres = ligas_config(api_cfg)
    fecha, fixtures = resolver_jornada(api, cfg)

    # Tope de partidos: cuida la cuota del plan free (cada partido = 2 peticiones).
    # Se ordena por hora de inicio para quedarse con los primeros del día.
    fixtures = sorted(fixtures, key=lambda fx: fx["fixture"].get("date", ""))
    max_p = int(api_cfg.get("max_partidos", 40))
    if len(fixtures) > max_p:
        print(f"Aviso: {len(fixtures)} partidos exceden el tope ({max_p}); "
              f"se recortan para cuidar la cuota de la API.")
        fixtures = fixtures[:max_p]

    etiqueta = ", ".join(nombres.values())
    print(f"Jornada {fecha} — {len(fixtures)} partidos ({etiqueta})")

    documentos: List[Document] = []
    snapshot: List[Dict] = []
    for fx in fixtures:
        fid = fx["fixture"]["id"]
        home = fx["teams"]["home"]["name"]
        away = fx["teams"]["away"]["name"]
        try:
            odds = api.cuotas_1x2(fid)
        except ApiFootballError as e:
            print(f"  [cuotas] {home} vs {away}: {e}")
            odds = []
        try:
            inj = api.lesiones(fid)
        except ApiFootballError as e:
            print(f"  [lesiones] {home} vs {away}: {e}")
            inj = []

        texto, meta = construir_ficha(fx, odds, inj)
        documentos.append(Document(page_content=texto, metadata=_sanitizar_meta(meta)))

        ag = agregar_cuotas(odds)
        snapshot.append({
            "fixture_id": fid, "home": home, "away": away,
            "liga": fx["league"]["name"],
            "favorito": ag["favorito"] if ag else None,
            "cuotas": ag["cuotas"] if ag else None,
            "resultado": _outcome(fx),
        })
        print(f"  Ficha creada: [{fx['league']['name']}] {home} vs {away} (fixture {fid})")

    return fecha, documentos, snapshot


# ------------------------------------------------------------
# ARCHIVO DE JORNADAS (histórico de partidos por fecha)
# ------------------------------------------------------------
def _jornadas_path(cfg: Dict) -> Path:
    p = Path(cfg["api"].get("jornadas_path", "apuestas/jornadas.json"))
    return p if p.is_absolute() else PROJECT_ROOT / p


def cargar_jornadas(cfg: Dict) -> Dict:
    path = _jornadas_path(cfg)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def archivar_jornada(cfg: Dict, fecha: str, snapshot: List[Dict]) -> None:
    data = cargar_jornadas(cfg)
    data[fecha] = snapshot
    with open(_jornadas_path(cfg), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_base(cfg: Dict):
    """Construye la base y devuelve la fecha de la jornada cargada (o None)."""
    api = ApiFootball()

    fecha, documentos, snapshot = construir_jornada(api, cfg)
    if not documentos:
        print("No hay partidos en la ventana disponible; no se crea la base.")
        return None
    archivar_jornada(cfg, fecha, snapshot)  # guarda la jornada en el histórico

    emb_cfg = cfg["embedding"]
    print(f"\nCargando modelo de embeddings: {emb_cfg['model_name']}")
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg["model_name"],
        model_kwargs={"device": emb_cfg.get("device", "cpu")},
        encode_kwargs={"normalize_embeddings": bool(emb_cfg.get("normalize_embeddings", True))},
    )

    dl_cfg = cfg["deeplake"]
    dataset_path = Path(dl_cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    print(f"Creando base Deep Lake en: {dataset_path} "
          f"(overwrite={dl_cfg.get('overwrite', True)})")
    DeepLake.from_documents(
        documents=documentos,
        embedding=embeddings,
        dataset_path=str(dataset_path),
        overwrite=bool(dl_cfg.get("overwrite", True)),
    )

    print(f"\nBase creada con {len(documentos)} fichas (jornada {fecha}).")
    print(f"Peticiones API usadas en esta construcción: {api.requests_realizadas}")
    return fecha


if __name__ == "__main__":
    cfg = load_config(PROJECT_ROOT / "config_apuestas.json")
    build_base(cfg)
