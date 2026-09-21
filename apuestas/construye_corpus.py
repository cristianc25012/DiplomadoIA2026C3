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
from datetime import date
from pathlib import Path
from typing import Dict, List

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake

from apuestas.api_football import ApiFootball, ApiFootballError
from apuestas.fichas import construir_ficha

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitizar_meta(meta: Dict) -> Dict:
    """Deep Lake no admite None en metadatos; se reemplaza por cadena vacía."""
    return {k: ("" if v is None else v) for k, v in meta.items()}


def construir_documentos(api: ApiFootball, cfg: Dict) -> List[Document]:
    api_cfg = cfg["api"]
    league = int(api_cfg["league_id"])
    tz = api_cfg.get("timezone", "America/Bogota")
    fecha = api_cfg.get("fecha", "hoy")
    if fecha == "hoy":
        fecha = date.today().isoformat()

    print(f"Buscando partidos de {api_cfg.get('league_name', league)} "
          f"({league}) para la fecha {fecha}...")
    fixtures = api.partidos_de_hoy(fecha, league=league, timezone=tz)
    print(f"Partidos encontrados: {len(fixtures)}")

    documentos: List[Document] = []
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
        print(f"  Ficha creada: {home} vs {away} (fixture {fid})")

    return documentos


def build_base(cfg: Dict) -> None:
    api = ApiFootball()

    documentos = construir_documentos(api, cfg)
    if not documentos:
        print("No hay partidos para la fecha; no se crea la base.")
        return

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

    print(f"\nBase creada con {len(documentos)} fichas.")
    print(f"Peticiones API usadas en esta construcción: {api.requests_realizadas}")


if __name__ == "__main__":
    cfg = load_config(PROJECT_ROOT / "config_apuestas.json")
    build_base(cfg)
