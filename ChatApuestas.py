"""Asistente de apuestas de fútbol — RAG híbrido (RAG + agente).

Flujo por pregunta:
  1. Recupera las fichas de partidos más relevantes de la base Deep Lake
     (top-k, no solo el primero).
  2. Las pasa como contexto a un LLM (Qwen vía OpenRouter, con fallback).
  3. El LLM analiza y recomienda picks 1X2 con cuota, confianza y justificación.
  4. Puede invocar la herramienta `refrescar_cuotas(fixture_id)` para traer
     cuotas en vivo desde la API (parte "agente" del híbrido).

Ejecutar:
    python ChatApuestas.py
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import deeplake
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from apuestas.api_football import ApiFootball, ApiFootballError
from apuestas.fichas import agregar_cuotas

warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / "secrets" / ".env"


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# RECUPERACIÓN (top-k, corrige el uso de un solo fragmento)
# ------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


class Recuperador:
    """Carga la base una sola vez y recupera las k fichas más similares."""

    def __init__(self, cfg: Dict):
        emb_cfg = cfg["embedding"]
        dl_cfg = cfg["deeplake"]
        self.k = int(cfg["retrieval"].get("k", 6))

        dataset_path = Path(dl_cfg["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"No existe la base en {dataset_path}. "
                f"Ejecuta primero: python -m apuestas.construye_corpus"
            )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=emb_cfg["model_name"],
            model_kwargs={"device": emb_cfg.get("device", "cpu")},
            encode_kwargs={"normalize_embeddings": bool(emb_cfg.get("normalize_embeddings", True))},
        )
        ds = deeplake.load(str(dataset_path), read_only=True, verbose=False)
        ds.checkout("main")
        self.textos: List[str] = ds["text"].data(aslist=True)["value"]
        self.metas: List[Dict] = ds["metadata"].data(aslist=True)["value"]
        self.vectores = np.array(ds["embedding"].numpy(), dtype=np.float32)
        print(f"Base cargada: {len(self.textos)} fichas.")

    def buscar(self, query: str) -> List[Tuple[str, Dict, float]]:
        q = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        scores = [_cosine(q, v) for v in self.vectores]
        orden = np.argsort(scores)[::-1][: min(self.k, len(scores))]
        return [(self.textos[i], self.metas[i], scores[i]) for i in orden]


def construir_contexto(resultados: List[Tuple[str, Dict, float]]) -> str:
    """Une las k fichas en un bloque de contexto, exponiendo el fixture_id."""
    bloques = []
    for n, (texto, meta, score) in enumerate(resultados, start=1):
        fid = meta.get("fixture_id", "?")
        bloques.append(
            f"FICHA {n} (relevancia {score:.2f}, fixture_id={fid}):\n{texto}"
        )
    return "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(bloques)


# ------------------------------------------------------------
# HERRAMIENTA DEL AGENTE: cuotas en vivo
# ------------------------------------------------------------
@tool
def refrescar_cuotas(fixture_id: int) -> str:
    """Consulta las cuotas 1X2 ACTUALES de un partido por su fixture_id.

    Úsala cuando el usuario pida cuotas actualizadas/en vivo o cuando quieras
    confirmar las cuotas de un partido concreto. Devuelve las cuotas promedio
    (Local/Empate/Visita) y sus probabilidades implícitas.
    """
    try:
        api = ApiFootball()
        odds = api.cuotas_1x2(int(fixture_id))
    except (ApiFootballError, ValueError) as e:
        return f"No se pudieron obtener cuotas para el fixture {fixture_id}: {e}"

    ag = agregar_cuotas(odds)
    if not ag:
        return f"Sin cuotas 1X2 disponibles para el fixture {fixture_id}."
    c, p = ag["cuotas"], ag["prob"]
    return (
        f"Cuotas 1X2 en vivo (fixture {fixture_id}, promedio de {ag['n_casas']} casas): "
        f"Local {c['Home']:.2f} ({p['Home']*100:.0f}%), "
        f"Empate {c['Draw']:.2f} ({p['Draw']*100:.0f}%), "
        f"Visita {c['Away']:.2f} ({p['Away']*100:.0f}%)."
    )


# ------------------------------------------------------------
# LLM CON FALLBACK + TOOLS
# ------------------------------------------------------------
def _crear_chat(cfg: Dict, api_key: str, model_name: str) -> ChatOpenAI:
    llm_cfg = cfg["llm"]
    return ChatOpenAI(
        openai_api_base=llm_cfg["api_base"],
        openai_api_key=api_key,
        model_name=model_name,
        temperature=float(llm_cfg.get("temperature", 0.4)),
        timeout=60,
        max_retries=1,  # ante 429 cae rápido al fallback en vez de reintentar mucho
        streaming=True,
    )


def construir_llms(cfg: Dict, api_key: str) -> List:
    """Modelos con herramientas, en orden de preferencia: [principal, fallback].

    Se devuelve como LISTA (no como `with_fallbacks`) porque el streaming a
    través de RunnableWithFallbacks se bufferiza y se pierde el token-a-token;
    el fallback en streaming se maneja a mano en `responder_stream`.
    """
    llm_cfg = cfg["llm"]
    herramientas = [refrescar_cuotas]
    modelos = [_crear_chat(cfg, api_key, llm_cfg["model_name"]).bind_tools(herramientas)]
    fb_name = llm_cfg.get("fallback_model")
    if fb_name:
        modelos.append(_crear_chat(cfg, api_key, fb_name).bind_tools(herramientas))
    return modelos


def construir_llm(cfg: Dict, api_key: str):
    """LLM con fallback automático para uso NO-streaming (responder/chat)."""
    modelos = construir_llms(cfg, api_key)
    return modelos[0].with_fallbacks(modelos[1:]) if len(modelos) > 1 else modelos[0]


# ------------------------------------------------------------
# BUCLE DE CHAT
# ------------------------------------------------------------
def _resolver_tool_calls(llm, mensajes: List, max_iter: int = 3) -> AIMessage:
    """Ejecuta las herramientas que pida el modelo hasta obtener respuesta final."""
    tools_by_name = {"refrescar_cuotas": refrescar_cuotas}
    ai = llm.invoke(mensajes)
    it = 0
    while getattr(ai, "tool_calls", None) and it < max_iter:
        mensajes.append(ai)
        for tc in ai.tool_calls:
            fn = tools_by_name.get(tc["name"])
            if fn is None:
                obs = f"Herramienta desconocida: {tc['name']}"
            else:
                try:
                    obs = fn.invoke(tc["args"])
                except Exception as e:  # noqa: BLE001
                    obs = f"Error ejecutando {tc['name']}: {e}"
            print(f"   [agente] {tc['name']}({tc['args']}) -> {obs}")
            mensajes.append(ToolMessage(content=str(obs), tool_call_id=tc["id"]))
        ai = llm.invoke(mensajes)
        it += 1
    return ai


def _system_con_contexto(system_instruction: str, contexto: str) -> str:
    return (
        f"{system_instruction}\n\n"
        "Las fichas del contexto YA incluyen las cuotas y probabilidades: úsalas "
        "directamente para tu análisis. Dispones además de la herramienta "
        "refrescar_cuotas(fixture_id), pero llámala SOLO si el usuario pide cuotas "
        "actualizadas/en vivo de un partido concreto; para el análisis general NO "
        "la uses. Usa el fixture_id que aparece en cada ficha.\n\n"
        "PARTIDOS DISPONIBLES (contexto recuperado de la base):"
        f"{contexto}"
    )


def responder(llm, recuperador: "Recuperador", system_instruction: str,
              pregunta: str, historial: List = None) -> Tuple[str, List[Tuple[str, Dict, float]]]:
    """Responde una pregunta usando RAG + agente. Reutilizable por consola y front.

    Devuelve (respuesta, fichas_recuperadas).
    """
    historial = historial or []
    resultados = recuperador.buscar(pregunta)
    contexto = construir_contexto(resultados)
    system_txt = _system_con_contexto(system_instruction, contexto)
    mensajes = [SystemMessage(content=system_txt)] + historial + [HumanMessage(content=pregunta)]
    ai = _resolver_tool_calls(llm, mensajes)
    return (ai.content or "").strip(), resultados


def responder_stream(llms, recuperador: "Recuperador", system_instruction: str,
                     pregunta: str, historial: List = None, max_iter: int = 4):
    """Versión *generadora* de `responder`: entrega el texto por trozos (streaming).

    `llms` es la lista de `construir_llms` ([principal, fallback]). Transmite
    con el primero; si falla ANTES de emitir texto (p. ej. 429), pasa al
    siguiente. Maneja el bucle del agente: si un turno pide herramientas, las
    ejecuta y continúa al turno siguiente (la respuesta final). Pensada para
    `st.write_stream` en el front.
    """
    if not isinstance(llms, (list, tuple)):
        llms = [llms]
    historial = historial or []
    resultados = recuperador.buscar(pregunta)
    contexto = construir_contexto(resultados)
    system_txt = _system_con_contexto(system_instruction, contexto)
    mensajes = [SystemMessage(content=system_txt)] + historial + [HumanMessage(content=pregunta)]
    tools_by_name = {"refrescar_cuotas": refrescar_cuotas}

    for _ in range(max_iter):
        acc = None
        emitido = False
        exito = False
        ultimo_error = None
        for modelo in llms:
            try:
                for chunk in modelo.stream(mensajes):
                    acc = chunk if acc is None else acc + chunk
                    if getattr(chunk, "content", ""):
                        emitido = True
                        yield chunk.content
                exito = True
                break
            except Exception as e:  # noqa: BLE001
                ultimo_error = e
                if emitido:
                    raise  # ya mostramos texto: no reintentar con otro modelo
                acc = None
                continue  # probar el siguiente modelo
        if not exito:
            raise ultimo_error or RuntimeError("Sin modelos disponibles")

        tool_calls = getattr(acc, "tool_calls", None) if acc is not None else None
        if not tool_calls:
            return
        # Turno de herramientas: sin texto que mostrar; ejecutar y seguir.
        mensajes.append(acc)
        for tc in tool_calls:
            fn = tools_by_name.get(tc["name"])
            if fn is None:
                obs = f"Herramienta desconocida: {tc['name']}"
            else:
                try:
                    obs = fn.invoke(tc["args"])
                except Exception as e:  # noqa: BLE001
                    obs = f"Error ejecutando {tc['name']}: {e}"
            mensajes.append(ToolMessage(content=str(obs), tool_call_id=tc["id"]))


def chat(cfg: Dict):
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY en secrets/.env")

    recuperador = Recuperador(cfg)
    llm = construir_llm(cfg, api_key)
    system_instruction = cfg.get("system_instruction", "Eres un analista de apuestas.")

    historial: List = []  # memoria corta de la conversación (Human/AI)

    print("\nAsistente de apuestas listo. Pregunta, por ejemplo:")
    print("  '¿Cuáles son las mejores apuestas de hoy?'")
    print("Escribe 'salir' para terminar.\n")

    while True:
        pregunta = input("👤 Usuario: ").strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            print("Fin de la sesión. Recuerda: apostar es solo para mayores de 18 y con responsabilidad.")
            break
        if not pregunta:
            continue

        try:
            respuesta, resultados = responder(llm, recuperador, system_instruction, pregunta, historial)
            print(f"(recuperadas {len(resultados)} fichas)")
            print("\n🤖 Analista:\n")
            print(respuesta)
            print("-" * 80)
            historial.append(HumanMessage(content=pregunta))
            historial.append(AIMessage(content=respuesta))
            # Limitar memoria corta a las últimas 6 intervenciones
            historial[:] = historial[-6:]
        except Exception as e:  # noqa: BLE001
            print(f"Error al consultar el modelo: {e}")


if __name__ == "__main__":
    cfg = load_config(PROJECT_ROOT / "config_apuestas.json")
    chat(cfg)
