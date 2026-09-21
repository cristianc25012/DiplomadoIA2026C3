"""Cliente ligero de API-Football (api-sports.io, ruta directa).

Documentación: https://www.api-football.com/documentation-v3

Uso:
    from apuestas.api_football import ApiFootball
    api = ApiFootball()                       # lee APIFOOTBALL_API_KEY de secrets/.env
    fixtures = api.partidos_de_fecha(39, 2026, "2026-09-20")

El plan gratuito permite 100 peticiones/día (10/min). El cliente lleva la
cuenta de las peticiones realizadas en la propiedad `requests_realizadas`
para que puedas vigilar el consumo durante la construcción del corpus.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / "secrets" / ".env"

BASE_URL = "https://v3.football.api-sports.io"


class ApiFootballError(RuntimeError):
    """Error devuelto por la API o por la validación de la respuesta."""


class ApiFootball:
    """Envoltura mínima sobre los endpoints que necesita el RAG de apuestas."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        if api_key is None:
            load_dotenv(ENV_FILE)
            api_key = os.getenv("APIFOOTBALL_API_KEY")
        if not api_key or api_key.strip() in ("", "tu-api-key-de-api-football"):
            raise ApiFootballError(
                "No se encontró APIFOOTBALL_API_KEY en secrets/.env "
                "(¿sigue como placeholder?)."
            )
        self._headers = {"x-apisports-key": api_key.strip()}
        self._timeout = timeout
        self.requests_realizadas = 0

    # ------------------------------------------------------------------
    # Núcleo
    # ------------------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict] = None) -> List[Dict]:
        """Hace GET a un endpoint y devuelve la lista `response`.

        Lanza ApiFootballError si la API reporta errores o un HTTP != 200.
        """
        url = f"{BASE_URL}/{path.lstrip('/')}"
        r = requests.get(url, headers=self._headers, params=params or {}, timeout=self._timeout)
        self.requests_realizadas += 1

        if r.status_code != 200:
            raise ApiFootballError(f"HTTP {r.status_code} en {path}: {r.text[:200]}")

        data = r.json()
        errors = data.get("errors")
        # La API devuelve `errors` como {} (ok) o como dict/list con mensajes.
        if errors:
            raise ApiFootballError(f"Error de API en {path}: {errors}")
        return data.get("response", [])

    # ------------------------------------------------------------------
    # Endpoints de utilidad
    # ------------------------------------------------------------------
    def status(self) -> Dict:
        """Info de la cuenta: plan y consumo. No gasta cuota de datos."""
        resp = self._get("status")
        return resp if isinstance(resp, dict) else (resp[0] if resp else {})

    def partidos_de_hoy(self, fecha: str, league: Optional[int] = None,
                        timezone: str = "America/Bogota") -> List[Dict]:
        """Partidos de una fecha (YYYY-MM-DD), opcionalmente filtrados por liga.

        En el PLAN FREE la API exige `season` si se envía `league`, y solo
        admite fechas dentro de la ventana de hoy (±1 día). Por eso pedimos
        TODOS los partidos de la fecha (sin `league`) y filtramos por liga en
        el cliente. Así obtenemos los partidos reales del día.
        """
        fixtures = self._get("fixtures", {"date": fecha, "timezone": timezone})
        if league is not None:
            fixtures = [fx for fx in fixtures if fx.get("league", {}).get("id") == league]
        return fixtures

    def partidos_de_fecha(self, league: int, season: int, fecha: str,
                          timezone: str = "America/Bogota") -> List[Dict]:
        """Partidos de una liga/temporada en una fecha (YYYY-MM-DD).

        Requiere temporada 2022-2024 en el plan free y la fecha debe caer
        dentro de esa temporada. Para el día actual usa `partidos_de_hoy`.
        """
        return self._get("fixtures", {
            "league": league, "season": season, "date": fecha, "timezone": timezone,
        })

    def posiciones(self, league: int, season: int) -> List[Dict]:
        """Tabla de posiciones de la liga en la temporada dada."""
        return self._get("standings", {"league": league, "season": season})

    def ultimos_partidos_equipo(self, team: int, last: int = 5,
                                timezone: str = "America/Bogota") -> List[Dict]:
        """Últimos `last` partidos de un equipo (forma reciente).

        NOTA: el parámetro `last` NO está disponible en el plan free
        ("Free plans do not have access to the Last parameter"). Requiere
        plan de pago; se deja como extensión futura.
        """
        return self._get("fixtures", {"team": team, "last": last, "timezone": timezone})

    def head_to_head(self, home: int, away: int, last: int = 5,
                     timezone: str = "America/Bogota") -> List[Dict]:
        """Historial de enfrentamientos directos (H2H).

        NOTA: usa `last`, bloqueado en el plan free. Extensión futura.
        """
        return self._get("fixtures/headtohead", {
            "h2h": f"{home}-{away}", "last": last, "timezone": timezone,
        })

    def lesiones(self, fixture: int) -> List[Dict]:
        """Lesiones/bajas reportadas para un partido concreto."""
        return self._get("injuries", {"fixture": fixture})

    def cuotas_1x2(self, fixture: int) -> List[Dict]:
        """Cuotas pre-partido del mercado 1X2 (Match Winner = bet id 1)."""
        return self._get("odds", {"fixture": fixture, "bet": 1})


if __name__ == "__main__":
    # Prueba manual rápida del cliente.
    api = ApiFootball()
    st = api.status()
    sub = st.get("subscription", {})
    reqs = st.get("requests", {})
    print(f"Plan: {sub.get('plan')} | activo={sub.get('active')} | "
          f"peticiones: {reqs.get('current')}/{reqs.get('limit_day')}")
