#!/usr/bin/env python3
import os
import re
import sys
import logging
import time
import asyncio
import secrets
from collections import deque
from typing import Optional, Any
from urllib.parse import urlencode

import httpx
import nh3
from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

ML_CLIENT_ID         = os.environ.get("ML_CLIENT_ID", "")
ML_CLIENT_SECRET     = os.environ.get("ML_CLIENT_SECRET", "")
ML_SITE              = os.environ.get("ML_SITE", "MLB")
TOKEN_REFRESH_BUFFER = int(os.environ.get("TOKEN_REFRESH_BUFFER", "1800"))
BEARER_TOKEN         = os.environ.get("BEARER_TOKEN", "")
ALLOW_TOKEN_QUERY_PARAM = os.environ.get("ALLOW_TOKEN_QUERY_PARAM", "") == "1"
MAX_REQUEST_BODY     = int(os.environ.get("MAX_REQUEST_BODY", str(1024 * 1024)))
PORT                 = int(os.environ.get("PORT", "8000"))

_RATE_LIMIT_RPM  = int(os.environ.get("RATE_LIMIT_RPM", "60"))
_MAX_TRACKED_IPS = int(os.environ.get("RATE_LIMIT_MAX_IPS", "10000"))
_TRUSTED_PROXIES = int(os.environ.get("TRUSTED_PROXY_COUNT", "1"))

ML_AUTH_BASE = "https://auth.mercadolivre.com.br"
ML_API_BASE  = "https://api.mercadolibre.com"
ML_TOKEN_URL = f"{ML_API_BASE}/oauth/token"

# Mercado Pago Brasil: acesso à própria conta por Client Credentials.
MP_CLIENT_ID = os.environ.get("MP_CLIENT_ID", "")
MP_CLIENT_SECRET = os.environ.get("MP_CLIENT_SECRET", "")
MP_AUTH_MODE = os.environ.get("MP_AUTH_MODE", "client_credentials").strip().lower()
MP_API_BASE = "https://api.mercadopago.com"
MP_TOKEN_URL = f"{MP_API_BASE}/oauth/token"
MP_ACCESS_TOKEN = os.environ.get("MP_ACCESS_TOKEN", "")
MP_REFRESH_TOKEN = os.environ.get("MP_REFRESH_TOKEN", "")
_MP_TIMEOUT = httpx.Timeout(30.0)

# Se aplica a cada llamada saliente a la API de ML — evita que los workers queden colgados
# cuando ML está lento o no responde.
_ML_TIMEOUT = httpx.Timeout(30.0)

# Compilado una sola vez al iniciar usando ML_SITE — evita path traversal en la interpolación de URLs.
_RE_ITEM_ID     = re.compile(rf"^{re.escape(ML_SITE)}[0-9]+$")
_RE_NUMERIC_ID  = re.compile(r"^\d+$")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("ml_mcp")

# Evita que httpx registre las URLs completas de los requests (expondría access tokens en headers/params).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Validación de inicio
# ---------------------------------------------------------------------------

if not BEARER_TOKEN and not os.environ.get("ALLOW_OPEN_SERVER"):
    logger.critical(
        "BEARER_TOKEN is not set. Refusing to start without authentication. "
        "Set BEARER_TOKEN in your environment variables, or set ALLOW_OPEN_SERVER=1 to bypass (not recommended)."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Limitador de tasa (ventana deslizante, en proceso)
# ---------------------------------------------------------------------------

_rate_limit_store: dict[str, deque] = {}
_rate_limit_lock = asyncio.Lock()


async def _check_rate_limit(ip: str) -> bool:
    now    = time.monotonic()
    window = 60.0
    async with _rate_limit_lock:
        if ip not in _rate_limit_store:
            if len(_rate_limit_store) >= _MAX_TRACKED_IPS:
                # Elimina entradas vencidas antes de fallar en modo abierto — recupera espacio
                # ocupado por la rotación de IPs de bots.
                stale = [k for k, v in _rate_limit_store.items() if not v or now - max(v) >= window]
                for k in stale:
                    del _rate_limit_store[k]
                if len(_rate_limit_store) >= _MAX_TRACKED_IPS:
                    logger.warning("rate-limit store full, failing open for %s", ip)
                    return True
            _rate_limit_store[ip] = deque()
        dq = _rate_limit_store[ip]
        while dq and now - dq[0] >= window:
            dq.popleft()
        if len(dq) >= _RATE_LIMIT_RPM:
            return False
        dq.append(now)
        return True


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff and _TRUSTED_PROXIES > 0:
        parts = [p.strip() for p in xff.split(",")]
        idx   = max(0, len(parts) - _TRUSTED_PROXIES)
        return parts[idx]
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Gestor de tokens
# ---------------------------------------------------------------------------

class TokenManager:
    """Gestiona los tokens OAuth 2.0 de ML con renovación automática."""

    def __init__(self, refresh_buffer: int = 1800):
        self._refresh_buffer = refresh_buffer
        self._access_token:  str   = os.environ.get("ML_ACCESS_TOKEN", "")
        # El refresh token rota en cada uso — persistir el último en memoria.
        self._refresh_token: str   = os.environ.get("ML_REFRESH_TOKEN", "")
        # Si se proveyó un token por variable de entorno, se asume que está vigente (los tokens de ML duran 6 h).
        # Poner _expires_at en 0 dispararía una renovación innecesaria en la primera llamada.
        # Não presumimos que um token salvo no Railway ainda esteja dentro das 6h.
        self._expires_at: float = (
            time.monotonic() + 21600
            if self._access_token and not self._refresh_token
            else 0.0
        )
        self._lock = asyncio.Lock()

    def set_tokens(self, access_token: str, refresh_token: str, expires_in: int) -> None:
        self._access_token  = access_token
        self._refresh_token = refresh_token
        self._expires_at    = time.monotonic() + expires_in

    async def get_token(self) -> str:
        if self._access_token and time.monotonic() < self._expires_at - self._refresh_buffer:
            return self._access_token
        async with self._lock:
            if self._access_token and time.monotonic() < self._expires_at - self._refresh_buffer:
                return self._access_token
            await self._do_refresh()
            return self._access_token

    async def _do_refresh(self) -> None:
        if not self._refresh_token:
            raise RuntimeError("No ML_REFRESH_TOKEN available — complete OAuth flow first via /auth/url")
        client_id     = ML_CLIENT_ID
        client_secret = ML_CLIENT_SECRET
        async with httpx.AsyncClient(timeout=_ML_TIMEOUT) as client:
            resp = await client.post(ML_TOKEN_URL, data={
                "grant_type":    "refresh_token",
                "client_id":     client_id,
                "client_secret": client_secret,
                "refresh_token": self._refresh_token,
            })
        if resp.status_code != 200:
            logger.error("Token refresh failed (%s): %s", resp.status_code, resp.text[:500])
            raise RuntimeError(f"Token refresh failed: {resp.status_code}")
        data = resp.json()
        self.set_tokens(
            data["access_token"],
            data.get("refresh_token") or self._refresh_token,
            data.get("expires_in", 21600),
        )
        logger.warning("ML token refreshed. Persist the rotated refresh token in Railway variables.")

    async def force_refresh(self) -> None:
        """Força renovação depois de uma resposta 401 do Mercado Livre."""
        async with self._lock:
            self._expires_at = 0.0
            await self._do_refresh()


_token_manager = TokenManager(TOKEN_REFRESH_BUFFER)


class MPTokenManager:
    """Obtém token próprio do Mercado Pago ou usa OAuth autorizado explicitamente."""

    def __init__(self, refresh_buffer: int = 1800):
        self._refresh_buffer = refresh_buffer
        self._access_token = MP_ACCESS_TOKEN
        self._refresh_token = MP_REFRESH_TOKEN
        self._expires_at = time.monotonic() + 21600 if self._access_token else 0.0
        self._lock = asyncio.Lock()

    def set_tokens(self, access_token: str, refresh_token: str = "", expires_in: int = 21600) -> None:
        if not access_token:
            raise RuntimeError("Mercado Pago não retornou access_token")
        self._access_token = access_token
        if refresh_token:
            self._refresh_token = refresh_token
        self._expires_at = time.monotonic() + max(int(expires_in or 21600), 60)

    async def get_token(self) -> str:
        if self._access_token and time.monotonic() < self._expires_at - self._refresh_buffer:
            return self._access_token
        async with self._lock:
            if self._access_token and time.monotonic() < self._expires_at - self._refresh_buffer:
                return self._access_token
            await self._obtain_token()
            return self._access_token

    async def _obtain_token(self) -> None:
        if not MP_CLIENT_ID or not MP_CLIENT_SECRET:
            raise RuntimeError("MP_CLIENT_ID e MP_CLIENT_SECRET precisam estar configurados")

        if MP_AUTH_MODE == "client_credentials":
            payload = {
                "client_id": MP_CLIENT_ID,
                "client_secret": MP_CLIENT_SECRET,
                "grant_type": "client_credentials",
            }
        elif MP_AUTH_MODE in {"authorization_code", "oauth"}:
            if not self._refresh_token:
                raise RuntimeError("MP_REFRESH_TOKEN ausente para o modo authorization_code")
            payload = {
                "client_id": MP_CLIENT_ID,
                "client_secret": MP_CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
            }
        else:
            raise RuntimeError("MP_AUTH_MODE inválido; use client_credentials ou authorization_code")

        async with httpx.AsyncClient(timeout=_MP_TIMEOUT) as client:
            resp = await client.post(
                MP_TOKEN_URL,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                json=payload,
            )
        if resp.status_code != 200:
            logger.error("Mercado Pago token failed: status=%s body=%s", resp.status_code, resp.text[:500])
            raise RuntimeError(f"Mercado Pago token failed: HTTP {resp.status_code}")
        data = resp.json()
        self.set_tokens(data.get("access_token", ""), data.get("refresh_token", ""), data.get("expires_in", 21600))

_mp_token_manager = MPTokenManager(TOKEN_REFRESH_BUFFER)

async def _mp_get(path: str, params: dict | None = None) -> Any:
    token = await _mp_token_manager.get_token()
    async with httpx.AsyncClient(timeout=_MP_TIMEOUT) as client:
        resp = await client.get(
            f"{MP_API_BASE}{path}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params=params or {},
        )
    resp.raise_for_status()
    return resp.json()

async def _mp_post(path: str, body: dict) -> Any:
    token = await _mp_token_manager.get_token()
    async with httpx.AsyncClient(timeout=_MP_TIMEOUT) as client:
        resp = await client.post(
            f"{MP_API_BASE}{path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=body,
        )
    if resp.status_code not in {200, 201, 202, 203, 204}:
        raise RuntimeError(f"Mercado Pago API HTTP {resp.status_code}: {resp.text[:500]}")
    if not resp.content:
        return {"status_code": resp.status_code}
    try:
        return resp.json()
    except ValueError:
        return {"status_code": resp.status_code, "content": resp.text}

async def _mp_download(path: str) -> dict:
    token = await _mp_token_manager.get_token()
    async with httpx.AsyncClient(timeout=_MP_TIMEOUT) as client:
        resp = await client.get(f"{MP_API_BASE}{path}", headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "application/octet-stream")
    if "text" in content_type or "csv" in content_type:
        return {"content_type": content_type, "content": resp.text}
    import base64
    return {"content_type": content_type, "content_base64": base64.b64encode(resp.content).decode("ascii")}

# ---------------------------------------------------------------------------
# Cache del ID de usuario (evita llamadas repetidas a /users/me)
# ---------------------------------------------------------------------------

_user_id: Optional[int] = None
_user_id_lock = asyncio.Lock()


async def _get_user_id() -> int:
    global _user_id
    if _user_id is not None:
        return _user_id
    async with _user_id_lock:
        if _user_id is not None:
            return _user_id
        token = await _token_manager.get_token()
        async with httpx.AsyncClient(timeout=_ML_TIMEOUT) as client:
            resp = await client.get(
                f"{ML_API_BASE}/users/me",
                headers={"Authorization": f"Bearer {token}"},
            )
        resp.raise_for_status()
        _user_id = resp.json()["id"]
        return _user_id


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def _error(detail: str, public_msg: str = "An error occurred") -> dict:
    logger.error("ml_error detail=%s", detail)
    return {"error": public_msg}


async def _ml_get(path: str, params: dict | None = None) -> Any:
    async with httpx.AsyncClient(timeout=_ML_TIMEOUT) as client:
        for attempt in range(2):
            token = await _token_manager.get_token()
            resp = await client.get(
                f"{ML_API_BASE}{path}",
                headers={"Authorization": f"Bearer {token}"},
                params=params or {},
            )
            if resp.status_code == 401 and attempt == 0 and _token_manager._refresh_token:
                await _token_manager.force_refresh()
                continue
            break
    resp.raise_for_status()
    return resp.json()


async def _ml_post(path: str, body: dict) -> Any:
    async with httpx.AsyncClient(timeout=_ML_TIMEOUT) as client:
        for attempt in range(2):
            token = await _token_manager.get_token()
            resp = await client.post(
                f"{ML_API_BASE}{path}",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body,
            )
            if resp.status_code == 401 and attempt == 0 and _token_manager._refresh_token:
                await _token_manager.force_refresh()
                continue
            break
    resp.raise_for_status()
    return resp.json()


async def _ml_put(path: str, body: dict) -> Any:
    async with httpx.AsyncClient(timeout=_ML_TIMEOUT) as client:
        for attempt in range(2):
            token = await _token_manager.get_token()
            resp = await client.put(
                f"{ML_API_BASE}{path}",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body,
            )
            if resp.status_code == 401 and attempt == 0 and _token_manager._refresh_token:
                await _token_manager.force_refresh()
                continue
            break
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Modelos de entrada
# ---------------------------------------------------------------------------

class CreateItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title:              str            = Field(..., min_length=1, max_length=60)
    category_id:        str            = Field(..., pattern=rf"^{ML_SITE}\d+$")
    price:              float          = Field(..., gt=0)
    currency_id:        str            = Field("BRL", pattern=r"^[A-Z]{3}$")
    available_quantity: int            = Field(..., ge=0)
    buying_mode:        str            = Field("buy_it_now")
    listing_type_id:    str            = Field("gold_special")
    condition:          str            = Field("new")
    description:        Optional[str]  = None
    pictures:           list[dict]     = Field(default_factory=list)
    attributes:         list[dict]     = Field(default_factory=list)

    @field_validator("description", mode="before")
    @classmethod
    def sanitize_description(cls, v):
        if v is None:
            return v
        # Elimina todo el HTML — las descripciones de ML solo aceptan texto plano vía el endpoint /description.
        return nh3.clean(v, tags=set())

    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, v):
        return str(v).strip()


class UpdateItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title:              Optional[str]        = None
    price:              Optional[float]      = Field(None, gt=0)
    available_quantity: Optional[int]        = Field(None, ge=0)
    description:        Optional[str]        = None
    pictures:           Optional[list[dict]] = None
    attributes:         Optional[list[dict]] = None

    @field_validator("description", mode="before")
    @classmethod
    def sanitize_description(cls, v):
        if v is None:
            return v
        return nh3.clean(v, tags=set())


# ---------------------------------------------------------------------------
# Servidor MCP
# ---------------------------------------------------------------------------

mcp = FastMCP("mercadolibre-mcp", host="0.0.0.0", port=PORT, json_response=True)


@mcp.tool()
async def ml_get_my_user() -> dict:
    """Obtiene el perfil de usuario del vendedor autenticado."""
    try:
        return await _ml_get("/users/me")
    except Exception as e:
        return _error(str(e), "Failed to get user profile")


@mcp.tool()
async def ml_list_items(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """
    Lista las publicaciones/artículos del vendedor.

    status: active | paused | closed | under_review (omitir para todos)
    limit: 1–50
    offset: desplazamiento de paginación
    """
    try:
        user_id = await _get_user_id()
        params: dict = {"limit": min(limit, 50), "offset": offset}
        if status:
            params["status"] = status
        search   = await _ml_get(f"/users/{user_id}/items/search", params)
        item_ids: list[str] = search.get("results", [])
        if not item_ids:
            return {"items": [], "paging": search.get("paging", {})}

        # Trae los detalles de los ítems en lotes — la búsqueda de ML solo devuelve IDs.
        batch_size = 20
        items = []
        for i in range(0, len(item_ids), batch_size):
            chunk = item_ids[i:i + batch_size]
            batch = await _ml_get("/items", {"ids": ",".join(chunk)})
            items.extend(entry["body"] for entry in batch if entry.get("code") == 200)

        return {"items": items, "paging": search.get("paging", {})}
    except Exception as e:
        return _error(str(e), "Failed to list items")


@mcp.tool()
async def ml_get_item(item_id: str) -> dict:
    """Obtiene un ítem por su ID de ML (por ej. MLA123456789)."""
    if not _RE_ITEM_ID.match(item_id):
        return {"error": f"Invalid item_id format (expected {ML_SITE} + digits)"}
    try:
        return await _ml_get(f"/items/{item_id}")
    except Exception as e:
        return _error(str(e), "Failed to get item")


@mcp.tool()
async def ml_get_item_description(item_id: str) -> dict:
    """Obtiene la descripción completa de un ítem (texto plano)."""
    if not _RE_ITEM_ID.match(item_id):
        return {"error": f"Invalid item_id format (expected {ML_SITE} + digits)"}
    try:
        return await _ml_get(f"/items/{item_id}/description")
    except Exception as e:
        return _error(str(e), "Failed to get item description")


@mcp.tool()
async def ml_create_item(
    title: str,
    category_id: str,
    price: float,
    available_quantity: int,
    currency_id: str = "BRL",
    buying_mode: str = "buy_it_now",
    listing_type_id: str = "gold_special",
    condition: str = "new",
    description: Optional[str] = None,
    pictures: Optional[list[dict]] = None,
    attributes: Optional[list[dict]] = None,
) -> dict:
    """
    Crea una nueva publicación de producto.

    pictures: lista de dicts {"source": "https://..."}
    attributes: lista de dicts {"id": "BRAND", "value_name": "..."}
    listing_type_id: free | bronze | silver | gold | gold_special | gold_premium | gold_pro
    """
    try:
        data = CreateItemInput(
            title=title,
            category_id=category_id,
            price=price,
            currency_id=currency_id,
            available_quantity=available_quantity,
            buying_mode=buying_mode,
            listing_type_id=listing_type_id,
            condition=condition,
            description=description,
            pictures=pictures or [],
            attributes=attributes or [],
        )
        body = data.model_dump(exclude={"description"})
        item = await _ml_post("/items", body)
        item_id = item.get("id")

        if item_id and data.description:
            try:
                await _ml_put(f"/items/{item_id}/description",
                              {"plain_text": data.description})
            except Exception as desc_err:
                logger.error("Failed to set description for %s: %s", item_id, desc_err)

        logger.info("AUDIT create_item item_id=%s", item_id)
        return item
    except Exception as e:
        return _error(str(e), "Failed to create item")


@mcp.tool()
async def ml_update_item(
    item_id: str,
    title: Optional[str] = None,
    price: Optional[float] = None,
    available_quantity: Optional[int] = None,
    description: Optional[str] = None,
    pictures: Optional[list[dict]] = None,
    attributes: Optional[list[dict]] = None,
) -> dict:
    """Actualiza una publicación existente. Solo se modifican los campos provistos."""
    if not _RE_ITEM_ID.match(item_id):
        return {"error": f"Invalid item_id format (expected {ML_SITE} + digits)"}
    try:
        data = UpdateItemInput(
            title=title,
            price=price,
            available_quantity=available_quantity,
            description=description,
            pictures=pictures,
            attributes=attributes,
        )
        body   = data.model_dump(exclude={"description"}, exclude_none=True)
        result: dict = {}
        if body:
            result = await _ml_put(f"/items/{item_id}", body)

        if data.description is not None:
            try:
                await _ml_put(f"/items/{item_id}/description",
                              {"plain_text": data.description})
            except Exception as desc_err:
                logger.error("Failed to update description for %s: %s", item_id, desc_err)

        logger.info("AUDIT update_item item_id=%s", item_id)
        return result or {"updated": True}
    except Exception as e:
        return _error(str(e), "Failed to update item")


@mcp.tool()
async def ml_change_item_status(item_id: str, status: str) -> dict:
    """
    Cambia el estado de una publicación.

    status: active | paused | closed
    """
    if not _RE_ITEM_ID.match(item_id):
        return {"error": f"Invalid item_id format (expected {ML_SITE} + digits)"}
    allowed = {"active", "paused", "closed"}
    if status not in allowed:
        return {"error": f"status must be one of: {', '.join(sorted(allowed))}"}
    try:
        result = await _ml_put(f"/items/{item_id}", {"status": status})
        logger.info("AUDIT change_item_status item_id=%s status=%s", item_id, status)
        return result
    except Exception as e:
        return _error(str(e), "Failed to change item status")


@mcp.tool()
async def ml_list_orders(
    status: Optional[str] = None,
    sort: str = "date_desc",
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """
    Lista los pedidos del vendedor.

    status: paid | pending | cancelled (omitir para todos)
    sort: date_desc | date_asc
    """
    if sort not in {"date_desc", "date_asc"}:
        return {"error": "sort must be date_desc or date_asc"}
    try:
        user_id = await _get_user_id()
        params: dict = {
            "seller": user_id,
            "sort":   sort,
            "limit":  min(limit, 50),
            "offset": offset,
        }
        if status:
            params["order.status"] = status
        return await _ml_get("/orders/search", params)
    except Exception as e:
        return _error(str(e), "Failed to list orders")


@mcp.tool()
async def ml_sales_summary(
    date_from: str = "2026-09-01",
    date_to: str = "2026-09-28",
) -> dict:
    """
    Gera um resumo consolidado de vendas por período.

    Inclui pedidos pagos, unidades vendidas, receita dos itens, valor pago,
    taxas, descontos/reembolsos e um detalhamento por anúncio. Os pedidos são
    deduplicados pelo ID e pagamentos múltiplos do mesmo pedido não são somados
    duas vezes.
    """
    try:
        start = date.fromisoformat(date_from)
        end = date.fromisoformat(date_to)
        if end < start:
            return {"error": "date_to must be on or after date_from"}
    except ValueError:
        return {"error": "date_from and date_to must use YYYY-MM-DD"}

    try:
        user_id = await _get_user_id()
        orders_by_id: dict[str, dict] = {}
        offset = 0
        page_size = 50

        # The API filter narrows the response; the local date check below keeps
        # the result correct even if ML interprets the boundary in UTC.
        api_params = {
            "seller": user_id,
            "order.status": "paid",
            "sort": "date_asc",
            "limit": page_size,
            "order.date_created.from": f"{date_from}T00:00:00.000-03:00",
            "order.date_created.to": f"{date_to}T23:59:59.999-03:00",
        }

        while True:
            params = {**api_params, "offset": offset}
            page = await _ml_get("/orders/search", params)
            results = page.get("results", [])
            if not results:
                break

            for order in results:
                order_id = str(order.get("id", ""))
                created = str(order.get("date_created", ""))[:10]
                if order_id and date_from <= created <= date_to:
                    orders_by_id[order_id] = order

            paging = page.get("paging", {})
            total = paging.get("total")
            offset += len(results)
            if len(results) < page_size or (isinstance(total, int) and offset >= total):
                break
            if offset >= 10000:
                logger.warning("Sales summary pagination capped at 10000 orders")
                break

        orders = list(orders_by_id.values())
        item_totals: dict[str, dict] = {}
        units_sold = 0
        item_revenue = 0.0
        list_price_revenue = 0.0
        paid_amount = 0.0
        refunded_amount = 0.0
        sale_fees = 0.0
        refunded_orders = 0
        orders_with_refund = 0

        for order in orders:
            order_refund = 0.0
            for payment in order.get("payments", []) or []:
                order_refund += float(payment.get("transaction_amount_refunded") or 0)
            refunded_amount += order_refund
            if order_refund > 0:
                orders_with_refund += 1

            order_paid = float(order.get("paid_amount") or 0)
            if order_paid > 0 and order_refund >= order_paid:
                refunded_orders += 1
            paid_amount += max(order_paid - order_refund, 0.0)

            for line in order.get("order_items", []) or []:
                item = line.get("item", {}) or {}
                item_id = str(item.get("id", "unknown"))
                title = item.get("title", item_id)
                quantity = int(line.get("quantity") or 0)
                unit_price = float(line.get("unit_price") or 0)
                gross_price = float(line.get("gross_price") or unit_price)
                fee = float(line.get("sale_fee") or 0)
                units_sold += quantity
                item_revenue += unit_price * quantity
                list_price_revenue += gross_price * quantity
                sale_fees += fee

                bucket = item_totals.setdefault(
                    item_id,
                    {
                        "item_id": item_id,
                        "title": title,
                        "units": 0,
                        "item_revenue": 0.0,
                        "list_price_revenue": 0.0,
                        "sale_fees": 0.0,
                    },
                )
                bucket["units"] += quantity
                bucket["item_revenue"] += unit_price * quantity
                bucket["list_price_revenue"] += gross_price * quantity
                bucket["sale_fees"] += fee

        def money(value: float) -> float:
            return round(value + 1e-9, 2)

        for bucket in item_totals.values():
            bucket["item_revenue"] = money(bucket["item_revenue"])
            bucket["list_price_revenue"] = money(bucket["list_price_revenue"])
            bucket["sale_fees"] = money(bucket["sale_fees"])

        return {
            "period": {
                "date_from": date_from,
                "date_to": date_to,
                "timezone": "America/Sao_Paulo",
            },
            "currency": "BRL",
            "orders": len(orders),
            "units_sold": units_sold,
            "item_revenue": money(item_revenue),
            "list_price_revenue": money(list_price_revenue),
            "paid_amount_net_of_refunds": money(paid_amount),
            "refunded_amount": money(refunded_amount),
            "sale_fees": money(sale_fees),
            "orders_with_refund": orders_with_refund,
            "fully_refunded_orders": refunded_orders,
            "items": sorted(item_totals.values(), key=lambda row: row["item_revenue"], reverse=True),
        }
    except Exception as e:
        return _error(str(e), "Failed to generate sales summary")


@mcp.tool()
async def ml_get_order(order_id: str) -> dict:
    """Obtiene un pedido por su ID."""
    if not _RE_NUMERIC_ID.match(order_id):
        return {"error": "Invalid order_id format (digits only)"}
    try:
        return await _ml_get(f"/orders/{order_id}")
    except Exception as e:
        return _error(str(e), "Failed to get order")


@mcp.tool()
async def ml_get_shipment(shipment_id: str) -> dict:
    """Obtiene los detalles de envío de un despacho."""
    if not _RE_NUMERIC_ID.match(shipment_id):
        return {"error": "Invalid shipment_id format (digits only)"}
    try:
        return await _ml_get(f"/shipments/{shipment_id}")
    except Exception as e:
        return _error(str(e), "Failed to get shipment")


@mcp.tool()
async def ml_get_shipment_label(shipment_id: str, format: str = "zpl2") -> dict:
    """
    Obtiene una etiqueta de envío para un despacho.

    format: zpl2 | pdf
    """
    if not _RE_NUMERIC_ID.match(shipment_id):
        return {"error": "Invalid shipment_id format (digits only)"}
    if format not in {"zpl2", "pdf"}:
        return {"error": "format must be zpl2 or pdf"}
    try:
        return await _ml_get(f"/shipments/{shipment_id}/labels",
                             {"response_type": format})
    except Exception as e:
        return _error(str(e), "Failed to get shipment label")


@mcp.tool()
async def ml_predict_category(query: str) -> dict:
    """
    Predice la mejor categoría de MercadoLibre para una descripción de producto.

    Devuelve la categoría mejor predicha junto con su esquema de atributos.
    """
    try:
        result = await _ml_get(
            f"/sites/{ML_SITE}/domain_discovery/search",
            {"q": query, "limit": 3},
        )
        return {"predictions": result}
    except Exception as e:
        return _error(str(e), "Failed to predict category")


@mcp.tool()
async def ml_get_category_attributes(category_id: str) -> dict:
    """Obtiene los atributos requeridos y opcionales de una categoría (por ej. MLA1234)."""
    try:
        return await _ml_get(f"/categories/{category_id}/attributes")
    except Exception as e:
        return _error(str(e), "Failed to get category attributes")


# ---------------------------------------------------------------------------
# Mercado Pago: leitura financeira
# ---------------------------------------------------------------------------

@mcp.tool()
async def mp_get_my_user() -> dict:
    """Consulta o usuário da conta Mercado Pago autenticada."""
    try:
        return await _mp_get("/users/me")
    except Exception as e:
        return _error(str(e), "Failed to get Mercado Pago user")

@mcp.tool()
async def mp_get_payment(payment_id: str) -> dict:
    """Consulta um pagamento Mercado Pago por ID."""
    if not _RE_NUMERIC_ID.match(str(payment_id)):
        return {"error": "payment_id must contain digits only"}
    try:
        return await _mp_get(f"/v1/payments/{payment_id}")
    except Exception as e:
        return _error(str(e), "Failed to get Mercado Pago payment")

@mcp.tool()
async def mp_search_payments(
    external_reference: Optional[str] = None,
    status: Optional[str] = None,
    begin_date: str = "NOW-30DAYS",
    end_date: str = "NOW",
    range_field: str = "date_created",
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Pesquisa pagamentos aprovados ou outros movimentos de pagamento."""
    allowed_ranges = {"date_created", "date_last_updated", "date_approved", "money_release_date"}
    if range_field not in allowed_ranges:
        return {"error": f"range_field must be one of: {', '.join(sorted(allowed_ranges))}"}
    params = {
        "sort": range_field,
        "criteria": "desc",
        "range": range_field,
        "begin_date": begin_date,
        "end_date": end_date,
        "limit": min(max(limit, 1), 50),
        "offset": max(offset, 0),
    }
    if external_reference:
        params["external_reference"] = external_reference
    if status:
        params["status"] = status
    try:
        return await _mp_get("/v1/payments/search", params)
    except Exception as e:
        return _error(str(e), "Failed to search Mercado Pago payments")

async def _mp_generate_report(report_type: str, begin_date: str, end_date: str) -> dict:
    if report_type not in {"settlement", "release"}:
        raise ValueError("invalid report type")
    return await _mp_post(
        f"/v1/account/{report_type}_report",
        {"begin_date": begin_date, "end_date": end_date},
    )

@mcp.tool()
async def mp_generate_account_money_report(begin_date: str, end_date: str) -> dict:
    """Gera relatório Account Money, com operações, taxas e impacto líquido."""
    try:
        return await _mp_generate_report("settlement", begin_date, end_date)
    except Exception as e:
        return _error(str(e), "Failed to generate Account Money report")

@mcp.tool()
async def mp_list_account_money_reports() -> dict:
    """Lista relatórios Account Money já gerados."""
    try:
        return {"reports": await _mp_get("/v1/account/settlement_report/list")}
    except Exception as e:
        return _error(str(e), "Failed to list Account Money reports")

@mcp.tool()
async def mp_download_account_money_report(file_name: str) -> dict:
    """Baixa um relatório Account Money pelo nome retornado pela API."""
    if not file_name or "/" in file_name or "\\" in file_name:
        return {"error": "file_name inválido"}
    try:
        return await _mp_download(f"/v1/account/settlement_report/{file_name}")
    except Exception as e:
        return _error(str(e), "Failed to download Account Money report")

@mcp.tool()
async def mp_generate_released_money_report(begin_date: str, end_date: str) -> dict:
    """Gera relatório Released Money, com liberações, taxas e datas de recebimento."""
    try:
        return await _mp_generate_report("release", begin_date, end_date)
    except Exception as e:
        return _error(str(e), "Failed to generate Released Money report")

@mcp.tool()
async def mp_list_released_money_reports() -> dict:
    """Lista relatórios Released Money já gerados."""
    try:
        return {"reports": await _mp_get("/v1/account/release_report/list")}
    except Exception as e:
        return _error(str(e), "Failed to list Released Money reports")

@mcp.tool()
async def mp_download_released_money_report(file_name: str) -> dict:
    """Baixa um relatório Released Money pelo nome retornado pela API."""
    if not file_name or "/" in file_name or "\\" in file_name:
        return {"error": "file_name inválido"}
    try:
        return await _mp_download(f"/v1/account/release_report/{file_name}")
    except Exception as e:
        return _error(str(e), "Failed to download Released Money report")


# ---------------------------------------------------------------------------
# Middleware de autenticación
# ---------------------------------------------------------------------------
from datetime import date, timedelta

async def _ml_get_api(path: str, params: dict | None = None, extra_headers: dict | None = None) -> Any:
    async with httpx.AsyncClient(timeout=_ML_TIMEOUT) as client:
        for attempt in range(2):
            token = await _token_manager.get_token()
            headers = {"Authorization": f"Bearer {token}"}
            if extra_headers:
                headers.update(extra_headers)
            resp = await client.get(f"{ML_API_BASE}{path}", headers=headers, params=params or {})
            if resp.status_code == 401 and attempt == 0 and _token_manager._refresh_token:
                await _token_manager.force_refresh()
                continue
            break
    resp.raise_for_status()
    return resp.json()

@mcp.tool()
async def ml_list_product_ads_campaigns(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Lista campanhas Product Ads e métricas de impressões, cliques, CTR, gasto e vendas."""
    try:
        today = date.today()
        date_to = date_to or today.isoformat()
        date_from = date_from or (today - timedelta(days=29)).isoformat()

        advertisers_raw = await _ml_get_api(
            "/advertising/advertisers",
            {"product_id": "PADS"},
            {"Api-Version": "2"},
        )
        advertisers = (
            advertisers_raw
            if isinstance(advertisers_raw, list)
            else advertisers_raw.get(
                "advertisers",
                advertisers_raw.get("results", []),
            )
        )

        if not advertisers:
            return {"advertisers": [], "campaigns": [], "message": "Nenhum anunciante Product Ads encontrado."}

        advertiser = advertisers[0]
        advertiser_id = advertiser.get("id") or advertiser.get("advertiser_id")
        advertiser_site_id = advertiser.get("site_id") or ML_SITE
        if not advertiser_id:
            return {"error": "O Mercado Livre não retornou um advertiser_id."}

        params = {
            "limit": min(max(limit, 1), 50),
            "offset": max(offset, 0),
            "date_from": date_from,
            "date_to": date_to,
            "metrics": (
                "clicks,prints,ctr,cost,cpc,acos,cvr,roas,"
                "units_quantity,total_amount"
            ),
            "metrics_summary": "true",
        }

        campaigns = await _ml_get_api(
            f"/advertising/{advertiser_site_id}/advertisers/{advertiser_id}/product_ads/campaigns/search",
            params,
            {"Api-Version": "2"},
        )

        return {
            "advertiser_id": advertiser_id,
            "advertiser_site_id": advertiser_site_id,
            "date_from": date_from,
            "date_to": date_to,
            "campaigns": campaigns,
        }
    except Exception as e:
        return _error(str(e), "Failed to list Product Ads campaigns")

@mcp.tool()
async def ml_get_item_promotions(item_id: str) -> dict:
    """Consulta promoções e eventual participação financeira do Mercado Livre para um anúncio."""
    if not _RE_ITEM_ID.match(item_id):
        return {"error": f"Invalid item_id format (expected {ML_SITE} + digits)"}
    try:
        return await _ml_get_api(
            f"/seller-promotions/items/{item_id}",
            {"app_version": "v2"},
        )
    except Exception as e:
        return _error(str(e), "Failed to get item promotions")
        
class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        ip = _client_ip(request)

        # Limita la tasa en todas las rutas — las verificaciones de auth ocurren más abajo, por ruta.
        if not await _check_rate_limit(ip):
            return JSONResponse({"error": "Too many requests"}, status_code=429)

        # Callback de OAuth — intercambia el código de autorización por tokens.
        if path == "/auth/callback":
            code = request.query_params.get("code", "")
            if not code:
                return HTMLResponse("Missing code parameter", status_code=400)
            client_id     = os.environ.get("ML_CLIENT_ID", "")
            client_secret = os.environ.get("ML_CLIENT_SECRET", "")
            redirect_uri  = os.environ.get("ML_REDIRECT_URI", "")
            try:
                async with httpx.AsyncClient(timeout=_ML_TIMEOUT) as client:
                    resp = await client.post(ML_TOKEN_URL, data={
                        "grant_type":    "authorization_code",
                        "client_id":     client_id,
                        "client_secret": client_secret,
                        "code":          code,
                        "redirect_uri":  redirect_uri,
                    })
                if resp.status_code != 200:
                    logger.error(
                        "OAuth token exchange failed: status=%s redirect_uri=%s body=%s",
                        resp.status_code, redirect_uri, resp.text,
                    )
                    hint = ""
                    try:
                        err = resp.json().get("error", "")
                        if err == "invalid_grant":
                            hint = (
                                "<p><strong>Hint:</strong> <code>invalid_grant</code> usually means "
                                "<code>ML_REDIRECT_URI</code> does not exactly match the URI registered "
                                "in your MercadoLibre app. Check for trailing slashes, http vs https, "
                                f"and case. Current value: <code>{redirect_uri}</code></p>"
                            )
                    except Exception:
                        pass
                    return HTMLResponse(
                        f"<h2>Token exchange failed (HTTP {resp.status_code})</h2>{hint}"
                        "<p>Check server logs for details.</p>",
                        status_code=502,
                    )
                data = resp.json()
                _token_manager.set_tokens(
                    data["access_token"],
                    data["refresh_token"],
                    data.get("expires_in", 21600),
                )
                logger.warning(
                    "OAuth complete. Save these in your environment — "
                    "ML_ACCESS_TOKEN=%s ML_REFRESH_TOKEN=%s",
                    data["access_token"], data["refresh_token"],
                )
                return HTMLResponse(
                    "<h2>Authorization complete.</h2>"
                    "<p>Copy the tokens from your server logs and set them as "
                    "<code>ML_ACCESS_TOKEN</code> and <code>ML_REFRESH_TOKEN</code> "
                    "environment variables so they survive restarts.</p>"
                )
            except Exception as e:
                logger.error("OAuth callback error: %s", e)
                return HTMLResponse("Internal error. Check server logs.", status_code=500)

        # Punto de entrada de OAuth — redirige al usuario a la pantalla de consentimiento de ML.
        if path == "/auth/url":
            client_id    = os.environ.get("ML_CLIENT_ID", "")
            redirect_uri = os.environ.get("ML_REDIRECT_URI", "")
            if not client_id or not redirect_uri:
                return JSONResponse(
                    {"error": "ML_CLIENT_ID and ML_REDIRECT_URI must be set"},
                    status_code=500,
                )
            params = urlencode({
                "response_type": "code",
                "client_id":     client_id,
                "redirect_uri":  redirect_uri,
            })
            url = f"{ML_AUTH_BASE}/authorization?{params}"
            return JSONResponse({"auth_url": url, "redirect_uri": redirect_uri})

        # Chequeo de salud — sin autenticación.
        if path == "/health":
            return JSONResponse({"status": "ok"})

        # Descubrimiento de metadatos de OAuth — Claude.ai consulta estas rutas sin token antes
        # de recurrir a la autenticación bearer. Se dejan pasar hacia el SDK de MCP.
        if path.startswith("/.well-known/"):
            return await call_next(request)

        # Verifica el tamaño del cuerpo antes de autenticar, para evitar leer cuerpos enormes.
        content_length = int(request.headers.get("content-length", 0))
        if content_length > MAX_REQUEST_BODY:
            return JSONResponse({"error": "Request body too large"}, status_code=413)

        token: str = ""
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:]
        elif ALLOW_TOKEN_QUERY_PARAM:
            token = request.query_params.get("token", "")

        if not token or not secrets.compare_digest(token.encode(), BEARER_TOKEN.encode()):
            logger.warning("Unauthorized attempt from %s %s %s", ip, request.method, path)
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)


class _HostRewriteMiddleware:
    """Reescribe el header Host a 'localhost' antes de la verificación de DNS-rebinding del SDK de MCP.
    Railway termina el TLS y valida el hostname real río arriba, así que esto es seguro."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            scope = {
                **scope,
                "headers": [
                    (b"host", b"localhost") if k == b"host" else (k, v)
                    for k, v in scope.get("headers", [])
                ],
            }
        await self.app(scope, receive, send)


# Se agrega el middleware directamente a la app de FastMCP para preservar su lifespan
# (el task group del gestor de sesiones).
app = mcp.streamable_http_app()
app.add_middleware(BearerAuthMiddleware)
# _HostRewriteMiddleware se agrega al final para que se ejecute primero (más externo) —
# debe reescribir el header antes de la autenticación.
app.add_middleware(_HostRewriteMiddleware)

# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    if not ML_CLIENT_ID:
        logger.warning("ML_CLIENT_ID is not set")
    if ALLOW_TOKEN_QUERY_PARAM:
        logger.info("Token query param: ENABLED")
    else:
        logger.info("Token query param: disabled")

    logger.info("Bearer auth: %s", "ENABLED" if BEARER_TOKEN else "DISABLED")
    logger.info("MCP endpoint: http://0.0.0.0:%d/mcp", PORT)
    logger.info("Mercado Pago auth mode: %s", MP_AUTH_MODE)

    # Deshabilita el access log de uvicorn cuando ?token= está activo — de lo contrario
    # registraría la URL completa, incluyendo el token.
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=not ALLOW_TOKEN_QUERY_PARAM)
