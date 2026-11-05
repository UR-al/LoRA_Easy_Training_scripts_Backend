"""Who may use the backend's API: the bind host, the remote token and the per-request checks.

The API starts training jobs and reads dataset folders, so it is not for anyone who can reach the port:
- the server binds 127.0.0.1 (DEFAULT_HOST) unless config.json "host" or "python main.py [workdir] --host HOST"
  names another address;
- when other machines can reach it (a non-loopback host, or config.json "remote" starting an ngrok / cloudflared
  tunnel, as the colab install does) the operator sets a token in LETS_BACKEND_TOKEN, and the server refuses to
  start without one;
- with a token, every request needs the header 'Authorization: Bearer <token>' (401 otherwise); a token set for a
  local server is enforced too;
- without one (the local default, where the GUI on the same PC needs nothing) a request must be addressed to a
  loopback name (Host header: a page that rebinds its own domain to 127.0.0.1 is refused) and must not come from a
  web page (Origin / Sec-Fetch-Site of another site: a page in the browser could otherwise post to 127.0.0.1).
"""

import hmac
import ipaddress
from typing import Mapping, NamedTuple, Optional

TOKEN_ENV = "LETS_BACKEND_TOKEN"
DEFAULT_HOST = "127.0.0.1"


def parse_cli(argv) -> tuple[Optional[str], Optional[str]]:
    """'main.py [workdir] [--host HOST]' -> (workdir, host). Other options are ignored, as before."""
    workdir = host = None
    args = list(argv)
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--host":
            if i + 1 >= len(args):
                raise SystemExit("--host needs an address, e.g. --host 0.0.0.0")
            host = args[i + 1]
            i += 2
            continue
        if arg.startswith("--host="):
            host = arg.split("=", 1)[1]
        elif workdir is None and not arg.startswith("-"):
            workdir = arg
        i += 1
    return workdir, host


def bind_host(config: Mapping, cli_host: Optional[str]) -> str:
    for host in (cli_host, config.get("host")):
        if host is not None and str(host).strip():
            return str(host).strip()
    return DEFAULT_HOST


def is_loopback_host(host: str) -> bool:
    host = str(host).strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_local_host_name(host: str) -> bool:
    """A Host header name of this machine: loopback, or 0.0.0.0 / :: (Linux connects those to the local host)."""
    if is_loopback_host(host):
        return True
    try:
        return ipaddress.ip_address(host.strip().strip("[]")).is_unspecified
    except ValueError:
        return False


def _host_name(netloc: str) -> str:
    """'127.0.0.1:8000' -> '127.0.0.1', '[::1]:8000' -> '::1', 'localhost' -> 'localhost'."""
    netloc = netloc.strip()
    if netloc.startswith("["):
        return netloc[1:netloc.find("]")] if "]" in netloc else netloc
    if netloc.count(":") == 1:
        return netloc.split(":", 1)[0]
    return netloc


class Security(NamedTuple):
    token: Optional[str]  # the token every request must carry, or None
    exposed: bool  # other machines can reach the API (non-loopback bind or a tunnel)
    refusal: Optional[str]  # why the server must not start, or None


def startup_security(host: str, config: Mapping, environ: Mapping[str, str]) -> Security:
    token = (environ.get(TOKEN_ENV) or "").strip() or None
    reasons = []
    if not is_loopback_host(host):
        reasons.append(f"it listens on {host}")
    if config.get("remote", False):
        reasons.append(f"config.json starts a {config.get('remote_mode', 'cloudflared')} tunnel (\"remote\": true)")
    refusal = None
    if reasons and token is None:
        refusal = (
            "Refusing to start the backend: " + " and ".join(reasons) + ", so other machines can reach its API, but "
            f"{TOKEN_ENV} is not set. Set {TOKEN_ENV} to a long random secret before starting the backend and give "
            "the same token to the GUI (it sends 'Authorization: Bearer <token>' with every request), or run the "
            f"backend on {DEFAULT_HOST} (the default) without a tunnel."
        )
    return Security(token, bool(reasons), refusal)


def bearer_token_matches(header: Optional[str], token: str) -> bool:
    if not header:
        return False
    scheme, _, value = header.strip().partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(value.strip().encode("utf-8"), token.encode("utf-8"))


def local_request_refusal(headers: Mapping[str, str]) -> Optional[str]:
    """Why a request to the token-less local server is refused, or None (headers: lower-case names)."""
    host = headers.get("host")
    if host and not _is_local_host_name(_host_name(host)):
        return f"this backend only answers requests addressed to {DEFAULT_HOST} / localhost (got Host: {host})"
    if headers.get("sec-fetch-site", "").lower() in ("cross-site", "same-site"):
        return "requests from web pages of other sites are refused"
    origin = headers.get("origin")
    if origin is not None:
        origin_host = origin.split("://", 1)[1].split("/", 1)[0] if "://" in origin else ""
        if not is_loopback_host(_host_name(origin_host)):
            return f"requests from web pages of other sites are refused (Origin: {origin})"
    return None


class BackendGuard:
    """ASGI middleware applying the token (when set) or the local request checks to every HTTP request."""

    def __init__(self, app, token: Optional[str] = None) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from starlette.datastructures import Headers
        from starlette.responses import JSONResponse

        headers = Headers(scope=scope)
        if self.token is not None:
            if not bearer_token_matches(headers.get("authorization"), self.token):
                response = JSONResponse(
                    {"detail": f"Unauthorized: send 'Authorization: Bearer <token>' with the backend's {TOKEN_ENV}"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        else:
            refusal = local_request_refusal(headers)
            if refusal:
                await JSONResponse({"detail": f"Forbidden: {refusal}"}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)
