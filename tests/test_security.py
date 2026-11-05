"""Who may use the backend's API (utils/security.py): bind host, the remote token, request checks.

The backend listened on 0.0.0.0 with no authentication, so anyone on the LAN (or behind a tunnel) could start jobs
and read files through /validate. Now:
- it binds 127.0.0.1 unless config.json "host" or --host names another address;
- when other machines can reach it (a non-loopback host, or config.json "remote" starting a tunnel) it needs the
  token from LETS_BACKEND_TOKEN and refuses to start without one;
- with a token every request needs 'Authorization: Bearer <token>' (401 otherwise);
- without one (the local default) a request must be addressed to a loopback name and must not come from a web page
  of another site (DNS rebinding, cross-site posts from the browser on the same PC).

Run from the backend folder with the backend's Python (the middleware tests need starlette and httpx):

    sd_scripts\\venv\\Scripts\\python -m unittest tests.test_security
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.security import (  # noqa: E402
    DEFAULT_HOST,
    TOKEN_ENV,
    bind_host,
    is_loopback_host,
    parse_cli,
    startup_security,
)

HAS_STARLETTE = all(importlib.util.find_spec(name) is not None for name in ("starlette", "httpx"))


class StartupTest(unittest.TestCase):
    def test_parse_cli(self):
        self.assertEqual(parse_cli([]), (None, None))
        self.assertEqual(parse_cli(["backend"]), ("backend", None))  # the GUI's launcher: main.py backend
        self.assertEqual(parse_cli(["backend", "--host", "0.0.0.0"]), ("backend", "0.0.0.0"))
        self.assertEqual(parse_cli(["--host=::", "work"]), ("work", "::"))
        with self.assertRaises(SystemExit):
            parse_cli(["--host"])

    def test_bind_host(self):
        self.assertEqual(DEFAULT_HOST, "127.0.0.1")
        self.assertEqual(bind_host({"port": 8000}, None), "127.0.0.1")
        self.assertEqual(bind_host({"host": "0.0.0.0"}, None), "0.0.0.0")
        self.assertEqual(bind_host({"host": "0.0.0.0"}, "127.0.0.1"), "127.0.0.1")  # --host wins
        self.assertEqual(bind_host({"host": ""}, None), "127.0.0.1")

    def test_is_loopback_host(self):
        for host in ("127.0.0.1", "127.0.0.2", "localhost", "LOCALHOST", "::1", "[::1]"):
            self.assertTrue(is_loopback_host(host), host)
        for host in ("0.0.0.0", "::", "192.168.0.10", "example.com", "", "127.0.0.1.example.com"):
            self.assertFalse(is_loopback_host(host), host)

    def test_local_default_needs_no_token(self):
        security = startup_security("127.0.0.1", {"remote": False, "port": 8000}, {})
        self.assertIsNone(security.refusal)
        self.assertIsNone(security.token)
        self.assertFalse(security.exposed)

    def test_non_loopback_host_without_token_refuses(self):
        security = startup_security("0.0.0.0", {}, {TOKEN_ENV: "  "})
        self.assertTrue(security.exposed)
        self.assertIn(TOKEN_ENV, security.refusal)
        self.assertIn("0.0.0.0", security.refusal)

    def test_tunnel_without_token_refuses(self):
        """colab / ngrok / cloudflared: bound to 127.0.0.1, but the tunnel puts the API on the internet."""
        security = startup_security("127.0.0.1", {"remote": True, "remote_mode": "ngrok"}, {})
        self.assertTrue(security.exposed)
        self.assertIn("ngrok", security.refusal)

    def test_remote_with_token_starts(self):
        for host, config in (("0.0.0.0", {}), ("127.0.0.1", {"remote": True, "colab": True})):
            security = startup_security(host, config, {TOKEN_ENV: " s3cret-token "})
            self.assertIsNone(security.refusal)
            self.assertEqual(security.token, "s3cret-token")
            self.assertTrue(security.exposed)

    def test_token_set_for_a_local_server_is_used(self):
        security = startup_security("127.0.0.1", {}, {TOKEN_ENV: "abc"})
        self.assertEqual(security.token, "abc")
        self.assertFalse(security.exposed)


@unittest.skipUnless(HAS_STARLETTE, "starlette / httpx are not installed in this Python")
class GuardTest(unittest.TestCase):
    def client(self, token=None, base_url="http://127.0.0.1:8000"):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from utils.security import BackendGuard

        async def is_training(_):
            return JSONResponse({"training": False, "errored": False})

        async def validate(request):
            return JSONResponse({"args": [], "dataset": {}, "tags": {}, "echo": (await request.json())})

        app = Starlette(routes=[Route("/is_training", is_training), Route("/validate", validate, methods=["POST"])])
        app.add_middleware(BackendGuard, token=token)
        return TestClient(app, base_url=base_url)

    def test_token_required(self):
        client = self.client(token="s3cret")
        response = client.get("/is_training")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("www-authenticate"), "Bearer")
        self.assertIn("detail", response.json())
        for header in ("Bearer wrong", "Basic czNjcmV0", "s3cret", "Bearer", "Bearer s3cret extra"):
            self.assertEqual(client.get("/is_training", headers={"Authorization": header}).status_code, 401, header)

    def test_token_accepted(self):
        client = self.client(token="s3cret", base_url="https://abc.trycloudflare.com")  # any Host with a token
        for header in ("Bearer s3cret", "bearer s3cret", "Bearer  s3cret "):
            response = client.get("/is_training", headers={"Authorization": header})
            self.assertEqual(response.status_code, 200, header)
            self.assertEqual(response.json(), {"training": False, "errored": False})
        response = client.post("/validate", json={"a": 1}, headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(response.json()["echo"], {"a": 1})

    def test_local_requests_from_the_gui(self):
        for base_url in ("http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1"):
            client = self.client(base_url=base_url)
            self.assertEqual(client.get("/is_training").status_code, 200, base_url)
            self.assertEqual(client.post("/validate", json={}).status_code, 200, base_url)
        # (the test client cannot take an IPv6 base URL: the Host header is given directly)
        for host in ("[::1]:8000", "0.0.0.0:8000"):  # http://0.0.0.0:8000 reaches the local host on Linux
            self.assertEqual(self.client().get("/is_training", headers={"Host": host}).status_code, 200, host)
        same_origin = {"Origin": "http://127.0.0.1:8000", "Sec-Fetch-Site": "same-origin"}
        self.assertEqual(self.client().get("/is_training", headers=same_origin).status_code, 200)

    def test_local_requests_from_other_sites_are_refused(self):
        rebinding = self.client(base_url="http://attacker.example:8000")  # DNS rebinding: Host is the attacker's
        self.assertEqual(rebinding.get("/is_training").status_code, 403)
        client = self.client()
        for headers in ({"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"},
                        {"Origin": "https://attacker.example"}, {"Origin": "null"}):
            response = client.post("/validate", json={}, headers=headers)
            self.assertEqual(response.status_code, 403, headers)
            self.assertIn("detail", response.json())


if __name__ == "__main__":
    unittest.main()
