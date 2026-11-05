"""main.py's own wiring of the remote-access and multi-GPU fixes (utils/security.py, utils/launch.py).

tests/test_security.py and tests/test_launch.py cover the helpers; these import the real main.py in a child Python
(importing it runs everything but server.run(), which only runs as __main__) and check that it:
- refuses to start on a non-loopback host without LETS_BACKEND_TOKEN (exit code 2);
- binds 127.0.0.1 by default and registers the request guard (a foreign Host header gets 403);
- refuses /start_tunnel_service without a token;
- with a token, binds the configured host and needs 'Authorization: Bearer <token>';
- answers bad multi-GPU settings of /train with 400 and builds the launch from the default Accelerate config.
Nothing listens on a port, no tunnel and no training process starts (Popen and create_tunnel are replaced in the
child), the child runs offline and without GPUs.

Run from the backend folder with the training venv (skipped without the backend's packages):

    sd_scripts\\venv\\Scripts\\python -m unittest tests.test_main_wiring
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from utils.security import TOKEN_ENV  # noqa: E402

HAS_BACKEND = all(
    importlib.util.find_spec(name) is not None
    for name in ("library", "LoraEasyCustomOptimizer", "PIL", "starlette", "httpx", "uvicorn", "transformers")
)
TOKEN = "test-token-1234"

DRIVER = r"""
import json, subprocess, sys
from pathlib import Path

backend, workdir, mode, *token = sys.argv[1:]
sys.path.insert(0, backend)
sys.argv = [str(Path(backend, "main.py")), workdir]  # python main.py <workdir>
import main  # exits with the refusal, or sets the app up (server.run() is only for __main__)

launched, tunnels = [], []


class NotStarted:
    def poll(self):
        return 0

    def wait(self):
        return 0


def popen(cmd, *args, **kwargs):
    launched.append([str(part) for part in cmd])
    return NotStarted()


def create_tunnel(*args, **kwargs):
    tunnels.append(True)
    raise RuntimeError("no tunnels in tests")


subprocess.Popen = popen
main.create_tunnel = create_tunnel

from starlette.testclient import TestClient

client = TestClient(main.app, base_url="http://127.0.0.1:8000", raise_server_exceptions=False)


def get(path, **kwargs):
    response = client.get(path, **kwargs)
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return [response.status_code, body]


results = {"host": main.uvi_config.host}
if mode == "token":
    results["no_header"] = get("/is_training")
    results["wrong"] = get("/is_training", headers={"Authorization": "Bearer nope"})
    results["bearer"] = get("/is_training", headers={"Authorization": "Bearer " + token[0]})
elif mode == "colab":
    results["colab_train"] = get("/train")
    results["should_exit"] = bool(main.server.should_exit)
else:
    results["stop_idle"] = get("/stop_training")
    results["is_training"] = get("/is_training")
    results["foreign_host"] = get("/is_training", headers={"Host": "attacker.example"})
    results["tunnel"] = get("/start_tunnel_service")
    results["bad_number"] = get("/train?accelerate_enabled=True&accelerate_num_processes=x")
    Path("runtime_store/config.toml").write_text("", encoding="utf-8")  # as left by a /validate
    Path("runtime_store/dataset.toml").write_text("", encoding="utf-8")
    results["bad_ids"] = get("/train?accelerate_enabled=True&accelerate_num_processes=1&accelerate_gpu_ids=0,1")
    results["two_gpus"] = get("/train?accelerate_enabled=True&accelerate_num_processes=2")
results["launched"] = launched
results["tunnels"] = len(tunnels)
print("RESULT " + json.dumps(results))
"""

USER_MULTI_GPU_CONFIG = (
    "compute_environment: LOCAL_MACHINE\ndistributed_type: MULTI_GPU\ngpu_ids: 1,2\nnum_processes: 2\n"
)


@unittest.skipUnless(HAS_BACKEND, "the backend's packages (training venv) are not importable here")
class MainWiringTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.workdir = self.root / "work"
        self.workdir.mkdir()
        self.hf_home = self.root / "hf"
        (self.hf_home / "accelerate").mkdir(parents=True)
        (self.hf_home / "accelerate" / "default_config.yaml").write_text(USER_MULTI_GPU_CONFIG, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, config: dict, mode: str = "local", token: str | None = None):
        (self.workdir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if key != TOKEN_ENV}
        env.update({
            "HF_HOME": str(self.hf_home),  # the default Accelerate config below, and no cached downloads
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        })
        if token:
            env[TOKEN_ENV] = token
        args = [sys.executable, "-c", DRIVER, str(BACKEND), str(self.workdir), mode] + ([token] if token else [])
        done = subprocess.run(args, cwd=str(self.root), env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=300)
        result = None
        for line in done.stdout.splitlines():
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
        return done, result

    def test_refuses_a_non_loopback_host_without_token(self):
        done, result = self.run_main({"host": "0.0.0.0", "port": 8000, "remote": False})
        self.assertEqual(done.returncode, 2, done.stderr[-2000:])
        self.assertIsNone(result)
        self.assertIn("Refusing to start the backend", done.stderr)
        self.assertIn(TOKEN_ENV, done.stderr)

    def test_token_backend_needs_the_bearer_token(self):
        done, result = self.run_main({"host": "0.0.0.0", "port": 8000, "remote": False}, mode="token", token=TOKEN)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(result["host"], "0.0.0.0")
        self.assertEqual(result["no_header"][0], 401)
        self.assertEqual(result["wrong"][0], 401)
        self.assertEqual(result["bearer"], [200, {"training": False, "errored": False}])

    def test_local_backend(self):
        done, result = self.run_main({"remote": False, "port": 8000})
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(result["host"], "127.0.0.1")
        # /stop_training before any job: a 400, not an AttributeError (500) from the old 'not X and X.poll()' guard
        self.assertEqual(result["stop_idle"], [400, {"detail": "Not Currently Training"}])
        self.assertEqual(result["is_training"], [200, {"training": False, "errored": False}])
        self.assertEqual(result["foreign_host"][0], 403)

        status, body = result["tunnel"]
        self.assertEqual(status, 403)
        self.assertFalse(body["service_started"])
        self.assertEqual(result["tunnels"], 0)

        for key in ("bad_number", "bad_ids"):
            status, body = result[key]
            self.assertEqual(status, 400, key)
            self.assertFalse(body["training"], key)
            self.assertIn("Invalid multi-GPU settings", body["detail"], key)

        # the user's MULTI_GPU config on GPUs 1,2 (HF_HOME) decides: no --multi_gpu / --gpu_ids of the backend's own
        self.assertEqual(result["two_gpus"], [200, {"detail": "Training Started", "training": True}])
        self.assertEqual(len(result["launched"]), 1, result["launched"])
        cmd = result["launched"][0]
        self.assertEqual(cmd[1:5], ["-m", "accelerate.commands.launch", "--num_processes=2",
                                    "--main_process_port=29500"])
        self.assertTrue(cmd[5].endswith("train_network.py"), cmd)

    def test_colab_train_answers_before_the_server_stops(self):
        # the GUI treats a non-200 /train as "did not start"; Colab used to return None (500) after stopping the server
        done, result = self.run_main({"colab": True, "remote": False, "port": 8000}, mode="colab")
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        status, body = result["colab_train"]
        self.assertEqual(status, 200)
        self.assertTrue(body["training"])
        self.assertTrue(result["should_exit"])  # the shutdown still ran, after the response
        self.assertEqual(result["launched"], [])


if __name__ == "__main__":
    unittest.main()
