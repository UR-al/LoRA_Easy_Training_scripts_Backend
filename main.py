import sys
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.routing import Route
from starlette import status
import json
from utils.validation import validate
from utils.process import process_args, process_dataset_args
from utils.train_script import pick_train_script
from utils.launch import build_train_command
from utils.security import TOKEN_ENV, BackendGuard, bind_host, parse_cli, startup_security
from pathlib import Path
import subprocess
from utils.tunnel_service import CloudflaredTunnel, create_tunnel
import uvicorn
import os
from threading import Thread
import warnings
import logging

from transformers import CLIPTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="torchao")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

try:
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
except Exception:
    try:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not load CLIP tokenizer - offline and not cached. Continuing without it."
        )
        tokenizer = None


# python main.py [workdir] [--host HOST]  (the GUI's launcher runs "main.py backend")
WORKDIR, CLI_HOST = parse_cli(sys.argv[1:])
if WORKDIR:
    os.chdir(WORKDIR)

if not Path("runtime_store").exists():
    Path("runtime_store").mkdir()


async def stop_server(_: Request = None) -> JSONResponse:
    global server
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse({"detail": "training still running"})
    server.should_exit = True
    server.force_exit = True


async def start_tunnel_service(request: Request) -> JSONResponse:
    config_data = json.loads(app.state.CONFIG.read_text())
    if app.state.TUNNEL:
        return JSONResponse({"service_started": False}, status_code=409)
    if not app.state.BACKEND_TOKEN:
        # a tunnel puts the API on the internet: only for a backend started with a token
        return JSONResponse(
            {
                "service_started": False,
                "detail": f"Start the backend with {TOKEN_ENV} set before starting a tunnel",
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )
    app.state.TUNNEL = create_tunnel(config_data)
    if isinstance(app.state.TUNNEL, CloudflaredTunnel):
        config_path = request.query_params.get(
            "config_path", config_data.get("cloudflared_config_path", None)
        )
        if config_path:
            config_path = Path(config_path)
        app.state.TUNNEL.run_tunnel(
            port=config_data.get("port", 8000),
            config=Path(config_path) if config_path else None,
        )
    else:
        app.state.TUNNEL.run_tunnel(port=config_data.get("port", 8000))
    return JSONResponse({"service_started": bool(app.state.TUNNEL)})


async def kill_tunnel_service(_: Request = None) -> JSONResponse:
    if not app.state.TUNNEL:
        return JSONResponse(
            {"killed": False, "reason": "No Tunnel Service Running"},
            status_code=400,
        )
    app.state.TUNNEL.kill_service()
    app.state.TUNNEL = None
    return JSONResponse({"killed": True, "reason": "Tunnel Service Successfully Killed"})


async def check_path(request: Request) -> JSONResponse:
    body = await request.body()
    body = json.loads(body)
    file_path = Path(body["path"])
    valid = False
    if body["type"] == "folder" and file_path.is_dir():
        valid = True
    if body["type"] == "file" and file_path.is_file() and file_path.suffix in body["extensions"]:
        valid = True
    return JSONResponse({"valid": valid})


async def validate_inputs(request: Request) -> JSONResponse:
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse(
            {"detail": "Training Already Running"},
            status_code=status.HTTP_409_CONFLICT,
        )
    body = await request.body()
    body = json.loads(body)
    passed_validation, sdxl, errors, args, dataset_args, tags = validate(body)
    if passed_validation:
        output_args, _ = process_args(args)
        output_dataset_args, _ = process_dataset_args(dataset_args)
        final_args = {"args": output_args, "dataset": output_dataset_args, "tags": tags}
        return JSONResponse(final_args)
    return JSONResponse(
        errors,
        status_code=status.HTTP_400_BAD_REQUEST,
    )


async def is_training(_: Request) -> JSONResponse:
    exit_id = app.state.TRAINING_THREAD.poll() if app.state.TRAINING_THREAD else 0
    return JSONResponse(
        {
            "training": exit_id is None,
            "errored": exit_id is not None and exit_id != 0,
        }
    )


async def tokenize_text(request: Request) -> JSONResponse:
    if tokenizer is None:
        return JSONResponse(
            {"detail": "CLIP tokenizer is not available (offline and not cached)"},
            status_code=503,
        )
    text = request.query_params.get("text")
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print("Original string:", text)
    # print("Tokenized string:", tokens)
    # print("Token IDs:", token_ids)
    return JSONResponse({"tokens": tokens, "token_ids": token_ids, "length": len(tokens)})


async def start_training(request: Request) -> JSONResponse:
    global server
    temp = json.loads(app.state.CONFIG.read_text())
    if "colab" in temp and temp["colab"]:
        # Colab trains in the notebook once this server exits. Answer first (the GUI now treats a non-200 /train as
        # "did not start" and keeps the queue item), then close the tunnel and the server after the response is sent.
        async def _colab_shutdown() -> None:
            await kill_tunnel_service()
            await stop_server()

        return JSONResponse(
            {"detail": "Colab: the server stops and the notebook starts training", "training": True},
            background=BackgroundTask(_colab_shutdown),
        )
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse(
            {"detail": "Training Already Running"},
            status_code=status.HTTP_409_CONFLICT,
        )
    is_sdxl = request.query_params.get("sdxl", "False") == "True"
    train_type = request.query_params.get("train_mode", "lora")
    is_flux = request.query_params.get("flux", "False") == "True"
    is_anima = request.query_params.get("anima", "False") == "True"
    is_moss = request.query_params.get("moss", "False") == "True"

    # Parse accelerate (multi-GPU) settings
    accelerate_enabled = request.query_params.get("accelerate_enabled", "False") == "True"
    accelerate_gpu_ids = request.query_params.get("accelerate_gpu_ids")  # optional, e.g. "0,1"
    try:
        accelerate_num_processes = int(request.query_params.get("accelerate_num_processes", "2"))
        accelerate_main_process_port = int(request.query_params.get("accelerate_main_process_port", "29500"))
    except ValueError as e:
        return JSONResponse(
            {"detail": f"Invalid multi-GPU settings: {e}", "training": False},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    train_script = pick_train_script(train_type, is_sdxl, is_flux, is_anima, is_moss)
    if train_script is None:
        print(f"Unknown training request: {request.query_params}")
        return JSONResponse(
            {
                "detail": "Invalid Train Parameters",
                "sdxl": is_sdxl,
                "train_type": train_type,
                "flux": is_flux,
                "anima": is_anima,
                "moss": is_moss,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    app.state.TRAIN_SCRIPT = train_script

    server_config_dict = json.loads(app.state.CONFIG.read_text()) if app.state.CONFIG else {}
    python = sys.executable
    config = Path("runtime_store/config.toml")
    dataset = Path("runtime_store/dataset.toml")
    if not config.is_file() or not dataset.is_file():
        return JSONResponse(
            {"detail": "No Previously Validated Args"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    print(app.state.TRAIN_SCRIPT)

    # Build command based on accelerate settings (more than one process: --multi_gpu and the GPU ids)
    try:
        cmd = build_train_command(
            python,
            str(Path(f"sd_scripts/{app.state.TRAIN_SCRIPT}").resolve()),
            str(config.resolve()),
            str(dataset.resolve()),
            accelerate=accelerate_enabled,
            num_processes=accelerate_num_processes,
            main_process_port=accelerate_main_process_port,
            gpu_ids=accelerate_gpu_ids,
        )
    except ValueError as e:
        return JSONResponse(
            {"detail": f"Invalid multi-GPU settings: {e}", "training": False},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if accelerate_enabled:
        print(f"Launching with accelerate: {accelerate_num_processes} processes")

    app.state.TRAINING_THREAD = subprocess.Popen(cmd)
    if (
        "kill_tunnel_on_train_start" in server_config_dict
        and server_config_dict["kill_tunnel_on_train_start"]
        and app.state.TUNNEL
    ):
        app.state.TUNNEL.kill_service()
        app.state.TUNNEL = None
    if "kill_server_on_train_end" in server_config_dict and server_config_dict["kill_server_on_train_end"]:
        app.state.MONITOR_THREAD = Thread(target=monitor_training_thread, daemon=True)
        app.state.MONITOR_THREAD.start()
    return JSONResponse({"detail": "Training Started", "training": True})


async def stop_training(request: Request) -> JSONResponse:
    force = bool(request.query_params.get("force", False))
    if not app.state.TRAINING_THREAD or app.state.TRAINING_THREAD.poll() is not None:
        return JSONResponse(
            {"detail": "Not Currently Training"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if force:
        app.state.TRAINING_THREAD.stderr = None
        app.state.TRAINING_THREAD.kill()
        return JSONResponse({"detail": "Training Thread Killed"})
    else:
        app.state.TRAINING_THREAD.terminate()
    return JSONResponse({"detail": "Training Thread Requested to Die"})


async def start_resize(request: Request) -> JSONResponse:
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse({"detail": "Training Already Running"}, status_code=status.HTTP_409_CONFLICT)
    data = await request.body()
    data: list[str] = json.loads(data)
    python = sys.executable
    app.state.TRAINING_THREAD = subprocess.Popen([python, f"{Path('utils/resize_lora.py').resolve()}"] + data)
    return JSONResponse({"detail": "Resizing Started"})


def monitor_training_thread():
    if not app.state.TRAINING_THREAD:
        return
    global server
    app.state.TRAINING_THREAD.wait()
    server.should_exit = True
    server.force_exit = True


routes = [
    Route("/stop_server", stop_server, methods=["GET"]),
    Route("/start_tunnel_service", start_tunnel_service, methods=["GET"]),
    Route("/kill_tunnel_service", kill_tunnel_service, methods=["GET"]),
    Route("/check_path", check_path, methods=["POST"]),
    Route("/validate", validate_inputs, methods=["POST"]),
    Route("/is_training", is_training, methods=["GET"]),
    Route("/train", start_training, methods=["GET"]),
    Route("/tokenize", tokenize_text, methods=["GET"]),
    Route("/stop_training", stop_training, methods=["GET"]),
    Route("/resize", start_resize, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)
app.state.TRAIN_SCRIPT = None
app.state.TRAINING_THREAD = None
app.state.CONFIG = Path("config.json")
app.state.MONITOR_THREAD = None
app.state.TUNNEL = None
app.state.BACKEND_TOKEN = None

if not app.state.CONFIG.exists():
    with app.state.CONFIG.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"remote": False, "port": 8000}, indent=2))

config_data = json.loads(app.state.CONFIG.read_text())

# Bind 127.0.0.1 unless config.json "host" or --host says otherwise; reachable from other machines (another host or
# a tunnel) only with LETS_BACKEND_TOKEN, which every request must then send (utils/security.py)
BIND_HOST = bind_host(config_data, CLI_HOST)
SECURITY = startup_security(BIND_HOST, config_data, os.environ)
if SECURITY.refusal:
    print(SECURITY.refusal, file=sys.stderr)
    sys.exit(2)
app.state.BACKEND_TOKEN = SECURITY.token
app.add_middleware(BackendGuard, token=SECURITY.token)
if SECURITY.exposed:
    app.debug = False  # no tracebacks (paths, values) in error responses to other machines

if config_data.get("remote", False):
    app.state.TUNNEL = create_tunnel(config_data)
    if isinstance(app.state.TUNNEL, CloudflaredTunnel):
        config_path = config_data.get("cloudflared_config_path", None)
        app.state.TUNNEL.run_tunnel(
            port=config_data.get("port", 8000), config=Path(config_path) if config_path else None
        )
    else:
        app.state.TUNNEL.run_tunnel(port=config_data.get("port", 8000))
print(f"Backend API on http://{BIND_HOST}:{config_data.get('port', 8000)}"
      + (f" (requests need the {TOKEN_ENV} bearer token)" if SECURITY.token else ""))
uvi_config = uvicorn.Config(
    app,
    host=BIND_HOST,
    loop="asyncio",
    log_level="critical",
    port=config_data.get("port", 8000),
)
server = uvicorn.Server(config=uvi_config)

if __name__ == "__main__":
    server.run()
