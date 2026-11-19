import json
from pathlib import Path
import sys
import subprocess
import os

from install_uv import ensure_uv, uv_pip_install, get_venv_python_version, get_venv_python

PLATFORM = "windows" if sys.platform == "win32" else "linux" if sys.platform == "linux" else ""

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def check_git_install() -> bool:
    try:
        subprocess.check_call(
            "git --version",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=PLATFORM == "linux",
        )
    except FileNotFoundError:
        logger.error("ERROR: git is not installed, please install git")
        return False
    return True

# The default Accelerate config for a new install: single process, bf16. Only keys Accelerate's config loader knows
# (it stops on any other key with "unknown keys", and 'accelerate launch' then never starts); the unused TPU-pod
# entries are left out (their defaults are the same). Multi-GPU runs pass --multi_gpu / --gpu_ids at launch time.
ACCELERATE_CONFIG = """compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
dynamo_backend: 'NO'
fsdp_config: {}
gpu_ids: '0'
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
"""

# Keys an older version of this installer wrote by mistake (a typo of downcast_bf16) and Accelerate rejects: removed
# from an existing default config, which is otherwise never touched.
INVALID_ACCELERATE_KEYS = ("downcase_fp16",)


def accelerate_config_path(platform: str) -> Path:
    home = Path(os.environ["USERPROFILE"]) if platform == "windows" else Path.home()
    return home.joinpath(".cache/huggingface/accelerate/default_config.yaml")


def repair_accelerate_config(path: Path) -> list:
    """Remove INVALID_ACCELERATE_KEYS lines from an existing config; returns the keys removed."""
    text = path.read_bytes().decode("utf-8")
    kept, removed = [], []
    for line in text.splitlines(keepends=True):
        key = line.split(":", 1)[0].strip()
        if ":" in line and not line[:1].isspace() and key in INVALID_ACCELERATE_KEYS:
            removed.append(key)
            continue
        kept.append(line)
    if removed:
        path.write_bytes("".join(kept).encode("utf-8"))
    return removed


def setup_accelerate(platform: str, path: Path | None = None) -> None:
    path = path or accelerate_config_path(platform)
    if path.exists():
        removed = repair_accelerate_config(path)
        if removed:
            logger.warning(f"Removed {', '.join(removed)} from {path}: Accelerate rejects it")
        else:
            logger.info("Default accelerate config already exists, skipping.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ACCELERATE_CONFIG, encoding="utf-8")


# The training venv: sd_scripts/venv_t214 with torch 2.14.0 + torchvision 0.29.0 for CUDA 13.0 (needs an NVIDIA driver
# with CUDA 13). triton 3.8 is pinned in sd_scripts/requirements.txt; flash-attn 2.8.4 has a wheel for Windows Python
# 3.11 only (no Linux wheel for torch 2.14 yet: use --attn_mode torch there); no xformers. torch 2.14's Windows build
# ships expandable_segments: the CUDA caching allocator stops fragmenting into exact-size segments that rebuilt up to
# the full 32 GB after every epoch-end empty_cache (WDDM then paged over PCIe: ~130 W, 20-500 s/step). The network
# trainers turn them on by themselves (library/cuda_memory.py).
TORCH_VENV = "venv_t214"
TORCH_VERSION = "2.14"
TORCH_PACKAGES = ("torch==2.14.0", "torchvision==0.29.0")
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"

# flash-attention wheel URLs for each platform/python combination
FLASH_ATTN_WHEELS = {
    ("win32", 11): "https://github.com/sdbds/flash-attention-for-windows/releases/download/2.8.4/flash_attn-2.8.4+cu130torch2.14.0cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl",
}


def create_torch_venv(uv: str) -> str:
    """sd_scripts/venv_t214, created when missing. Only ever this directory: another venv in sd_scripts (venv, .venv)
    is never reused."""
    if get_venv_python(TORCH_VENV).exists():
        logger.info(f"Reusing existing venv at '{TORCH_VENV}'")
        return TORCH_VENV
    logger.info(f"Creating venv at '{TORCH_VENV}' with Python 3.11...")
    subprocess.check_call([uv, "venv", "--python", "3.11", "--seed", TORCH_VENV])
    return TORCH_VENV


def installed_torch_version(venv_path: str) -> str | None:
    """torch's version from its dist-info in the venv (no interpreter started), or None."""
    for site in list(Path(venv_path).glob("Lib/site-packages")) + list(Path(venv_path).glob("lib/python3*/site-packages")):
        for info in site.glob("torch-*.dist-info"):
            return info.name[len("torch-"):-len(".dist-info")]
    return None


def check_venv_torch(venv_path: str = TORCH_VENV) -> None:
    """Warns when the venv holds another torch (for instance one upgraded by hand): torch 2.14 is installed over it."""
    version = installed_torch_version(venv_path)
    if version and ".".join(version.split("+")[0].split(".")[:2]) != TORCH_VERSION:
        logger.warning(f"'{venv_path}' holds torch {version}; installing torch {TORCH_VERSION} over it")


def _install_flash_attn(uv: str, venv_path: str = TORCH_VENV) -> None:
    """Install flash-attention wheel for the current platform and Python version.

    Detects the Python version from the venv executable. This is done
    explicitly before requirements.txt to avoid uv reinstalling flash-attn
    due to version normalization differences (FALSE vs false in local
    version identifiers).
    """
    python_version = get_venv_python_version(venv_path)
    python_minor = int(python_version.split(".")[1])
    platform_key = "win32" if sys.platform == "win32" else "linux"
    wheel_url = FLASH_ATTN_WHEELS.get((platform_key, python_minor))

    if wheel_url:
        logger.info(f"Installing flash-attn for {platform_key} Python 3.{python_minor}...")
        uv_pip_install(uv, "--no-deps", wheel_url, venv_path=venv_path)
    else:
        logger.info(f"No flash-attn wheel for {platform_key} Python 3.{python_minor}, skipping")


def setup_venv(uv: str, venv_path: str = TORCH_VENV):
    """Install all backend + sd_scripts packages into the venv using uv."""
    uv_pip_install(uv, "-U", "typing-extensions~=4.16.0", venv_path=venv_path)

    uv_pip_install(
        uv,
        "-U", *TORCH_PACKAGES, "numpy~=2.2.6",
        "--index-url", TORCH_INDEX_URL,
        venv_path=venv_path,
    )

    # git packages are pinned to the same commits as sd_scripts/requirements.txt (bump both together)
    uv_pip_install(
        uv,
        "-U", "--force-reinstall", "--no-deps",
        "git+https://github.com/67372a/RamTorch@feae2437e2f4ecabefdb410e28342433be8c05cd",
        venv_path=venv_path,
    )

    uv_pip_install(
        uv,
        "-U", "--force-reinstall", "--no-deps",
        "git+https://github.com/67372a/customized-optimizers@1d0e9426c862fabd14460905ff57e289572b6a34",
        venv_path=venv_path,
    )

    uv_pip_install(
        uv,
        "-U", "--no-deps", "torchao~=0.13.0",
        "--index-strategy", "unsafe-best-match",
        "--extra-index-url", TORCH_INDEX_URL,
        venv_path=venv_path,
    )

    uv_pip_install(
        uv,
        "-U", "--force-reinstall", "--no-deps",
        "git+https://github.com/67372a/LyCORIS@c2cfd8ae5c5418ea7c77923eda4afa099bc69ba5",
        venv_path=venv_path,
    )

    # Pre-install flash-attn explicitly to avoid uv version normalization issues
    # (uv normalizes FALSE -> false in local version identifiers, causing reinstall loops)
    _install_flash_attn(uv, venv_path)

    uv_pip_install(uv, "-r", "requirements.txt", venv_path=venv_path)
    uv_pip_install(uv, "-e", "../custom_scheduler/.", venv_path=venv_path)
    uv_pip_install(uv, "-r", "../requirements.txt", venv_path=venv_path)


# colab only
def setup_colab(uv: str, venv_path: str = TORCH_VENV):
    setup_venv(uv, venv_path)
    setup_accelerate("linux")


def ask_yes_no(question: str) -> bool:
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"{question} (y/n): ")
    return reply == "y"


def setup_config(colab: bool = False, local: bool = False) -> None:
    if colab:
        config = {
            "remote": True,
            "remote_mode": "cloudflared",
            "kill_tunnel_on_train_start": True,
            "kill_server_on_train_end": True,
            "colab": True,
            "port": 8000,
        }
        with open("config.json", "w") as f:
            f.write(json.dumps(config, indent=2))
        return
    is_remote = False if local else ask_yes_no("are you using this remotely?")
    remote_mode = "none"
    if is_remote:
        remote_mode = "ngrok" if ask_yes_no("do you want to use ngrok?") else "cloudflared"
    ngrok_token = ""
    if remote_mode == "ngrok":
        ngrok_token = input(
            "copy paste your token from your ngrok dashboard (https://dashboard.ngrok.com/get-started/your-authtoken) (requires account): "
        )

    with open("config.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "remote": is_remote,
                    "remote_mode": remote_mode,
                    "ngrok_token": ngrok_token,
                    "port": 8000,
                },
                indent=2,
            )
        )


def main():
    if not check_git_install():
        quit()

    subprocess.check_call("git submodule update --init --recursive", shell=PLATFORM == "linux")

    colab = len(sys.argv) > 1 and sys.argv[1] == "colab"
    setup_config(
        colab,
        len(sys.argv) > 1 and sys.argv[1] == "local",
    )

    os.chdir("sd_scripts")

    logger.info("creating venv and installing requirements")
    uv = ensure_uv()
    # any later argument (an old --torch choice too) is ignored: the venv is always venv_t214 with torch 2.14
    venv_path = create_torch_venv(uv)
    check_venv_torch(venv_path)

    if colab:
        setup_colab(uv, venv_path)
        logger.info("completed installing")
        quit()

    setup_venv(uv, venv_path)
    setup_accelerate(PLATFORM)

    logger.info("Completed installing, you can run the server via the run.bat or run.sh files")


if __name__ == "__main__":
    main()
