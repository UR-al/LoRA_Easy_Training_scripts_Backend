import json
from pathlib import Path
import sys
import subprocess
import os

from install_uv import (ensure_uv, create_venv, uv_pip_install, get_venv_python_version, get_venv_python,
                        find_existing_venv)

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


# flash-attention wheel URLs for each platform/python combination (torch 2.7 profile)
FLASH_ATTN_WHEELS = {
    ("win32", 11): "https://github.com/sdbds/flash-attention-for-windows/releases/download/2.8.0.post2/flash_attn-2.8.0.post2+cu128torch2.7.1cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl",
    ("linux", 10): "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    ("linux", 11): "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    ("linux", 12): "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    ("linux", 13): "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp313-cp313-linux_x86_64.whl",
}

# PyTorch profiles (2026-11-04). Each is a set of packages that ran together (torch, torchvision, triton, flash-attn,
# xformers) and lives in a venv of its own, so both can stay installed side by side:
#   "2.7"  -> sd_scripts/venv       torch 2.7.1 + CUDA 12.8 (the original set; the default)
#   "2.14" -> sd_scripts/venv_t214  torch 2.14.0 + CUDA 13.0 (needs an NVIDIA driver with CUDA 13)
# torch 2.14's Windows build ships expandable_segments: the CUDA caching allocator stops fragmenting into exact-size
# segments that rebuilt up to the full 32 GB after every epoch-end empty_cache (WDDM then paged over PCIe: ~130 W,
# 20-500 s/step). The network trainers turn them on by themselves under torch >= 2.14 (library/cuda_memory.py). No
# Linux flash-attn wheel exists for torch 2.14 yet (use --attn_mode torch there) and no xformers build.
# Install: install.bat [--torch 2.7|2.14|both] (asks when not given). Update: update.bat updates every profile venv
# that exists; update.bat --torch 2.14 updates (or adds) that one. Train with 2.14: LETS_TORCH=2.14 (run.bat, the
# examples' launchers) or "torch_profile": "2.14" in the UI's config.json.
DEFAULT_TORCH_PROFILE = "2.7"
TORCH_PROFILE_MARKER = "lets_torch_profile.txt"  # written into each venv: which profile it holds
TORCH_PROFILE_ALIASES = {"2.7": "2.7", "2.7.1": "2.7", "27": "2.7", "2.14": "2.14", "2.14.0": "2.14", "214": "2.14"}
BOTH_PROFILES = ("both", "all")
TORCH_PROFILES = {
    "2.7": {
        "venv": "venv",
        "packages": ("torch~=2.7.1", "torchvision~=0.22.1"),
        "index_url": "https://download.pytorch.org/whl/cu128",
        "xformers": "xformers==0.0.31.post1",
        "flash_attn": FLASH_ATTN_WHEELS,
        "overrides": (),  # sd_scripts/requirements.txt pins triton for this profile
    },
    "2.14": {
        "venv": "venv_t214",
        "packages": ("torch==2.14.0", "torchvision==0.29.0"),
        "index_url": "https://download.pytorch.org/whl/cu130",
        "xformers": None,
        "flash_attn": {
            ("win32", 11): "https://github.com/sdbds/flash-attention-for-windows/releases/download/2.8.4/flash_attn-2.8.4+cu130torch2.14.0cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl",
        },
        # requirements.txt pins triton 3.3 (torch 2.7); torch 2.14 needs triton 3.8 (Linux: torch's own triton~=3.8.0)
        "overrides": (
            'triton-windows==3.8.0.post29; platform_system == "Windows"',
            'triton~=3.8.0; platform_system == "Linux"',
        ),
    },
}


def normalize_torch_profile(value: str) -> str:
    """'2.14' / '2.14.0' / '214' -> '2.14'; raises SystemExit for an unknown profile."""
    profile = TORCH_PROFILE_ALIASES.get(str(value).strip().lower())
    if profile is None:
        raise SystemExit(f"Unknown PyTorch profile {value!r}: use one of {', '.join(TORCH_PROFILES)} or both")
    return profile


def _profiles_of(value: str) -> list:
    """'2.14' -> ['2.14']; 'both' / 'all' -> every profile, the default first."""
    if str(value).strip().lower() in BOTH_PROFILES:
        return list(TORCH_PROFILES)
    return [normalize_torch_profile(value)]


def parse_torch_profiles(argv, use_env: bool = True) -> list | None:
    """The profiles chosen on the command line (--torch 2.14 / --torch=both), else LETS_TORCH (use_env); None = not
    chosen. updater.py passes use_env=False: LETS_TORCH picks the venv to train in and must not narrow an update."""
    argv = list(argv)
    for i, arg in enumerate(argv):
        if arg == "--torch":
            if i + 1 >= len(argv):
                raise SystemExit("--torch needs a value: " + ", ".join(TORCH_PROFILES) + " or both")
            return _profiles_of(argv[i + 1])
        if arg.startswith("--torch="):
            return _profiles_of(arg.split("=", 1)[1])
    env = os.environ.get("LETS_TORCH", "").strip() if use_env else ""
    return _profiles_of(env) if env else None


def ask_torch_profiles() -> list:
    """Asks which profile(s) to install; without an interactive console (pipes, CI, scheduled jobs) the default."""
    if sys.stdin is None or not sys.stdin.isatty():
        logger.info(f"no interactive console: installing PyTorch {DEFAULT_TORCH_PROFILE} (--torch picks another)")
        return [DEFAULT_TORCH_PROFILE]
    print("Which PyTorch should the training venv use? (Enter = 2.7)")
    print("  2.7  : torch 2.7.1 + CUDA 12.8 - the long-tested default (sd_scripts/venv)")
    print("  2.14 : torch 2.14.0 + CUDA 13.0 - Windows: expandable CUDA cache, no slowdowns when VRAM fills;"
          " needs an NVIDIA driver with CUDA 13 (sd_scripts/venv_t214)")
    print("  both : install both side by side")
    while True:
        try:
            reply = input("PyTorch (2.7/2.14/both): ").strip() or DEFAULT_TORCH_PROFILE
        except EOFError:
            return [DEFAULT_TORCH_PROFILE]
        if reply.lower() in TORCH_PROFILE_ALIASES or reply.lower() in BOTH_PROFILES:
            return _profiles_of(reply)


def profile_venv_dir(profile: str) -> str:
    return TORCH_PROFILES[normalize_torch_profile(profile)]["venv"]


def create_profile_venv(uv: str, profile: str) -> str:
    """The venv of a profile, created when missing. 2.7 keeps create_venv's behaviour (reuses venv or .venv); other
    profiles only ever use their own directory, never another profile's .venv."""
    profile = normalize_torch_profile(profile)
    path = profile_venv_dir(profile)
    if profile == DEFAULT_TORCH_PROFILE:
        return create_venv(uv, path, "3.11")
    if get_venv_python(path).exists():
        logger.info(f"Reusing existing venv at '{path}'")
        return path
    logger.info(f"Creating venv at '{path}' with Python 3.11...")
    subprocess.check_call([uv, "venv", "--python", "3.11", "--seed", path])
    return path


def existing_profile_venvs() -> list:
    """[(profile, venv path)] for every profile whose venv exists in the current directory (sd_scripts)."""
    out = []
    for profile in TORCH_PROFILES:
        if profile == DEFAULT_TORCH_PROFILE:
            path = find_existing_venv(profile_venv_dir(profile))
        else:
            path = profile_venv_dir(profile) if get_venv_python(profile_venv_dir(profile)).exists() else None
        if path:
            out.append((profile, path))
    return out


def installed_torch_version(venv_path: str) -> str | None:
    """torch's version from its dist-info in the venv (no interpreter started), or None."""
    for site in list(Path(venv_path).glob("Lib/site-packages")) + list(Path(venv_path).glob("lib/python3*/site-packages")):
        for info in site.glob("torch-*.dist-info"):
            return info.name[len("torch-"):-len(".dist-info")]
    return None


def check_venv_torch(venv_path: str, profile: str) -> None:
    """Warns when the venv holds a torch of another profile (for instance a venv upgraded by hand): the profile's
    torch is installed over it."""
    version = installed_torch_version(venv_path)
    want = normalize_torch_profile(profile)
    if version and TORCH_PROFILE_ALIASES.get(".".join(version.split("+")[0].split(".")[:2])) != want:
        logger.warning(f"'{venv_path}' holds torch {version}; installing the PyTorch {want} profile over it")


def read_torch_profile(venv_path: str) -> str | None:
    """The profile recorded in the venv by the last install / update, or None."""
    try:
        text = Path(venv_path, TORCH_PROFILE_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return TORCH_PROFILE_ALIASES.get(text)


def write_torch_profile(venv_path: str, profile: str) -> None:
    Path(venv_path, TORCH_PROFILE_MARKER).write_text(profile + "\n", encoding="utf-8")


def _install_flash_attn(uv: str, venv_path: str = "venv", wheels: dict | None = None) -> None:
    """Install flash-attention wheel for the current platform and Python version.

    Detects the Python version from the venv executable. This is done
    explicitly before requirements.txt to avoid uv reinstalling flash-attn
    due to version normalization differences (FALSE vs false in local
    version identifiers).
    """
    wheels = FLASH_ATTN_WHEELS if wheels is None else wheels
    python_version = get_venv_python_version(venv_path)
    python_minor = int(python_version.split(".")[1])
    platform_key = "win32" if sys.platform == "win32" else "linux"
    wheel_url = wheels.get((platform_key, python_minor))

    if wheel_url:
        logger.info(f"Installing flash-attn for {platform_key} Python 3.{python_minor}...")
        uv_pip_install(uv, "--no-deps", wheel_url, venv_path=venv_path)
    else:
        logger.info(f"No flash-attn wheel for {platform_key} Python 3.{python_minor}, skipping")


def setup_venv(uv: str, venv_path: str = "venv", torch_profile: str = DEFAULT_TORCH_PROFILE):
    """Install all backend + sd_scripts packages into the venv using uv, with the given PyTorch profile."""
    torch_profile = normalize_torch_profile(torch_profile)
    profile = TORCH_PROFILES[torch_profile]
    logger.info(f"PyTorch profile {torch_profile}: {' '.join(profile['packages'])} from {profile['index_url']}")
    uv_pip_install(uv, "-U", "typing-extensions~=4.16.0", venv_path=venv_path)

    uv_pip_install(
        uv,
        "-U", *profile["packages"], "numpy~=2.2.6",
        "--index-url", profile["index_url"],
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

    if profile["xformers"]:  # none for torch 2.14 (the trainers import it only optionally, for attn_mode xformers)
        uv_pip_install(
            uv,
            "-U", "--no-deps", profile["xformers"],
            "--index-url", profile["index_url"],
            venv_path=venv_path,
        )

    uv_pip_install(
        uv,
        "-U", "--no-deps", "torchao~=0.13.0",
        "--index-strategy", "unsafe-best-match",
        "--extra-index-url", profile["index_url"],
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
    _install_flash_attn(uv, venv_path, profile["flash_attn"])

    requirements = ["-r", "requirements.txt"]
    if profile["overrides"]:
        # uv --override replaces requirements.txt's own spec for these packages (its triton pins belong to 2.7)
        overrides = Path(venv_path, "lets_torch_overrides.txt")
        overrides.write_text("\n".join(profile["overrides"]) + "\n", encoding="utf-8")
        requirements += ["--override", str(overrides)]
    uv_pip_install(uv, *requirements, venv_path=venv_path)
    uv_pip_install(uv, "-e", "../custom_scheduler/.", venv_path=venv_path)
    uv_pip_install(uv, "-r", "../requirements.txt", venv_path=venv_path)
    write_torch_profile(venv_path, torch_profile)
    logger.info(f"PyTorch profile {torch_profile} recorded in {Path(venv_path, TORCH_PROFILE_MARKER)}")


# colab only
def setup_colab(uv: str, venv_path: str = "venv", torch_profile: str = DEFAULT_TORCH_PROFILE):
    setup_venv(uv, venv_path, torch_profile)
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
    # --torch 2.7 / 2.14 / both (or LETS_TORCH), else ask (colab and non-interactive consoles: the default)
    profiles = parse_torch_profiles(sys.argv[1:]) or ([DEFAULT_TORCH_PROFILE] if colab else ask_torch_profiles())

    for torch_profile in profiles:
        venv_path = create_profile_venv(uv, torch_profile)
        check_venv_torch(venv_path, torch_profile)
        if colab:
            setup_colab(uv, venv_path, torch_profile)
        else:
            setup_venv(uv, venv_path, torch_profile)
    if colab:
        logger.info("completed installing")
        quit()

    setup_accelerate(PLATFORM)

    logger.info("Completed installing, you can run the server via the run.bat or run.sh files")


if __name__ == "__main__":
    main()
