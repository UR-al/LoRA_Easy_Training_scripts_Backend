from subprocess import check_call
import os
import sys

from install_uv import ensure_uv, find_existing_venv
from installer import (PLATFORM, DEFAULT_TORCH_PROFILE, setup_venv, parse_torch_profiles, create_profile_venv,
                       existing_profile_venvs, check_venv_torch)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_targets(uv: str, argv) -> list:
    """[(profile, venv path)] to update: the profiles named by --torch (their venvs are created when missing, which
    adds a profile), else every profile venv that exists, else the default venv. LETS_TORCH is ignored here: it picks
    the venv to train in, and a plain update.bat must keep every installed venv current."""
    chosen = parse_torch_profiles(argv, use_env=False)
    if chosen:
        return [(profile, create_profile_venv(uv, profile)) for profile in chosen]
    return existing_profile_venvs() or [(DEFAULT_TORCH_PROFILE, find_existing_venv("venv") or "venv")]


def main():
    check_call("git submodule update --init --recursive", shell=PLATFORM == "linux")
    os.chdir("sd_scripts")

    uv = ensure_uv()
    for torch_profile, venv_path in update_targets(uv, sys.argv[1:]):
        logger.info(f"updating '{venv_path}' (PyTorch {torch_profile})")
        check_venv_torch(venv_path, torch_profile)
        setup_venv(uv, venv_path, torch_profile)


if __name__ == "__main__":
    main()
