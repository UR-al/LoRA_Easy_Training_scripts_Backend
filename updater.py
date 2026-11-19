from subprocess import check_call
import os

from install_uv import ensure_uv
from installer import (PLATFORM, TORCH_VERSION, setup_venv, create_torch_venv, check_venv_torch,
                       accelerate_config_path, setup_accelerate)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    check_call("git submodule update --init --recursive", shell=PLATFORM == "linux")
    os.chdir("sd_scripts")

    uv = ensure_uv()
    # the training venv sd_scripts/venv_t214, created when missing; arguments (an old --torch choice too) are ignored
    venv_path = create_torch_venv(uv)
    logger.info(f"updating '{venv_path}' (PyTorch {TORCH_VERSION})")
    check_venv_torch(venv_path)
    setup_venv(uv, venv_path)
    # an existing default Accelerate config loses the keys Accelerate rejects (an older installer wrote one)
    if accelerate_config_path(PLATFORM).exists():
        setup_accelerate(PLATFORM)


if __name__ == "__main__":
    main()
