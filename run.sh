#!/bin/bash

# PyTorch profile: LETS_TORCH=2.14 runs the backend in sd_scripts/venv_t214 (installer.py --torch 2.14);
# with only venv_t214 installed it is used anyway
case "${LETS_TORCH//[[:space:]]/}" in
    2.14|2.14.0|214) venv=venv_t214 ;;
    *) venv=venv ;;
esac
if [ ! -f "sd_scripts/$venv/bin/activate" ] && [ -f sd_scripts/venv_t214/bin/activate ]; then
    venv=venv_t214
fi
source "sd_scripts/$venv/bin/activate"
python3 main.py
