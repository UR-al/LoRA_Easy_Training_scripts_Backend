"""The command a /train request starts (pure apart from reading Accelerate's default config; nothing is launched here).

Without Accelerate: python <script>. With Accelerate: python -m accelerate.commands.launch <script>, whose default
config (the one 'accelerate launch' itself reads) decides how much the backend adds to the command line:
- no default config, or a single-process one (distributed_type NO, what the installer writes): --num_processes alone
  still starts one process on the config's GPU, so more than one process adds --multi_gpu and the GPU ids;
- the user's own multi-process config ('accelerate config': MULTI_GPU, DEEPSPEED, FSDP, ...): only --num_processes,
  as before, so that config keeps choosing the engine and the GPUs (an explicit --multi_gpu would replace DeepSpeed /
  FSDP with plain DDP, and --gpu_ids would replace its GPU list).
GPU ids the request names (accelerate_gpu_ids) are passed for any number of processes; without them a multi-GPU
launch on a single-process config uses the first N of the backend's CUDA_VISIBLE_DEVICES, else 0..N-1.
"""

import os
from pathlib import Path
from typing import Mapping, Optional

# distributed types for which 'accelerate launch' turns multi_gpu on by itself (accelerate/commands/launch.py)
MULTI_DEVICE_TYPES = frozenset(
    {"MULTI_GPU", "MULTI_NPU", "MULTI_MLU", "MULTI_SDAA", "MULTI_MUSA", "MULTI_XPU", "MULTI_HPU", "MULTI_NEURON"}
)
_READ_DEFAULT = object()


def default_accelerate_config_path(environ: Optional[Mapping[str, str]] = None) -> Path:
    """Where 'accelerate launch' looks for its default config (accelerate.commands.config.config_args)."""
    environ = os.environ if environ is None else environ
    hf_home = environ.get("HF_HOME", os.path.join(environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
    return Path(os.path.expanduser(hf_home), "accelerate", "default_config.yaml")


def read_accelerate_config(path: Path) -> Optional[dict]:
    """The config as a dict; None when there is no such file, {} when it cannot be read (Accelerate reports that)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            import json

            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _distributed_type(config: Mapping) -> str:
    value = config.get("distributed_type")
    if value is False:  # an unquoted NO in YAML
        return "NO"
    return str(value or "").strip().upper()


def _split_ids(value) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _requested_ids(requested, num_processes: int) -> Optional[list[str]]:
    if requested is None or not str(requested).strip():
        return None
    ids = _split_ids(requested)
    if not all(item.isdigit() for item in ids):
        raise ValueError(f"GPU ids must be numbers separated by commas, got {requested!r}")
    if len(ids) != num_processes:
        raise ValueError(f"{num_processes} processes need {num_processes} GPU ids, got {requested!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"GPU ids must be different, got {requested!r}")
    return ids


def gpu_ids_for(num_processes: int, requested: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    """Comma-separated GPU ids for num_processes processes: the requested ones, else the first N of
    CUDA_VISIBLE_DEVICES, else 0..N-1; raises ValueError."""
    ids = _requested_ids(requested, num_processes)
    if ids is None:
        environ = os.environ if environ is None else environ
        visible = _split_ids(environ.get("CUDA_VISIBLE_DEVICES", ""))
        if visible:
            if len(visible) < num_processes:
                raise ValueError(
                    f"{num_processes} processes need {num_processes} GPUs, "
                    f"but CUDA_VISIBLE_DEVICES lists {len(visible)}: {environ['CUDA_VISIBLE_DEVICES']!r}"
                )
            ids = visible[:num_processes]
        else:
            ids = [str(i) for i in range(num_processes)]
        if len(set(ids)) != len(ids):
            raise ValueError(f"GPU ids must be different, got {','.join(ids)!r}")
    return ",".join(ids)


def accelerate_launch_options(
    num_processes: int,
    gpu_ids: str | None = None,
    config: Optional[Mapping] = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """The accelerate launch options before --main_process_port; config is the default config (None: no file)."""
    num_processes = int(num_processes)
    if num_processes < 1:
        raise ValueError(f"the number of processes must be at least 1, got {num_processes}")
    requested = _requested_ids(gpu_ids, num_processes)

    if config is None or _distributed_type(config) == "NO":
        # no config or the installer's single-process one: --num_processes alone would run one process on its GPU
        if num_processes > 1:
            return ["--multi_gpu", f"--num_processes={num_processes}",
                    f"--gpu_ids={gpu_ids_for(num_processes, gpu_ids, environ)}"]
        return ["--num_processes=1"] + ([f"--gpu_ids={requested[0]}"] if requested else [])

    # the user's own multi-process config decides the engine and, unless the request names them, the GPUs
    distributed_type = _distributed_type(config)
    options = [f"--num_processes={num_processes}"]
    if requested:
        if num_processes == 1 and distributed_type in MULTI_DEVICE_TYPES:
            raise ValueError(
                f"the default Accelerate config is multi-GPU ({distributed_type}), which accelerate launch refuses "
                f"to run on the single GPU id {requested[0]}: choose at least 2 GPUs, or leave the GPU ids empty"
            )
        options.append(f"--gpu_ids={','.join(requested)}")
    elif distributed_type in MULTI_DEVICE_TYPES:
        configured = str(config.get("gpu_ids") if config.get("gpu_ids") is not None else "all").strip()
        if configured.lower() != "all" and len(_split_ids(configured)) < num_processes:
            raise ValueError(
                f"the default Accelerate config runs on GPU ids {configured!r}, fewer than the {num_processes} "
                f"processes requested: name {num_processes} GPU ids, or change the config with 'accelerate config'"
            )
    return options


def build_train_command(
    python: str,
    script: str,
    config: str,
    dataset: str,
    accelerate: bool = False,
    num_processes: int = 1,
    main_process_port: int = 29500,
    gpu_ids: str | None = None,
    environ: Mapping[str, str] | None = None,
    accelerate_config=_READ_DEFAULT,
) -> list[str]:
    """argv for the training subprocess; raises ValueError for an impossible multi-GPU request.

    accelerate_config: the parsed default Accelerate config (None: there is none); read from the default location
    for environ (the training process inherits the backend's environment) when not given.
    """
    tail = [str(script), f"--config_file={config}", f"--dataset_config={dataset}"]
    if not accelerate:
        return [python, *tail]
    if accelerate_config is _READ_DEFAULT:
        accelerate_config = read_accelerate_config(default_accelerate_config_path(environ))
    cmd = [python, "-m", "accelerate.commands.launch"]
    cmd += accelerate_launch_options(num_processes, gpu_ids, accelerate_config, environ)
    cmd += [f"--main_process_port={int(main_process_port)}"]
    return cmd + tail
