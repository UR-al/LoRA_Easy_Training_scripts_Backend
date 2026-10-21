"""Model-type rules of /validate for the fork's trainers (pure; validation.py imports torch, this does not).

- Anima: the anima_args section, or the Anima-only 'qwen3' key.
- MOSS-Image (moss_train_network.py): the moss_args section, or the MOSS-only 'moss_text_encoder' key.
- Anima 3.8B v1.1 connector: anima38_connector = true (inside the Anima section).
MOSS-Image and the Anima 3.8B connector load their text encoder only while caching its outputs, so both need
cache_text_encoder_outputs (MOSS also takes cache_text_encoder_outputs_to_disk alone: its trainer turns caching on).
"""

import os

# Keys whose empty string is a value, not "not filled in": the flatten loop keeps "" for them.
# moss_sample_uncond_skip_layers = "" switches the skipped negative-branch block off (the default is "22").
KEEP_EMPTY = {"moss_sample_uncond_skip_layers"}

# Keys the GUI reads from its own TOML (saving_args) and the trainer does not know: never written to the trainer
# config, so strict_config_keys = true does not stop on them.
GUI_ONLY_KEYS = ("save_toml", "save_toml_location", "tag_occurrence", "tag_file_location")

TE_CACHE_ERROR = "MOSS-Image / Anima 3.8B need Cache Text Encoder Outputs (Network section)"

# Paths a model section must hold itself: an empty one would otherwise be filled from General Args by the flatten
# loop (an SD / SDXL checkpoint and VAE handed to the DiT trainer).
SECTION_PATHS = {
    "moss_args": (("pretrained_model_name_or_path", "DiT Model"), ("vae", "VAE Model"),
                  ("moss_text_encoder", "Text Encoder")),
    "anima_args": (("pretrained_model_name_or_path", "DiT Model"), ("qwen3", "Qwen3 Model"), ("vae", "VAE Model")),
}
SECTION_TITLES = {"moss_args": "MOSS-Image Args", "anima_args": "Anima Args"}

# Tokenizer folders: the trainers fall back to these environment variables (of the backend process, which the
# training subprocess inherits). The Anima 3.8B Qwen3.5 tokenizer (qwen35_tokenizer_dir) has no rule here: an empty
# one is allowed, since the trainer then uses $ANIMA38_QWEN35_TOKENIZER_DIR or its own default folder, and stops with
# a clear error if neither exists.
TOKENIZER_RULES = (
    # (model flag, arg, environment variable, label)
    ("moss", "moss_tokenizer_dir", "MOSS_TOKENIZER_DIR", "MOSS-Image Tokenizer Folder (moss_tokenizer_dir)"),
)


def detect_models(sections: dict, output_args: dict) -> dict:
    """sections: the GUI args by section ({"anima_args": {...}, ...}); output_args: the flattened args."""
    sections = sections if isinstance(sections, dict) else {}
    is_anima = "qwen3" in output_args or bool(sections.get("anima_args"))
    is_moss = "moss_text_encoder" in output_args or bool(sections.get("moss_args"))
    is_anima38 = bool(output_args.get("anima38_connector"))
    return {"anima": is_anima, "moss": is_moss, "anima38": is_anima38}


def file_inputs(models: dict) -> list[dict]:
    """Path args checked for existence (and converted to posix) by validate_args, with whether each is required."""
    return [
        {"name": "pretrained_model_name_or_path", "required": True},
        {"name": "qwen3", "required": models["anima"]},
        {"name": "vae", "required": models["anima"] or models["moss"]},
        {"name": "moss_text_encoder", "required": models["moss"]},
        {"name": "moss_tokenizer_dir", "required": False},
        {"name": "qwen35_path", "required": models["anima38"]},
        {"name": "qwen35_tokenizer_dir", "required": False},
        {"name": "sample_prompts", "required": False},
        {"name": "output_dir", "required": True},
        {"name": "logging_dir", "required": False},
        {"name": "t5_tokenizer_path", "required": False},
    ]


def _filled(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def section_path_errors(sections: dict, models: dict) -> list[str]:
    """A MOSS / Anima section must hold its own model paths (not General Args' base model / VAE)."""
    sections = sections if isinstance(sections, dict) else {}
    errors = []
    for name, paths in SECTION_PATHS.items():
        section = sections.get(name)
        if not isinstance(section, dict) or not section:
            continue
        for key, label in paths:
            if not _filled(section.get(key)):
                errors.append(f"{SECTION_TITLES[name]}: {label} ({key}) is empty; General Args' paths are not used")
    anima = sections.get("anima_args")
    if models.get("anima38") and isinstance(anima, dict) and anima and not _filled(anima.get("qwen35_path")):
        errors.append("Anima Args: the Anima 3.8B connector needs the Qwen3.5 Model (qwen35_path)")
    return errors


def tokenizer_errors(models: dict, output_args: dict, environ=None) -> list[str]:
    environ = os.environ if environ is None else environ
    errors = []
    for flag, arg, env, label in TOKENIZER_RULES:
        if models.get(flag) and not _filled(output_args.get(arg)) and not environ.get(env):
            errors.append(f"{label} is required: set the folder with tokenizer.json, or {env} for the backend")
    return errors


def text_encoder_cache_errors(models: dict, output_args: dict) -> list[str]:
    cached = bool(output_args.get("cache_text_encoder_outputs"))
    if models["anima38"] and not cached:
        return [TE_CACHE_ERROR]
    if models["moss"] and not (cached or output_args.get("cache_text_encoder_outputs_to_disk")):
        return [TE_CACHE_ERROR]
    return []


def drop_gui_only_keys(output_args: dict) -> dict:
    for key in GUI_ONLY_KEYS:
        output_args.pop(key, None)
    return output_args


def keep_empty(arg: str, val) -> bool:
    """True for an empty string that must reach the trainer as "" (see KEEP_EMPTY)."""
    return arg in KEEP_EMPTY and isinstance(val, str) and val.strip() == ""
