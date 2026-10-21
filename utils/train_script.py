"""Which sd-scripts training script a /train request runs (pure; no torch, no server state)."""

LORA_SCRIPTS = {
    None: "train_network.py",
    "sdxl": "sdxl_train_network.py",
    "flux": "flux_train_network.py",
    "anima": "anima_train_network.py",
    "moss": "moss_train_network.py",
}
TEXTUAL_INVERSION_SCRIPTS = {
    None: "train_textual_inversion.py",
    "sdxl": "sdxl_train_textual_inversion.py",
}


def pick_train_script(
    train_type: str,
    sdxl: bool = False,
    flux: bool = False,
    anima: bool = False,
    moss: bool = False,
) -> str | None:
    """The script for this train mode and model type, or None when the combination is not supported
    (an unknown train mode, more than one model type, or a model type the train mode has no script for)."""
    model_types = [name for name, on in (("sdxl", sdxl), ("flux", flux), ("anima", anima), ("moss", moss)) if on]
    if len(model_types) > 1:
        return None
    model_type = model_types[0] if model_types else None
    if train_type == "lora":
        return LORA_SCRIPTS.get(model_type)
    if train_type == "textual_inversion":
        return TEXTUAL_INVERSION_SCRIPTS.get(model_type)
    return None
