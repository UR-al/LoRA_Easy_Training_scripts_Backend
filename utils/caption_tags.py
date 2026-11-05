"""Caption files /validate checks and reads for the tag count (pure; validation.py imports torch, this does not).

/validate takes the dataset folders and the caption extension from the request. It used to read every file with that
extension in each folder and return its text as "tags" on every request, so a folder and ".json" returned e.g. the
backend's config.json (ngrok token) to whoever could reach the API. Now:
- every caption extension must be a caption text extension (CAPTION_EXTENSIONS); others are a validation error;
- the dataset folders must be real directories (validation.validate_subset; the backend has no configured data root,
  a remote backend's datasets can be anywhere, so the folders are the ones the request's own config names);
- tags are read only when the request asks for the tag count (saving_args.tag_occurrence), and only from the caption
  of an image: "<image stem><caption extension>" in the image's folder, which is the file the trainer reads, and only
  when it resolves inside that folder (a symlink out of it is not followed).
"""

from pathlib import Path

# Caption text extensions: the GUI offers .txt and .caption (the trainer's default); .tags / .cap are the other names
# taggers use. Never a config, script, key or model file (.json, .toml, .yaml, .py, .env, .npz, ...).
CAPTION_EXTENSIONS = (".txt", ".caption", ".tags", ".cap")
DEFAULT_CAPTION_EXTENSION = ".caption"  # sd-scripts --caption_extension default
# The trainer's image types (library/train_util.IMAGE_EXTENSIONS; .avif / .jxl when their Pillow plugins are there)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl")


def normalize_extension(extension) -> str:
    """As the trainer does: "txt" -> ".txt"."""
    extension = str(extension).strip()
    return extension if not extension or extension.startswith(".") else "." + extension


def is_caption_extension(extension) -> bool:
    return normalize_extension(extension).lower() in CAPTION_EXTENSIONS


def _filled(value) -> bool:
    return value is not None and not (isinstance(value, str) and value.strip() == "")


def caption_extension_errors(dataset: dict) -> list[str]:
    """dataset: {"general": {...}, "subsets": [{...}]} as validate_dataset_args builds it."""
    errors = []
    general = dataset.get("general") or {}
    allowed = ", ".join(CAPTION_EXTENSIONS)
    if _filled(general.get("caption_extension")) and not is_caption_extension(general["caption_extension"]):
        errors.append(f"Caption Extension '{general['caption_extension']}' (general) is not a caption text file: "
                      f"use one of {allowed}")
    for subset in dataset.get("subsets") or []:
        extension = subset.get("caption_extension")
        if _filled(extension) and not is_caption_extension(extension):
            name = subset.get("name", "subset")
            errors.append(f"Caption Extension '{extension}' of '{name}' is not a caption text file: "
                          f"use one of {allowed}")
    return errors


def wants_tag_count(sections) -> bool:
    """Whether the request asks for the tag count file (saving_args.tag_occurrence in the GUI's args sections)."""
    if not isinstance(sections, dict):
        return False
    for section in sections.values():
        if not isinstance(section, dict):
            continue
        value = section.get("tag_occurrence")
        if value is True or (isinstance(value, str) and value.strip().lower() == "true"):
            return True
    return False


def add_tags_from_text(text: str, tags: dict) -> None:
    for tag in text.replace(", ", ",").split(","):
        tags[tag] = tags.get(tag, 0) + 1


def _caption_files(folder: Path, extension: str):
    try:
        root = folder.resolve(strict=True)
    except OSError:
        return
    if not root.is_dir():
        return
    for image in sorted(root.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption = image.with_suffix(extension)
        if not caption.is_file():
            continue
        try:
            resolved = caption.resolve(strict=True)
        except OSError:
            continue
        if resolved.parent != root:
            continue
        yield resolved


def count_tags(subsets: list, default_extension: str = DEFAULT_CAPTION_EXTENSION) -> dict:
    """Tag -> count over the training subsets' captions, most frequent first."""
    tags = {}
    for subset in subsets:
        if subset.get("is_val"):
            continue
        extension = subset.get("caption_extension")
        extension = normalize_extension(extension if _filled(extension) else default_extension)
        if not is_caption_extension(extension) or not _filled(subset.get("image_dir")):
            continue
        for caption in _caption_files(Path(subset["image_dir"]), extension):
            with open(caption, "r", encoding="utf-8") as f:
                add_tags_from_text(f.read(), tags)
    return dict(sorted(tags.items(), key=lambda item: item[1], reverse=True))
