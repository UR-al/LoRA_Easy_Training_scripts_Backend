"""Model-type rules of /validate (utils/model_args.py): required files, the section's own paths, tokenizer folders,
the TE cache rule, GUI-only keys and KEEP_EMPTY.

Run from the backend folder with any Python 3.10+:

    python -m unittest tests.test_model_args
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.model_args import (
    GUI_ONLY_KEYS,
    KEEP_EMPTY,
    TE_CACHE_ERROR,
    detect_models,
    drop_gui_only_keys,
    file_inputs,
    keep_empty,
    section_path_errors,
    text_encoder_cache_errors,
    tokenizer_errors,
)

MOSS_SECTION = {"pretrained_model_name_or_path": "dit", "vae": "vae", "moss_text_encoder": "te"}
ANIMA_SECTION = {"pretrained_model_name_or_path": "dit", "qwen3": "q", "vae": "vae"}


def required(models: dict) -> set:
    return {entry["name"] for entry in file_inputs(models) if entry["required"]}


def checked(models: dict) -> set:
    return {entry["name"] for entry in file_inputs(models)}


class ModelArgsTest(unittest.TestCase):
    def test_plain_model_as_before(self):
        models = detect_models({"general_args": {}}, {"pretrained_model_name_or_path": "x"})
        self.assertEqual(models, {"anima": False, "moss": False, "anima38": False})
        self.assertEqual(required(models), {"pretrained_model_name_or_path", "output_dir"})
        self.assertEqual(text_encoder_cache_errors(models, {}), [])

    def test_plain_anima_as_before(self):
        models = detect_models({}, {"qwen3": "q", "vae": "v"})
        self.assertEqual(models, {"anima": True, "moss": False, "anima38": False})
        self.assertEqual(required(models), {"pretrained_model_name_or_path", "qwen3", "vae", "output_dir"})
        self.assertEqual(text_encoder_cache_errors(models, {}), [])  # 0.6B Qwen3 can run uncached

    def test_anima_section_without_qwen3_requires_it(self):
        models = detect_models({"anima_args": {"vae": "v"}}, {"vae": "v"})
        self.assertIn("qwen3", required(models))

    def test_moss(self):
        models = detect_models({"moss_args": {"timestep_sampling": "moss"}}, {"timestep_sampling": "moss"})
        self.assertTrue(models["moss"])
        self.assertFalse(models["anima"])
        self.assertTrue({"moss_text_encoder", "vae", "pretrained_model_name_or_path"} <= required(models))
        self.assertNotIn("qwen3", required(models))
        self.assertIn("moss_tokenizer_dir", checked(models))
        self.assertNotIn("moss_tokenizer_dir", required(models))
        # also detected from the flattened key alone
        self.assertTrue(detect_models({}, {"moss_text_encoder": "te"})["moss"])

    def test_moss_needs_te_cache(self):
        models = detect_models({}, {"moss_text_encoder": "te"})
        self.assertEqual(text_encoder_cache_errors(models, {"moss_text_encoder": "te"}), [TE_CACHE_ERROR])
        self.assertEqual(
            text_encoder_cache_errors(models, {"moss_text_encoder": "te", "cache_text_encoder_outputs": True}), []
        )

    def test_anima38(self):
        flat = {"qwen3": "q", "vae": "v", "anima38_connector": True}
        models = detect_models({}, flat)
        self.assertTrue(models["anima38"])
        self.assertIn("qwen35_path", required(models))
        self.assertIn("qwen35_tokenizer_dir", checked(models))
        self.assertEqual(text_encoder_cache_errors(models, flat), [TE_CACHE_ERROR])
        self.assertEqual(text_encoder_cache_errors(models, {**flat, "cache_text_encoder_outputs": True}), [])
        self.assertNotIn("qwen35_path", required(detect_models({}, {"qwen3": "q"})))

    def test_moss_accepts_te_cache_to_disk_alone(self):
        # moss_train_network.py turns cache_text_encoder_outputs on itself; anima38 needs the flag
        flat = {"moss_text_encoder": "te", "cache_text_encoder_outputs_to_disk": True}
        self.assertEqual(text_encoder_cache_errors(detect_models({}, flat), flat), [])
        flat38 = {"qwen3": "q", "anima38_connector": True, "cache_text_encoder_outputs_to_disk": True}
        self.assertEqual(text_encoder_cache_errors(detect_models({}, flat38), flat38), [TE_CACHE_ERROR])

    def test_section_paths_are_not_taken_from_general(self):
        # the flatten loop drops the section's empty strings, so General's SDXL paths would stand in for them
        sections = {
            "general_args": {"pretrained_model_name_or_path": "sdxl.safetensors", "vae": "sdxl_vae.safetensors"},
            "moss_args": {"pretrained_model_name_or_path": "", "vae": "", "moss_text_encoder": "te",
                          "timestep_sampling": "moss"},
        }
        models = detect_models(sections, {"moss_text_encoder": "te"})
        errors = section_path_errors(sections, models)
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("DiT Model" in e for e in errors) and any("VAE Model" in e for e in errors))
        sections["moss_args"].update(MOSS_SECTION)
        self.assertEqual(section_path_errors(sections, models), [])

        anima = {"general_args": {"vae": "sdxl_vae"}, "anima_args": {**ANIMA_SECTION, "vae": " "}}
        self.assertEqual(len(section_path_errors(anima, detect_models(anima, {"qwen3": "q"}))), 1)
        # Anima 3.8B: qwen35_path in the section
        anima38 = {"anima_args": {**ANIMA_SECTION, "anima38_connector": True}}
        models38 = detect_models(anima38, {"qwen3": "q", "anima38_connector": True})
        self.assertEqual(len(section_path_errors(anima38, models38)), 1)
        anima38["anima_args"]["qwen35_path"] = "q35"
        self.assertEqual(section_path_errors(anima38, models38), [])
        # no section (a hand-made flat request): nothing to check here, the file checks still run
        self.assertEqual(section_path_errors({}, detect_models({}, {"moss_text_encoder": "te"})), [])

    def test_tokenizer_folders(self):
        moss = detect_models({}, {"moss_text_encoder": "te"})
        self.assertEqual(len(tokenizer_errors(moss, {}, environ={})), 1)
        self.assertEqual(tokenizer_errors(moss, {"moss_tokenizer_dir": "tok"}, environ={}), [])
        self.assertEqual(tokenizer_errors(moss, {}, environ={"MOSS_TOKENIZER_DIR": "tok"}), [])
        # Anima 3.8B: an empty Qwen3.5 tokenizer folder is allowed (the trainer uses
        # $ANIMA38_QWEN35_TOKENIZER_DIR or its own default, and stops with a clear error if neither exists)
        anima38 = detect_models({}, {"qwen3": "q", "anima38_connector": True})
        self.assertEqual(tokenizer_errors(anima38, {}, environ={}), [])
        self.assertEqual(tokenizer_errors(anima38, {}, environ={"ANIMA38_QWEN35_TOKENIZER_DIR": "t"}), [])
        self.assertEqual(tokenizer_errors(anima38, {"qwen35_tokenizer_dir": "t"}, environ={}), [])
        # plain Anima / SDXL: no tokenizer rule
        self.assertEqual(tokenizer_errors(detect_models({}, {"qwen3": "q"}), {}, environ={}), [])
        self.assertEqual(tokenizer_errors(detect_models({}, {}), {}, environ={}), [])

    def test_gui_only_keys_dropped(self):
        flat = {"save_toml": True, "save_toml_location": "x", "tag_occurrence": True, "tag_file_location": "y",
                "output_name": "out"}
        self.assertEqual(drop_gui_only_keys(flat), {"output_name": "out"})
        self.assertEqual(set(GUI_ONLY_KEYS), {"save_toml", "save_toml_location", "tag_occurrence", "tag_file_location"})

    def test_keep_empty(self):
        self.assertIn("moss_sample_uncond_skip_layers", KEEP_EMPTY)
        self.assertTrue(keep_empty("moss_sample_uncond_skip_layers", ""))
        self.assertFalse(keep_empty("moss_sample_uncond_skip_layers", "22"))
        self.assertFalse(keep_empty("output_name", ""))


if __name__ == "__main__":
    unittest.main()
