"""utils.validation.validate(), the /validate handler's work, on a real dataset folder.

tests/test_caption_tags.py and tests/test_schedule_args.py cover the pure helpers; this covers that validate() uses
them: a caption extension that is not a caption text file is an error (a folder's .json was returned as tags), tags are
read only when the request asks for the tag count, an image path that is a file is an error (not a 500), and the
warmup ratio goes to the trainer as a ratio for a default scheduler.

Run from the backend folder with the training venv (validation.py imports sd-scripts' library, torch and the custom
optimizers; the test is skipped without them):

    sd_scripts\\venv\\Scripts\\python -m unittest tests.test_validate_request
"""

import copy
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAS_TRAINER = all(importlib.util.find_spec(name) is not None for name in ("library", "LoraEasyCustomOptimizer", "PIL"))
SECRET = "SECRET-NGROK-TOKEN"
CUSTOM_SCHEDULER = "LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts.CosineAnnealingWarmRestarts"


@unittest.skipUnless(HAS_TRAINER, "sd-scripts' library / LoraEasyCustomOptimizer / PIL are not importable here")
class ValidateRequestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils.validation import validate

        cls.validate = staticmethod(validate)

    def setUp(self):
        from PIL import Image

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._cwd = os.getcwd()
        os.chdir(self.root)  # validate() reads ./config.json (the backend's): none here
        self.data = self.root / "data"
        self.data.mkdir()
        for name in ("a", "b"):
            Image.new("RGB", (64, 64)).save(self.data / f"{name}.png")
        (self.data / "a.txt").write_text("1girl, smile", encoding="utf-8")
        (self.data / "b.txt").write_text("1girl, solo", encoding="utf-8")
        (self.data / "config.json").write_text('{"remote": true, "ngrok_token": "%s"}' % SECRET, encoding="utf-8")
        (self.root / "model.safetensors").write_bytes(b"x")
        (self.root / "out").mkdir()

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def request(self, extension=".txt", tag_occurrence=True, image_dir=None, general_extension=None,
                scheduler=None) -> dict:
        optimizer = {"optimizer_type": "AdamW", "lr_scheduler": "cosine", "warmup_ratio": 0.1, **(scheduler or {})}
        dataset_general = {"resolution": 64, "batch_size": 1, "enable_bucket": False}
        if general_extension:
            dataset_general["caption_extension"] = general_extension
        return copy.deepcopy({
            "args": {
                "general_args": {"pretrained_model_name_or_path": str(self.root / "model.safetensors"),
                                 "max_train_epochs": 10, "mixed_precision": "bf16"},
                "optimizer_args": optimizer,
                "saving_args": {"output_dir": str(self.root / "out"), "output_name": "x",
                                "tag_occurrence": tag_occurrence},
            },
            "dataset": {
                "general_args": dataset_general,
                "subsets": [{"name": "s", "image_dir": str(image_dir or self.data), "num_repeats": 1,
                             "caption_extension": extension}],
            },
        })

    def run_validate(self, request: dict):
        passed, _sdxl, errors, args, dataset, tags = self.validate(request)
        self.assertNotIn(SECRET, repr((errors, args, dataset, tags)))
        return passed, errors, args, tags

    def test_tags_only_when_the_tag_count_is_asked_for(self):
        passed, errors, args, tags = self.run_validate(self.request())
        self.assertTrue(passed, errors)
        self.assertEqual(tags, {"1girl": 2, "smile": 1, "solo": 1})

        passed, errors, _, tags = self.run_validate(self.request(tag_occurrence=False))
        self.assertTrue(passed, errors)
        self.assertEqual(tags, {})

    def test_non_caption_extensions_are_errors(self):
        for extension in (".json", "json", ".toml", ".yaml", ".py", ".env"):
            with self.subTest(extension=extension):
                passed, errors, _, tags = self.run_validate(self.request(extension=extension))
                self.assertFalse(passed)
                self.assertEqual(tags, {})
                self.assertTrue(any("of 's' is not a caption text file" in error for error in errors), errors)
        passed, errors, _, tags = self.run_validate(self.request(general_extension=".json"))
        self.assertFalse(passed)
        self.assertEqual(tags, {})
        self.assertTrue(any("(general) is not a caption text file" in error for error in errors), errors)

    def test_image_dir_that_is_a_file(self):
        passed, errors, _, tags = self.run_validate(self.request(image_dir=self.data / "a.png"))
        self.assertFalse(passed)
        self.assertEqual(tags, {})
        self.assertIn("Image directory path for 's' is not a folder", errors)

    def test_warmup_ratio(self):
        _, errors, args, _ = self.run_validate(self.request())
        self.assertNotIn("warmup_ratio", args, errors)
        self.assertEqual(args["lr_warmup_steps"], 0.1)  # the trainer turns it into steps
        self.assertIsInstance(args["lr_warmup_steps"], float)

        # a custom restart scheduler gets fractions of the run, which the trainer's get_scheduler_fix turns into steps
        # with its real count (2 cycles: each half the run; warmup 10 % of the run split over the cycles)
        _, errors, args, _ = self.run_validate(self.request(scheduler={
            "lr_scheduler_type": CUSTOM_SCHEDULER, "lr_scheduler_num_cycles": 2, "lr_scheduler_args": {"min_lr": "1e-6"}
        }))
        self.assertNotIn("lr_warmup_steps", args, errors)
        self.assertIn("warmup_steps=0.1", args["lr_scheduler_args"])
        self.assertIn("first_cycle_max_steps=0.5", args["lr_scheduler_args"])


if __name__ == "__main__":
    unittest.main()
