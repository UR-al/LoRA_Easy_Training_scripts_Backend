"""Caption files /validate reads for the tag count (utils/caption_tags.py).

/validate took the folder and the caption extension from the request and returned the text of every file with that
extension as tags, on every request: a folder with ".json" returned config.json's ngrok token. Now the extension
must be a caption text extension, tags are read only when the request asks for the tag count, and only from the
caption beside an image (<image stem><extension>, what the trainer reads) that stays inside the folder.

Run from the backend folder with any Python 3.10+:

    python -m unittest tests.test_caption_tags
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.caption_tags import (  # noqa: E402
    CAPTION_EXTENSIONS,
    caption_extension_errors,
    count_tags,
    wants_tag_count,
)


class CaptionTagsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name: str, text: str = "", folder: Path | None = None) -> Path:
        path = (folder or self.data) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_counts_captions_beside_images(self):
        self.write("a.png")
        self.write("a.txt", "1girl, solo, smile")
        self.write("b.jpg")
        self.write("b.txt", "1girl, outdoors")
        self.write("notes.txt", "not a caption: no image")  # no image of that name: not read
        tags = count_tags([{"image_dir": str(self.data), "caption_extension": ".txt"}])
        self.assertEqual(tags, {"1girl": 2, "solo": 1, "smile": 1, "outdoors": 1})
        self.assertEqual(next(iter(tags)), "1girl")  # most frequent first, as the GUI writes them

    def test_json_next_to_nothing_is_never_read(self):
        """The reviewer's case: a folder with a .json token file and no images."""
        self.write("config.json", '{"ngrok_token": "secret-token"}')
        self.assertEqual(count_tags([{"image_dir": str(self.data), "caption_extension": ".json"}]), {})
        # even beside an image, a non-caption extension is not read
        self.write("config.png")
        self.assertEqual(count_tags([{"image_dir": str(self.data), "caption_extension": ".json"}]), {})

    def test_validation_and_missing_folders_are_skipped(self):
        self.write("a.png")
        self.write("a.txt", "tag")
        subsets = [
            {"image_dir": str(self.data), "caption_extension": ".txt", "is_val": True},
            {"image_dir": str(self.root / "missing"), "caption_extension": ".txt"},
            {"image_dir": str(self.data / "a.png"), "caption_extension": ".txt"},  # a file, not a folder
        ]
        self.assertEqual(count_tags(subsets), {})

    def test_extension_without_dot_and_default(self):
        self.write("a.webp")
        self.write("a.caption", "x, y")
        self.write("a.txt", "z")
        self.assertEqual(count_tags([{"image_dir": str(self.data), "caption_extension": "caption"}]), {"x": 1, "y": 1})
        self.assertEqual(count_tags([{"image_dir": str(self.data)}]), {"x": 1, "y": 1})  # the trainer's default
        self.assertEqual(count_tags([{"image_dir": str(self.data)}], default_extension=".txt"), {"z": 1})

    def test_caption_symlink_out_of_the_folder_is_not_read(self):
        outside = self.write("secret.txt", "secret", folder=self.root)
        self.write("a.png")
        try:
            os.symlink(outside, self.data / "a.txt")
        except (OSError, NotImplementedError):  # Windows without Developer Mode
            self.skipTest("cannot create symlinks here")
        self.assertEqual(count_tags([{"image_dir": str(self.data), "caption_extension": ".txt"}]), {})

    def test_caption_extension_errors(self):
        dataset = {
            "general": {"resolution": 1024},
            "subsets": [
                {"name": "ok", "image_dir": "x", "caption_extension": ".txt"},
                {"name": "upper", "image_dir": "x", "caption_extension": ".TXT"},
                {"name": "plain", "image_dir": "x", "caption_extension": "caption"},
                {"name": "default", "image_dir": "x"},
                {"name": "json", "image_dir": "x", "caption_extension": ".json"},
                {"name": "env", "image_dir": "x", "caption_extension": "env"},
            ],
        }
        errors = caption_extension_errors(dataset)
        self.assertEqual(len(errors), 2, errors)
        self.assertIn("'json'", errors[0])
        self.assertIn("'env'", errors[1])
        for bad in (".json", ".toml", ".yaml", ".py", ".env", ".npz", ".safetensors"):
            self.assertNotIn(bad, CAPTION_EXTENSIONS)
        self.assertEqual(caption_extension_errors({"general": {"caption_extension": ".toml"}, "subsets": []}),
                         ["Caption Extension '.toml' (general) is not a caption text file: use one of "
                          + ", ".join(CAPTION_EXTENSIONS)])

    def test_wants_tag_count(self):
        self.assertTrue(wants_tag_count({"saving_args": {"tag_occurrence": True}}))
        self.assertTrue(wants_tag_count({"saving_args": {"tag_occurrence": "True"}}))
        self.assertFalse(wants_tag_count({"saving_args": {"tag_occurrence": False}}))
        self.assertFalse(wants_tag_count({"saving_args": {"output_name": "x"}}))
        self.assertFalse(wants_tag_count({"saving_args": None}))
        self.assertFalse(wants_tag_count(None))


if __name__ == "__main__":
    unittest.main()
