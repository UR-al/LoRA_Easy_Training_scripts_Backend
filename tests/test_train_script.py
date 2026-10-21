"""pick_train_script (utils/train_script.py): the script /train runs for each train mode and model type.

Run from the backend folder with any Python 3.10+:

    python -m unittest tests.test_train_script
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.train_script import pick_train_script


class PickTrainScriptTest(unittest.TestCase):
    def test_supported(self):
        cases = [
            (("lora", False, False, False, False), "train_network.py"),
            (("lora", True, False, False, False), "sdxl_train_network.py"),
            (("lora", False, True, False, False), "flux_train_network.py"),
            (("lora", False, False, True, False), "anima_train_network.py"),
            (("lora", False, False, False, True), "moss_train_network.py"),
            (("textual_inversion", False, False, False, False), "train_textual_inversion.py"),
            (("textual_inversion", True, False, False, False), "sdxl_train_textual_inversion.py"),
        ]
        for params, script in cases:
            with self.subTest(params=params):
                self.assertEqual(pick_train_script(*params), script)

    def test_unsupported(self):
        cases = [
            ("lora", False, False, True, True),  # anima + moss
            ("lora", False, True, False, True),  # flux + moss
            ("lora", True, False, False, True),  # sdxl + moss
            ("lora", True, True, False, False),  # sdxl + flux
            ("lora", False, True, True, False),  # flux + anima
            ("textual_inversion", False, True, False, False),
            ("textual_inversion", False, False, True, False),
            ("textual_inversion", False, False, False, True),
            ("dreambooth", False, False, False, False),
        ]
        for params in cases:
            with self.subTest(params=params):
                self.assertIsNone(pick_train_script(*params))

    def test_old_gui_without_moss_flag(self):
        # A GUI that does not send "moss" keeps working (moss defaults to False)
        self.assertEqual(pick_train_script("lora", False, False, True), "anima_train_network.py")


if __name__ == "__main__":
    unittest.main()
