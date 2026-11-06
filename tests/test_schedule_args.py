"""Warmup ratio and restart cycles of /validate (utils/schedule_args.py).

The GUI's step estimate (calculate_steps) uses the global batch size, while the trainer batches each folder with its
own batch size: 8 images, global batch 4, folder batch 1, 10 epochs is 20 steps for the GUI and 80 for the trainer,
so a 10 % warmup became 2 steps instead of 8. Default schedulers now get the ratio itself (lr_warmup_steps = 0.1),
which the trainer turns into steps with its real DataLoader length. The custom restart schedulers get warmup_steps and
first_cycle_max_steps as float fractions, which sd-scripts' get_scheduler_fix converts the same way
(sd_scripts/tests/test_custom_scheduler_fraction_args.py). The estimate is never used.

Run from the backend folder with any Python 3.10+:

    python -m unittest tests.test_schedule_args
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.schedule_args import apply_restart_cycles, apply_warmup_ratio  # noqa: E402


def no_estimate():
    raise AssertionError("the step estimate must not be used for a default scheduler")


def trainer_warmup_steps(lr_warmup_steps, num_training_steps: int) -> int:
    """library/train_util.get_scheduler_fix: a float is a ratio of the trainer's own step count, turned into steps by
    fraction_to_steps (the floor of ratio * steps after rounding away float noise)."""
    if isinstance(lr_warmup_steps, float):
        return int(round(lr_warmup_steps * num_training_steps, 6))
    return lr_warmup_steps


class WarmupRatioTest(unittest.TestCase):
    def test_default_scheduler_gets_the_ratio(self):
        args = {"warmup_ratio": 0.1, "lr_scheduler": "cosine", "max_train_epochs": 10}
        apply_warmup_ratio(args, no_estimate)
        self.assertNotIn("warmup_ratio", args)
        self.assertIsInstance(args["lr_warmup_steps"], float)
        self.assertEqual(args["lr_warmup_steps"], 0.1)
        # the reviewer's dataset: the trainer counts 80 updates, so the warmup is 8 steps (the GUI's 20 gave 2)
        self.assertEqual(trainer_warmup_steps(args["lr_warmup_steps"], 80), 8)

    def test_float_noise_does_not_lose_a_step(self):
        # 0.29 * 100 = 28.999999999999996: a plain int() gave 28
        args = {"warmup_ratio": 0.29, "lr_scheduler": "cosine"}
        apply_warmup_ratio(args, no_estimate)
        self.assertEqual(args["lr_warmup_steps"], 0.29)
        self.assertEqual(trainer_warmup_steps(args["lr_warmup_steps"], 100), 29)

    def test_whole_number_ratio_stays_a_ratio(self):
        args = {"warmup_ratio": 1, "lr_scheduler": "linear"}
        apply_warmup_ratio(args, no_estimate)
        self.assertIsInstance(args["lr_warmup_steps"], float)
        self.assertEqual(trainer_warmup_steps(args["lr_warmup_steps"], 80), 80)

    def test_max_train_steps_does_not_matter_for_the_ratio(self):
        args = {"warmup_ratio": 0.25, "lr_scheduler": "cosine_with_restarts", "lr_scheduler_num_cycles": 3,
                "max_train_steps": 1000}
        apply_warmup_ratio(args, no_estimate)
        self.assertEqual(args["lr_warmup_steps"], 0.25)
        self.assertEqual(args["lr_scheduler_num_cycles"], 3)

    def test_custom_scheduler_gets_the_ratio_as_a_float(self):
        # the trainer (get_scheduler_fix) turns warmup_steps=<float> into fraction_to_steps(0.1, steps) // cycles,
        # 8 // 2 = 4 for 80 steps
        args = {"warmup_ratio": 0.1, "lr_scheduler_type": "LoraEasyCustomOptimizer.X.X", "lr_scheduler_num_cycles": 2,
                "lr_scheduler_args": ["min_lr=1e-06"]}
        apply_warmup_ratio(args, no_estimate)
        self.assertNotIn("lr_warmup_steps", args)
        self.assertNotIn("warmup_ratio", args)
        self.assertEqual(args["lr_scheduler_args"], ["min_lr=1e-06", "warmup_steps=0.1"])

    def test_custom_scheduler_whole_number_ratio_stays_a_float(self):
        # an int 1 would mean "1 step" to the trainer
        args = {"warmup_ratio": 1, "lr_scheduler_type": "LoraEasyCustomOptimizer.X.X"}
        apply_warmup_ratio(args, no_estimate)
        self.assertEqual(args["lr_scheduler_args"], ["warmup_steps=1.0"])

    def test_without_warmup_ratio_nothing_changes(self):
        args = {"lr_scheduler": "cosine"}
        apply_warmup_ratio(args, no_estimate)
        self.assertEqual(args, {"lr_scheduler": "cosine"})


class RestartCyclesTest(unittest.TestCase):
    def test_custom_scheduler_first_cycle_is_a_fraction(self):
        # the trainer turns it into ceil(steps / 3) with its real step count
        args = {"lr_scheduler_type": "LoraEasyCustomOptimizer.X.X", "lr_scheduler_num_cycles": 3,
                "lr_scheduler_args": []}
        apply_restart_cycles(args, no_estimate)
        self.assertEqual(args["lr_scheduler_args"], [f"first_cycle_max_steps={1 / 3!r}"])
        self.assertEqual(args["lr_scheduler_num_cycles"], 3)  # the trainer reads it too

    def test_one_cycle_is_the_whole_run(self):
        args = {"lr_scheduler_type": "LoraEasyCustomOptimizer.X.X", "lr_scheduler_num_cycles": 1}
        apply_restart_cycles(args, no_estimate)
        self.assertEqual(args["lr_scheduler_args"], ["first_cycle_max_steps=1.0"])

    def test_default_scheduler_cycles_are_left_to_the_trainer(self):
        args = {"lr_scheduler": "cosine_with_restarts", "lr_scheduler_num_cycles": 3}
        apply_restart_cycles(args, no_estimate)
        self.assertEqual(args, {"lr_scheduler": "cosine_with_restarts", "lr_scheduler_num_cycles": 3})


if __name__ == "__main__":
    unittest.main()
