"""Scheduler args of /validate: the GUI's warmup ratio and restart cycles (pure; validation.py imports torch).

The backend's step count (validation.calculate_steps) is an estimate: it uses the global batch size, while the
trainer batches each folder with its own batch size (and knows the validation split, resolution pools and the number
of processes). So nothing here is converted to steps any more:
- a default scheduler gets the ratio itself, lr_warmup_steps = <ratio> as a float, which the trainer turns into steps
  with the length of the DataLoader it built (library/train_util.get_scheduler_fix: warmup =
  fraction_to_steps(ratio, steps), the floor of ratio * steps after rounding away float noise, so 0.29 of 100 is 29);
- the custom restart schedulers (lr_scheduler_type, e.g. LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts) get
  warmup_steps=<ratio> and first_cycle_max_steps=<1 / cycles> as floats in [0, 1]; get_scheduler_fix converts them with
  the real step count (first cycle = max(1, ceil(round(fraction * steps, 6))),
  warmup = fraction_to_steps(ratio, steps) // cycles, i.e. the same rule per cycle).
  Integers in lr_scheduler_args stay absolute steps there, so hand-written TOMLs are unchanged.
estimate_steps is kept in the signatures for callers but is no longer called.
"""

from typing import Callable


def _fraction(value) -> float:
    return min(1.0, max(0.0, float(value)))


def apply_warmup_ratio(args: dict, estimate_steps: Callable[[], int]) -> None:
    if "warmup_ratio" not in args:
        return
    ratio = _fraction(args.pop("warmup_ratio"))
    if "lr_scheduler_type" in args:
        args.setdefault("lr_scheduler_args", []).append(f"warmup_steps={ratio!r}")
    else:
        args["lr_warmup_steps"] = ratio


def apply_restart_cycles(args: dict, estimate_steps: Callable[[], int]) -> None:
    if "lr_scheduler_num_cycles" not in args or "lr_scheduler_type" not in args:
        return
    cycles = max(1, int(args["lr_scheduler_num_cycles"]))
    args.setdefault("lr_scheduler_args", []).append(f"first_cycle_max_steps={1.0 / cycles!r}")
