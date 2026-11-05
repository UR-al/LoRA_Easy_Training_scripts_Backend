"""CAME foreach=True gives the same parameters and optimizer state as the per-parameter path (foreach=False).

The foreach path handles unfactored parameters with numel >= 16 (1-D vectors); matrices and small or 0-D tensors
always go through the per-parameter loop. It used to read the state before creating it (KeyError: 'exp_avg' on the
first step) and used the stored first moment as the final-update buffer, so the LR scale (and the cautious mask)
were written into exp_avg: after two steps at lr=0.01 the stored moment was ~0.001 instead of ~0.19.

These tests run on CPU: CAME's CUDA-only calls (``torch.cuda.current_device``, ``torch.cuda.synchronize``,
``Tensor.pin_memory``) are replaced with CPU stand-ins. State is stored in float32 so no stochastic rounding is
involved and the two paths can be compared bit for bit.
"""

import os
import sys
import types
from unittest import mock

import pytest
import torch

_PKG_DIR = os.path.join(os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer")
try:
    from LoraEasyCustomOptimizer.came import CAME
except ImportError:
    # The package __init__ pulls in optional dependencies; came.py only needs torch,
    # pytorch_optimizer and .utils, so load it through a bare package module.
    _pkg = types.ModuleType("LoraEasyCustomOptimizer")
    _pkg.__path__ = [_PKG_DIR]
    sys.modules["LoraEasyCustomOptimizer"] = _pkg
    from LoraEasyCustomOptimizer.came import CAME


@pytest.fixture(autouse=True)
def cpu_only_came():
    with mock.patch.object(torch.cuda, "current_device", lambda: torch.device("cpu")), \
         mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None), \
         mock.patch.object(torch.Tensor, "pin_memory", lambda self, *a, **k: self):
        yield


# 1-D vectors with numel >= 16 take the foreach path; the 7-element vector, the 0-D scalar and the matrices
# (factored) take the per-parameter path in both modes.
SHAPES = [(16,), (40,), (7,), (8, 4), (3, 5, 6), (24,), ()]

CONFIGS = {
    "defaults": {},
    "cautious_wd": {"weight_decay": 0.1, "update_strategy": "cautious"},
    "grams_coupled_wd": {"weight_decay": 0.05, "weight_decouple": False, "update_strategy": "grams"},
    "both_cwd_nfc": {"weight_decay": 0.1, "cautious_weight_decay": True, "update_strategy": "both",
                     "non_factored_confidence": True},
    "amsbound_nfc_fixed_decay": {"ams_bound": True, "non_factored_confidence": True, "fixed_decay": True,
                                 "weight_decay": 0.01},
    "grams_nfc": {"update_strategy": "grams", "non_factored_confidence": True, "betas": (0.8, 0.99, 0.999)},
}


def _run(foreach: bool, steps: int = 4, **kw):
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(shape)) for shape in SHAPES]
    opt = CAME(params, lr=1e-2, foreach=foreach, state_storage_dtype="float32", state_storage_device="cpu", **kw)
    for step in range(steps):
        gen = torch.Generator().manual_seed(100 + step)
        for p in params:
            p.grad = torch.randn(p.shape, generator=gen)
        opt.step()
    return params, opt


@pytest.mark.parametrize("config", list(CONFIGS.values()), ids=list(CONFIGS))
def test_foreach_matches_per_parameter_path(config):
    ref_params, ref_opt = _run(False, **config)
    fe_params, fe_opt = _run(True, **config)

    for shape, ref, fe in zip(SHAPES, ref_params, fe_params):
        assert torch.equal(ref, fe), f"param {shape}: max diff {(ref - fe).abs().max().item()}"

    for shape, ref, fe in zip(SHAPES, ref_params, fe_params):
        ref_state, fe_state = ref_opt.state[ref], fe_opt.state[fe]
        assert set(ref_state) == set(fe_state), f"state keys of {shape}"
        for key, value in ref_state.items():
            assert torch.equal(value, fe_state[key]), (
                f"state {key} of {shape}: max diff {(value - fe_state[key]).abs().max().item()}"
            )
    assert [g["step"] for g in ref_opt.param_groups] == [g["step"] for g in fe_opt.param_groups]


def test_first_foreach_step_creates_state_and_keeps_the_moment():
    """The reviewer's case: a length-16 fp32 vector, all-ones gradient, lr 0.01, two steps."""
    moments = {}
    for foreach in (False, True):
        p = torch.nn.Parameter(torch.zeros(16))
        opt = CAME([p], lr=0.01, foreach=foreach, state_storage_dtype="float32", state_storage_device="cpu")
        for _ in range(2):
            p.grad = torch.ones(16)
            opt.step()
        moments[foreach] = opt.state[p]["exp_avg"].clone()
    assert torch.allclose(moments[False], torch.full((16,), 0.19)), moments[False]
    assert torch.equal(moments[True], moments[False])


def test_foreach_with_bf16_state_storage_runs_and_tracks_the_reference():
    """Default storage (bf16, stochastic rounding): the paths draw their rounding noise in a different order, so the
    results are only close, not identical."""
    results = {}
    for foreach in (False, True):
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(shape)) for shape in SHAPES]
        opt = CAME(params, lr=1e-2, foreach=foreach, state_storage_dtype="bfloat16", state_storage_device="cpu",
                   update_strategy="cautious", weight_decay=0.1)
        for step in range(3):
            gen = torch.Generator().manual_seed(100 + step)
            for p in params:
                p.grad = torch.randn(p.shape, generator=gen)
            opt.step()
        results[foreach] = (params, opt)
    (ref_params, ref_opt), (fe_params, fe_opt) = results[False], results[True]
    for shape, ref, fe in zip(SHAPES, ref_params, fe_params):
        torch.testing.assert_close(fe, ref, rtol=2e-2, atol=1e-3, msg=f"param {shape}")
        for key, value in ref_opt.state[ref].items():
            torch.testing.assert_close(fe_opt.state[fe][key].float(), value.float(), rtol=3e-2, atol=1e-3,
                                       msg=f"state {key} of {shape}")
