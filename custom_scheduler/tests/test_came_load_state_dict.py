"""Resuming CAME keeps the state storage policy (state_storage_dtype / state_storage_device).

torch's ``Optimizer.load_state_dict`` casts every floating-point state tensor to the
parameter's dtype and device. For a bf16 parameter with ``state_storage_dtype=float32``
that turned the restored moments (and the always-fp32 factored statistics) into bf16,
and because CAME writes fp32 state back with a plain ``copy_``, the beta2/beta3 EMAs then
barely moved after a resume.

These tests run on CPU: CAME's CUDA-only calls (``torch.cuda.current_device``,
``torch.cuda.synchronize``, ``Tensor.pin_memory``) are replaced with CPU stand-ins.
"""

import copy
import io
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


FP32_KEYS = ("exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_res_row", "exp_avg_res_col", "kahan_comp")


def _make(params_init, **kw):
    params = [torch.nn.Parameter(t.clone()) for t in params_init]
    opt = CAME(params, lr=1e-2, weight_decay=0.1, update_strategy="cautious",
               state_storage_dtype="float32", state_storage_device="cpu", **kw)
    return params, opt


def _step(params, opt, seed):
    gen = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen).to(p.dtype)
    torch.manual_seed(seed)  # stochastic rounding of the bf16 write-back
    opt.step()


def _roundtrip(sd):
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("kahan_sum", [False, True])
def test_load_state_dict_keeps_storage_dtype_and_matches_uninterrupted_run(kahan_sum):
    torch.manual_seed(0)
    init = [torch.randn(8, 6).to(torch.bfloat16),   # factored
            torch.randn(20).to(torch.bfloat16)]     # unfactored
    ref_params, ref_opt = _make(init, kahan_sum=kahan_sum)
    _step(ref_params, ref_opt, 1)
    saved_params = [p.detach().clone() for p in ref_params]
    saved_sd = _roundtrip(copy.deepcopy(ref_opt.state_dict()))
    _step(ref_params, ref_opt, 2)

    res_params, res_opt = _make(saved_params, kahan_sum=kahan_sum)
    res_opt.load_state_dict(saved_sd)

    for p in res_params:
        st = res_opt.state[p]
        assert st["exp_avg"].dtype == torch.float32
        assert st["exp_avg"].device.type == "cpu"
        for k in ("exp_avg_sq",) + FP32_KEYS:
            if k in st:
                assert st[k].dtype == torch.float32, k
                assert st[k].device.type == "cpu", k
        if kahan_sum:
            assert "kahan_comp" in st

    _step(res_params, res_opt, 2)

    for a, b in zip(ref_params, res_params):
        assert a.dtype == b.dtype == torch.bfloat16
        assert torch.equal(a, b)
    for a, b in zip(ref_params, res_params):
        sa, sb = ref_opt.state[a], res_opt.state[b]
        assert sa.keys() == sb.keys()
        for k in sa:
            if torch.is_tensor(sa[k]):
                assert sa[k].dtype == sb[k].dtype, k
                assert torch.equal(sa[k], sb[k]), k


def test_load_state_dict_bf16_storage_policy_is_kept_for_fp32_param():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 6))
    opt = CAME([p], lr=1e-2, state_storage_dtype="bfloat16", state_storage_device="cpu")
    p.grad = torch.randn(8, 6)
    opt.step()
    sd = _roundtrip(copy.deepcopy(opt.state_dict()))

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = CAME([p2], lr=1e-2, state_storage_dtype="bfloat16", state_storage_device="cpu")
    opt2.load_state_dict(sd)
    st = opt2.state[p2]
    assert st["exp_avg"].dtype == torch.bfloat16
    for k in ("exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_res_row", "exp_avg_res_col"):
        assert st[k].dtype == torch.float32, k
        assert torch.equal(st[k], opt.state[p][k]), k
