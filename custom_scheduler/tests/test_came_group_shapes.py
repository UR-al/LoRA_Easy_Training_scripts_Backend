"""CAME group_shapes (2026-11-14): same-shape factored parameters stepped together in stacked chunks.

What is guarded here:
  * flag off: bit-identical to came.py.bak_20261114_before_shapebatch for unaffected options;
  * flag on: the same math per parameter - parameters and state within a few ulps of the per-parameter path (only the
    summation order of some CUDA reductions changes with the number of outputs), and the weights actually move under a
    TENSOR lr warmed up from 0 in place (the LyCORIS + constant_with_warmup case that froze the first 11-14 hotfix);
  * the configurations it does not cover (bf16 params, bf16 state storage) fall back bit for bit;
  * state_dict keeps the per-parameter format, and save -> load -> resume equals an uninterrupted run.

The grouped path only runs for CUDA parameters with fp32 state on the same device, so these tests need a GPU (tiny:
a few MB).
"""

import copy
import importlib.machinery
import importlib.util
import io
import os
import sys
import types
import unittest

import torch

_PKG_DIR = os.path.join(os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer")
try:
    from LoraEasyCustomOptimizer import came as came_mod
except ImportError:
    # The package __init__ pulls in optional dependencies; came.py only needs torch, pytorch_optimizer and .utils.
    _pkg = types.ModuleType("LoraEasyCustomOptimizer")
    _pkg.__path__ = [_PKG_DIR]
    sys.modules["LoraEasyCustomOptimizer"] = _pkg
    from LoraEasyCustomOptimizer import came as came_mod

CAME = came_mod.CAME
_BACKUP = os.path.join(_PKG_DIR, "came.py.bak_20261114_before_shapebatch")

DEV = "cuda"
# LoKr w1 / w2 / dora_scale-like shapes of the Anima run (scaled-down counts), a singleton shape (per-parameter path),
# a 3-D shape and an unfactored one.
SHAPES = ([(4, 4)] * 6 + [(64, 128)] * 3 + [(96, 32)] * 3 + [(256, 1)] * 4 + [(128, 128)] * 1 + [(3, 4, 5)] * 2
          + [(8,)] * 2)
BASE = dict(betas=(0.9, 0.999, 0.9999), weight_decay=0.1, weight_decouple=True, fixed_decay=False,
            update_strategy="cautious", clip_threshold=1.0, eps1=1e-30, eps2=1e-16,
            state_storage_device=DEV, state_storage_dtype="float32")
STEPS, WARM = 10, 4


def _load_backup():
    name = "LoraEasyCustomOptimizer.came_before_shapebatch"
    loader = importlib.machinery.SourceFileLoader(name, _BACKUP)
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    mod.__package__ = "LoraEasyCustomOptimizer"
    loader.exec_module(mod)
    return mod


def _make(cls, kw, shapes=SHAPES, param_dtype=torch.float32):
    gen = torch.Generator(device=DEV).manual_seed(0)
    params = [torch.nn.Parameter((torch.randn(s, generator=gen, device=DEV) * 0.02).to(param_dtype)) for s in shapes]
    opt = cls([{"params": params, "lr": torch.tensor(2e-6)}], lr=2e-6, **kw)  # LyCORIS: tensor lr
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / WARM))  # warmup from 0, changed in place
    return params, opt, sched


def _steps(params, opt, sched, k0, k1):
    for k in range(k0, k1):
        gen = torch.Generator(device=DEV).manual_seed(1000 + k)
        for p in params:
            p.grad = (torch.randn(p.shape, generator=gen, device=DEV) * 1e-3).to(p.dtype)
        torch.manual_seed(k)  # stochastic rounding of bf16 write-backs
        opt.step()
        sched.step()


def _run(cls, kw, **make_kw):
    params, opt, sched = _make(cls, kw, **make_kw)
    init = [p.detach().clone() for p in params]
    _steps(params, opt, sched, 0, STEPS)
    moved = sum(int((p.detach() != i).sum()) for p, i in zip(params, init))
    return params, opt, moved, sum(p.numel() for p in params)


def _pairs(pa, oa, pb, ob):
    out = [(a.detach(), b.detach()) for a, b in zip(pa, pb)]
    for a, b in zip(pa, pb):
        for k, v in oa.state[a].items():
            if torch.is_tensor(v):
                out.append((v, ob.state[b][k]))
    return out


def _max_ulp(pairs):
    """max |a-b| in ulps of each tensor's largest |element| (fp32)."""
    worst = 0.0
    for a, b in pairs:
        a, b = a.float(), b.float()
        if a.numel() == 0:
            continue
        m = a.abs().max()
        u = torch.nextafter(m, torch.tensor(float("inf"), device=m.device)) - m
        worst = max(worst, float((a - b).abs().max() / u))
    return worst


def _identical(pairs):
    return all(torch.equal(a, b) for a, b in pairs)


def _roundtrip(sd):
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


@unittest.skipUnless(torch.cuda.is_available(), "the grouped path runs on CUDA parameters only")
class CameGroupShapesTest(unittest.TestCase):
    def test_flag_off_is_bit_identical_to_backup(self):
        old = _load_backup()
        # BUG-08 corrected the AMS variance units; the backup is no longer its
        # numerical oracle. test_came_verified_bugs.py checks the AMS formula.
        for extra in ({}, {"foreach": True}, {"update_strategy": "both"}):
            kw = dict(BASE, **extra)
            pa, oa, ma, total = _run(old.CAME, kw)
            pb, ob, mb, _ = _run(CAME, kw)  # group_shapes defaults to False
            self.assertTrue(_identical(_pairs(pa, oa, pb, ob)), extra)
            self.assertEqual(ma, mb)
            self.assertGreater(mb, total // 2)

    def test_flag_on_same_math_within_ulps_and_moves(self):
        configs = (
            {},                                                     # the run's config: decoupled wd 0.1, cautious
            {"update_strategy": "both", "ams_bound": True},
            {"update_strategy": "grams", "cautious_weight_decay": True},
            {"weight_decouple": False, "update_strategy": "unmodified"},
            {"fixed_decay": True, "foreach": True},
        )
        for extra in configs:
            kw = dict(BASE, **extra)
            pa, oa, ma, total = _run(CAME, kw)
            pb, ob, mb, _ = _run(CAME, dict(kw, group_shapes=True))
            worst = _max_ulp(_pairs(pa, oa, pb, ob))
            self.assertLessEqual(worst, 16.0, extra)  # measured <= 3 ulp; 16 still means "same math"
            self.assertGreater(mb, total // 2, extra)
            self.assertLessEqual(abs(ma - mb), 4, extra)
            if extra.get("weight_decouple") is False:
                # the per-parameter path adds the L2 term to p.grad in place; the grouped path keeps that side effect
                self.assertLessEqual(_max_ulp([(a.grad, b.grad) for a, b in zip(pa, pb)]), 16.0)

    def test_handles_only_same_shape_factored_fp32_buckets(self):
        params, opt, sched = _make(CAME, dict(BASE, group_shapes=True))
        for p in params:
            p.grad = torch.ones_like(p)
        opt._step_serial = 1
        opt._compile_core_fns()
        handled = opt._grouped_factored_step(opt.param_groups[0], 0)
        expect = {id(p) for p in params if p.dim() >= 2 and SHAPES.count(tuple(p.shape)) >= 2}
        self.assertEqual(handled, expect)  # not the (8,) params, not the single (128, 128)

    def test_chunks_respect_max_numel(self):
        opt = CAME([torch.nn.Parameter(torch.zeros(2, 2, device=DEV))], group_shapes=True, group_shapes_max_numel=1000,
                   state_storage_device=DEV, state_storage_dtype="float32")
        ps = [torch.zeros(10, 10) for _ in range(25)]  # 2500 elements -> 3 chunks of 9, 9, 7
        self.assertEqual([len(c) for c in opt._grouped_chunks(ps)], [9, 9, 7])
        ps = [torch.zeros(40, 40) for _ in range(3)]  # each over the cap -> chunks of one -> per-parameter path
        self.assertEqual(opt._grouped_chunks(ps), [])
        # chunked vs one chunk: same math
        pa, oa, _, total = _run(CAME, dict(BASE, group_shapes=True))
        pb, ob, mb, _ = _run(CAME, dict(BASE, group_shapes=True, group_shapes_max_numel=2 * 64 * 128))
        self.assertLessEqual(_max_ulp(_pairs(pa, oa, pb, ob)), 16.0)
        self.assertGreater(mb, total // 2)

    def test_uncovered_configs_fall_back_bit_identically(self):
        cases = (
            (dict(BASE, state_storage_dtype="bfloat16"), torch.float32),  # bf16 state: staging + stochastic rounding
            (dict(BASE, kahan_sum=True), torch.bfloat16),                 # bf16 params with Kahan
            (dict(BASE), torch.float16),                                  # fp16 params
        )
        for kw, dtype in cases:
            pa, oa, _, _ = _run(CAME, kw, param_dtype=dtype)
            pb, ob, _, _ = _run(CAME, dict(kw, group_shapes=True), param_dtype=dtype)
            self.assertTrue(_identical(_pairs(pa, oa, pb, ob)), (kw.get("state_storage_dtype"), dtype))

    def test_state_dict_format_unchanged_and_resume(self):
        _, off_opt, _, _ = _run(CAME, BASE)
        _, on_opt, _, _ = _run(CAME, dict(BASE, group_shapes=True))
        sa, sb = off_opt.state_dict(), on_opt.state_dict()
        self.assertEqual(sa["param_groups"][0].keys(), sb["param_groups"][0].keys())  # no new group key
        self.assertEqual(sa["state"].keys(), sb["state"].keys())
        for i, st in sa["state"].items():
            self.assertEqual(st.keys(), sb["state"][i].keys())
            for k, v in st.items():
                if torch.is_tensor(v):
                    self.assertEqual((v.shape, v.dtype), (sb["state"][i][k].shape, sb["state"][i][k].dtype), k)

        half = STEPS // 2
        ref_p, ref_o, ref_s = _make(CAME, dict(BASE, group_shapes=True))
        _steps(ref_p, ref_o, ref_s, 0, STEPS)
        for saved_with_flag in (True, False):
            sp, so, ss = _make(CAME, dict(BASE, group_shapes=saved_with_flag))
            _steps(sp, so, ss, 0, half)
            saved_p = [p.detach().clone() for p in sp]
            saved_opt, saved_sched = _roundtrip(copy.deepcopy(so.state_dict())), ss.state_dict()

            rp, ro, rs = _make(CAME, dict(BASE, group_shapes=True))
            with torch.no_grad():
                for p, v in zip(rp, saved_p):
                    p.copy_(v)
            ro.load_state_dict(saved_opt)
            rs.load_state_dict(saved_sched)
            self.assertTrue(ro.group_shapes)  # the flag is not part of the saved state, so a resume keeps it
            _steps(rp, ro, rs, half, STEPS)
            pairs = _pairs(ref_p, ref_o, rp, ro)
            if saved_with_flag:
                self.assertTrue(_identical(pairs))
            else:
                self.assertLessEqual(_max_ulp(pairs), 16.0)  # an old (flag-off) state resumes under the flag

    def test_compile_step_grouped_matches(self):
        kw = dict(BASE, compile_step=True)
        pa, oa, _, total = _run(CAME, kw)
        pb, ob, mb, _ = _run(CAME, dict(kw, group_shapes=True))
        self.assertLessEqual(_max_ulp(_pairs(pa, oa, pb, ob)), 16.0)
        self.assertGreater(mb, total // 2)
        self.assertIsNot(ob._compiled_grouped, CAME._core_factored_grouped_fp32)  # compiled, not the eager function


if __name__ == "__main__":
    unittest.main()
