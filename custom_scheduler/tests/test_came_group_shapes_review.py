"""CAME group_shapes: the reviewer's tests (2026-11-14), next to the implementer's test_came_group_shapes.py.

Why a second file: test_came_group_shapes.py compares final tensors of one parameter group whose gradients all have the
same scale. That cannot see a reduction leaking across the stacked parameters by a small amount, a parameter group
reading another group's lr, a parameter that is skipped by both paths, or what happens to the run when the set of
parameters that have a gradient changes. Those are checked here, against the per-parameter path (flag off) and against
came.py.bak_20261114_before_shapebatch.

CameGroupShapesReviewTest      - behaviour that was verified to be correct (regression guards).
CameGroupShapesOpenIssuesTest  - four problems found by the review, each written as the behaviour it SHOULD have and
                                 first marked expectedFailure; all four fixed in came.py the same day and the decorators
                                 removed, so they are plain regression tests now. None of them changes the numbers of the Anima LoKr run as configured
                                 on 2026-11-14 (constant set of 1,920 parameters with gradients, flag spelled True).

GPU: tiny (a few MB), the compile tests take a few seconds each.
"""

import importlib.machinery
import importlib.util
import io
import itertools
import os
import sys
import types
import unittest
from unittest import mock

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
# the optimizer_args of config_anima29.toml (2026-11-14)
BASE = dict(betas=(0.9, 0.999, 0.9999), weight_decay=0.1, weight_decouple=True, fixed_decay=False,
            update_strategy="cautious", clip_threshold=1.0, eps1=1e-30, eps2=1e-16,
            state_storage_device=DEV, state_storage_dtype="float32")
# LoKr w1 / w2 / dora_scale-like shapes; (9, 9) and (7,) never stack (one of a kind / unfactored)
SHAPES = ([(4, 4)] * 5 + [(16, 32)] * 3 + [(32, 1)] * 4 + [(1, 24)] * 3 + [(3, 4, 5)] * 2 + [(2, 3, 4, 5)] * 2
          + [(7,)] * 2 + [(9, 9)])
ULPS = 16.0  # "same math": measured <= 4 ulp of a tensor's largest element; a wrong reduction is off by 1e5 ulps or more


def _load_backup():
    name = "LoraEasyCustomOptimizer.came_before_shapebatch_review"
    loader = importlib.machinery.SourceFileLoader(name, _BACKUP)
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    mod.__package__ = "LoraEasyCustomOptimizer"
    loader.exec_module(mod)
    return mod


def _make(kw, shapes=SHAPES, groups=None, cls=None):
    gen = torch.Generator(device=DEV).manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(s, generator=gen, device=DEV) * 0.02) for s in shapes]
    spec = [dict(params=params, lr=torch.tensor(1e-3))] if groups is None else groups(params)  # LyCORIS: tensor lr
    return params, (cls or CAME)(spec, lr=1e-3, **kw)


def _grads(params, k, scale=None, skip=()):
    gen = torch.Generator(device=DEV).manual_seed(1000 + k)
    for i, p in enumerate(params):
        g = torch.randn(p.shape, generator=gen, device=DEV) * 1e-3
        if scale is not None:
            g = g * scale[i % len(scale)]
        p.grad = None if i in skip else g


def _ulps(a, b):
    """max |a-b| in ulps of the largest finite |element| of a; NaN in the same places counts as equal."""
    a, b = a.detach().float(), b.detach().float()
    if a.numel() == 0:
        return 0.0
    if not torch.equal(torch.isnan(a), torch.isnan(b)):
        return float("inf")
    d = torch.nan_to_num((a - b).abs(), nan=0.0, posinf=float("inf"))
    m = torch.where(torch.isfinite(a), a.abs(), torch.zeros_like(a)).max()
    return float(d.max() / (torch.nextafter(m, torch.tensor(float("inf"), device=m.device)) - m))


def _worst(pa, oa, pb, ob):
    """Largest difference (ulps) over parameters and every state tensor; the state layouts must be the same."""
    worst = max(_ulps(a, b) for a, b in zip(pa, pb))
    for a, b in zip(pa, pb):
        sa, sb = oa.state[a], ob.state[b]
        assert sa.keys() == sb.keys()
        for key, v in sa.items():
            if torch.is_tensor(v):
                assert (v.shape, v.dtype, v.device) == (sb[key].shape, sb[key].dtype, sb[key].device), key
                worst = max(worst, _ulps(v, sb[key]))
    return worst


def _roundtrip(sd):
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


def _dynamo_entries():
    """Cache entries of the code object every torch.compile'd @torch.no_grad() function is traced through
    (torch/_dynamo/external_utils.py wrap_inline.inner), or None if this torch does not expose it."""
    try:
        from torch._dynamo import external_utils
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        return len(_debug_get_cache_entry_list(external_utils.wrap_inline(lambda: None).__code__))
    except Exception:
        return None


@unittest.skipUnless(torch.cuda.is_available(), "the grouped path runs on CUDA parameters only")
class CameGroupShapesReviewTest(unittest.TestCase):
    def test_step_sends_each_parameter_through_exactly_one_path(self):
        # The other tests compare results; if step() stopped calling the grouped path they would still pass (0 ulp).
        for flag in (False, True):
            params, opt = _make(dict(BASE, group_shapes=flag))
            _grads(params, 0)
            grouped_rows, single = [], []
            real_grouped, real_single = CAME._core_factored_grouped_fp32, CAME._core_factored_full_fp32
            with mock.patch.object(CAME, "_core_factored_grouped_fp32", staticmethod(
                    lambda grad, *a, **k: (grouped_rows.append(grad.shape[0]), real_grouped(grad, *a, **k))[1])), \
                 mock.patch.object(CAME, "_core_factored_full_fp32", staticmethod(
                    lambda grad, *a, **k: (single.append(tuple(grad.shape)), real_single(grad, *a, **k))[1])):
                before = [p.detach().clone() for p in params]
                opt.step()
            if flag:
                self.assertEqual(sorted(grouped_rows), [2, 2, 3, 3, 4, 5])  # one call per same-shape bucket
                self.assertEqual(single, [(9, 9)])                           # the one-of-a-kind matrix
            else:
                self.assertEqual(grouped_rows, [])
                self.assertEqual(len(single), sum(len(s) >= 2 for s in SHAPES))
            self.assertTrue(all(bool((p != b).any()) for p, b in zip(params, before)))  # nobody skipped by both paths

    def test_reductions_stay_inside_one_parameter(self):
        # Same bucket, gradient scales 1e6 .. 1e-20 and an all-zero gradient: an RMS, mask mean or row/col mean taken
        # over the stack instead of one parameter changes the small-gradient parameters by many orders of magnitude.
        shapes = [(64, 48)] * 8 + [(4, 4)] * 8 + [(128, 1)] * 8
        scale = [1e6, 1e-12, 0.0, 1.0, 1e3, 1e-6, 30.0, 1e-20]
        pa, oa = _make(BASE, shapes)
        pb, ob = _make(dict(BASE, group_shapes=True), shapes)
        for k in range(8):
            for ps, o in ((pa, oa), (pb, ob)):
                _grads(ps, k, scale=scale)
                o.step()
        self.assertLessEqual(_worst(pa, oa, pb, ob), ULPS)

    def test_non_finite_gradient_stays_in_its_parameter(self):
        shapes = [(64, 48)] * 6 + [(4, 4)] * 6
        for bad in (float("nan"), float("inf")):
            pa, oa = _make(BASE, shapes)
            pb, ob = _make(dict(BASE, group_shapes=True), shapes)
            for k in range(3):
                for ps, o in ((pa, oa), (pb, ob)):
                    _grads(ps, k)
                    ps[2].grad.view(-1)[5] = bad
                    ps[7].grad.view(-1)[0] = bad
                    o.step()
            broken = [i for i, p in enumerate(pb) if not bool(torch.isfinite(p).all())]
            self.assertEqual(broken, [i for i, p in enumerate(pa) if not bool(torch.isfinite(p).all())], bad)
            self.assertEqual(broken, [2, 7], bad)
            self.assertLessEqual(max(_ulps(pa[i], pb[i]) for i in range(len(pa)) if i not in broken), ULPS)

    def test_every_option_combination_with_an_lr_tensor_changed_in_place(self):
        # lr goes 0 -> 2e-3 -> 4e-3 -> 0 .. through the SAME tensor object (what LambdaLR does to LyCORIS' tensor lr): a
        # cached lr that is compared by value / identity instead of refreshed per step freezes at 0 (the first 11-14 hotfix).
        combos = itertools.product((False, True), ((True, False), (False, False), (True, True)),
                                   ("unmodified", "cautious", "grams", "both"), (False, True))
        for ams, (decouple, cautious_wd), strategy, fixed in combos:
            kw = dict(BASE, ams_bound=ams, weight_decouple=decouple, cautious_weight_decay=cautious_wd,
                      update_strategy=strategy, fixed_decay=fixed)
            pa, oa = _make(kw)
            pb, ob = _make(dict(kw, group_shapes=True))
            start = [p.detach().clone() for p in pb]
            for k in range(6):
                for ps, o in ((pa, oa), (pb, ob)):
                    o.param_groups[0]["lr"].fill_(2e-3 * (k % 3))
                    _grads(ps, k)
                    o.step()
            label = (ams, decouple, cautious_wd, strategy, fixed)
            self.assertLessEqual(_worst(pa, oa, pb, ob), ULPS, label)
            self.assertTrue(all(bool((p != s).any()) for p, s in zip(pb, start)), label)
            if not decouple and not cautious_wd:  # coupled L2: the per-parameter path leaves grad + wd * p in p.grad
                self.assertLessEqual(max(_ulps(a.grad, b.grad) for a, b in zip(pa, pb)), ULPS, label)

    def test_param_groups_keep_their_own_lr_decay_and_betas(self):
        # --dora_learning_rate (library/dora_lr.py) and LoRA+ put same-shape tensors into different groups.
        def groups(ps):
            third = len(ps) // 3
            return [dict(params=ps[:third], lr=torch.tensor(1e-3)),
                    dict(params=ps[third:2 * third], lr=torch.tensor(3e-4), weight_decay=0.0, betas=(0.8, 0.99, 0.999)),
                    dict(params=ps[2 * third:], lr=5e-4, weight_decay=0.3, update_strategy="grams")]

        shapes = ([(4, 4)] * 4 + [(16, 32)] * 3 + [(32, 1)] * 3) * 3
        pa, oa = _make(BASE, shapes, groups)
        pb, ob = _make(dict(BASE, group_shapes=True), shapes, groups)
        for k in range(8):
            for ps, o in ((pa, oa), (pb, ob)):
                o.param_groups[0]["lr"].mul_(0.9)
                o.param_groups[1]["lr"].add_(1e-4)
                o.param_groups[2]["lr"] = 5e-4 * (k + 1)
                _grads(ps, k)
                o.step()
        self.assertLessEqual(_worst(pa, oa, pb, ob), ULPS)
        # and the reference really used three different settings: one shared setting gives other numbers
        pc, oc = _make(BASE, shapes)
        for k in range(8):
            _grads(pc, k)
            oc.step()
        self.assertGreater(_ulps(pa[-1], pc[-1]), 1e3)

    def test_changing_gradient_sets_and_a_late_param_group(self):
        # eager: a parameter is stepped exactly in the steps where it has a gradient, whatever bucket it lands in
        shapes = [(4, 4)] * 9 + [(16, 32)] * 5 + [(32, 1)] * 4
        pa, oa = _make(BASE, shapes)
        pb, ob = _make(dict(BASE, group_shapes=True), shapes)
        late_a = [torch.nn.Parameter(torch.full((4, 4), 0.01 * (i + 1), device=DEV)) for i in range(3)]
        late_b = [torch.nn.Parameter(torch.full((4, 4), 0.01 * (i + 1), device=DEV)) for i in range(3)]
        for k in range(12):
            skip = {(k * 3 + j) % len(shapes) for j in range(k % 5)}
            for ps, o, late in ((pa, oa, late_a), (pb, ob, late_b)):
                if k == 4:
                    o.add_param_group(dict(params=late, lr=torch.tensor(2e-3)))  # state is created inside the grouped path
                _grads(ps + (late if k >= 4 else []), k, skip=skip)
                o.step()
        self.assertLessEqual(_worst(pa + late_a, oa, pb + late_b, ob), ULPS)

    def test_resume_in_both_directions(self):
        shapes = [(4, 4)] * 9 + [(16, 32)] * 5 + [(32, 1)] * 4
        for saved_with, resumed_with in ((True, False), (False, True), (True, True)):
            ref_p, ref_o = _make(dict(BASE, group_shapes=resumed_with), shapes)
            src_p, src_o = _make(dict(BASE, group_shapes=saved_with), shapes)
            for k in range(4):
                for ps, o in ((ref_p, ref_o), (src_p, src_o)):
                    _grads(ps, k)
                    o.step()
            res_p, res_o = _make(dict(BASE, group_shapes=resumed_with), shapes)
            with torch.no_grad():
                for p, v in zip(res_p, src_p):
                    p.copy_(v)
            res_o.load_state_dict(_roundtrip(src_o.state_dict()))
            for k in range(4, 8):
                for ps, o in ((ref_p, ref_o), (res_p, res_o)):
                    _grads(ps, k)
                    o.step()
            self.assertLessEqual(_worst(ref_p, ref_o, res_p, res_o), 0.0 if saved_with == resumed_with else ULPS,
                                 (saved_with, resumed_with))

    def test_flag_off_matches_the_backup_under_compile_step(self):
        # test_came_group_shapes.py checks flag off == backup eagerly; the run uses compile_step=True
        old = _load_backup()
        kw = dict(BASE, compile_step=True)
        shapes = [(4, 4)] * 3 + [(16, 32)] * 2 + [(32, 1)] * 2
        try:
            pa, oa = _make(kw, shapes, cls=old.CAME)
            pb, ob = _make(kw, shapes)
            for k in range(5):
                for ps, o in ((pa, oa), (pb, ob)):
                    o.param_groups[0]["lr"].fill_(1e-3 * (k % 3))
                    _grads(ps, k)
                    o.step()
            self.assertEqual(_worst(pa, oa, pb, ob), 0.0)
            self.assertIsNone(ob._compiled_grouped)  # never built with the flag off
        finally:
            torch._dynamo.reset()  # leave the shared compile cache as this test found it

    def test_compile_step_builds_no_graph_after_the_first_step(self):
        # constant buckets (the Anima LoKr run): the lr tensor changing in place must not trigger a recompile
        if _dynamo_entries() is None:
            self.skipTest("this torch does not expose the dynamo cache entries")
        kw = dict(BASE, compile_step=True)
        shapes = [(4, 4)] * 6 + [(16, 32)] * 4 + [(32, 1)] * 5 + [(9, 9)]
        try:
            torch._dynamo.reset()
            pa, oa = _make(kw, shapes)
            pb, ob = _make(dict(kw, group_shapes=True), shapes)
            counts = []
            for k in range(6):
                for ps, o in ((pa, oa), (pb, ob)):
                    o.param_groups[0]["lr"].fill_(1e-3 * (k % 3))
                    _grads(ps, k)
                    o.step()
                counts.append(_dynamo_entries())
            self.assertEqual(counts, [counts[0]] * 6)
            self.assertEqual(counts[0], 4 + 3)  # per-parameter: 4 shapes; grouped: 3 buckets ((9, 9) reuses its graph)
            self.assertLessEqual(_worst(pa, oa, pb, ob), ULPS)
            self.assertIsNot(ob._compiled_grouped, CAME._core_factored_grouped_fp32)
        finally:
            torch._dynamo.reset()

    def test_global_step_hooks_and_memory(self):
        # the sibling 11-14 item (lycoris_fast.ckpt_merged_invalidate) hangs on torch's global optimizer-step hook
        from torch.optim.optimizer import register_optimizer_step_post_hook

        calls = []
        handle = register_optimizer_step_post_hook(lambda *a: calls.append(1))

        def steps(k0, k1):
            for k in range(k0, k1):
                _grads(params, k, skip={k % len(params)})
                opt.step()
            for p in params:
                p.grad = None  # like zero_grad(set_to_none=True): only parameters and optimizer state stay allocated
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated()

        try:
            params, opt = _make(dict(BASE, group_shapes=True))
            held = steps(0, 20)
            after = steps(20, 120)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 120)
        self.assertEqual(after, held)  # nothing kept from step to step

    def test_real_lokr_dora_adapters_train(self):
        # LyCORIS LoKr (factor 4, full matrix) + DoRA on a toy model, fp16 autocast + GradScaler, accumulation 4, clip 1.0,
        # tensor lr warmed up from 0 in place: the loss falls, every adapter tensor moves (also the zero-initialised
        # factor), and the run stays next to the flag-off run. Measured 11-14: loss gap 1.8e-4 over 40 steps, the same
        # size as flag off eager vs flag off compile_step (1.2e-4): the ulp differences are amplified by the fp16
        # forward like any other rounding difference.
        try:
            from lycoris.wrapper import LycorisNetwork, create_lycoris
        except Exception as e:
            self.skipTest(f"lycoris is not importable: {e}")
        import logging

        class Block(torch.nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.q, self.k, self.v, self.o = (torch.nn.Linear(d, d) for _ in range(4))
                self.fc1, self.fc2 = torch.nn.Linear(d, 4 * d), torch.nn.Linear(4 * d, d)

            def forward(self, x):
                a = torch.softmax(self.q(x) @ self.k(x).transpose(-1, -2) / 8.0, dim=-1) @ self.v(x)
                x = x + self.o(a)
                return x + self.fc2(torch.nn.functional.gelu(self.fc1(x)))

        def train(flag):
            torch.manual_seed(0)
            model = torch.nn.Sequential(*(Block() for _ in range(4))).to(DEV).requires_grad_(False)
            LycorisNetwork.apply_preset({"target_module": ["Block"]})
            net = create_lycoris(model, 1.0, linear_dim=100000, linear_alpha=1, algo="lokr", factor=4, full_matrix=True,
                                 dora_wd=True, wd_on_output=True)
            net.apply_to()
            net.to(DEV)
            params = [p for p in net.parameters() if p.requires_grad]
            start = [p.detach().clone() for p in params]
            opt = CAME([dict(params=params, lr=torch.tensor(2e-4))], lr=2e-4, group_shapes=flag, **BASE)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 5))
            scaler = torch.amp.GradScaler("cuda")
            gen = torch.Generator(device=DEV).manual_seed(123)
            teacher = torch.randn(64, 64, generator=gen, device=DEV) * 0.2
            losses = []
            for _ in range(30):
                total = 0.0
                for _ in range(4):
                    x = torch.randn(4, 16, 64, generator=gen, device=DEV)
                    with torch.autocast("cuda", dtype=torch.float16):
                        y = model(x)
                    loss = torch.nn.functional.mse_loss(y.float(), x @ teacher) / 4
                    scaler.scale(loss).backward()
                    total += float(loss.detach())
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                losses.append(total)
            return params, start, losses

        logging.disable(logging.WARNING)
        saved_targets = LycorisNetwork.TARGET_REPLACE_MODULE  # apply_preset writes class attributes
        try:
            _, _, off = train(False)
            params, start, on = train(True)
        finally:
            LycorisNetwork.TARGET_REPLACE_MODULE = saved_targets
            logging.disable(logging.NOTSET)
        self.assertGreaterEqual(sum(float(s.abs().max()) == 0.0 for s in start), 1)  # LoKr has a zero-initialised factor
        self.assertTrue(all(bool((p != s).any()) for p, s in zip(params, start)))
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in params))
        self.assertLess(on[-1], 0.9 * on[0])
        self.assertLess(max(abs(a - b) / abs(a) for a, b in zip(off, on)), 1e-2)


class CameGroupShapesOpenIssuesTest(unittest.TestCase):
    """Found by the 2026-11-14 review; each test states the behaviour the code should have (see the module docstring)."""

    @unittest.skipUnless(torch.cuda.is_available(), "compiled grouped path: CUDA only")
    def test_a_new_bucket_size_must_not_stop_training_when_the_compile_cache_is_full(self):
        # compile_step + group_shapes: every (number of stacked parameters, shape) is its own graph, and ALL
        # torch.compile'd @torch.no_grad() functions of the process share one dynamo cache (torch 2.14: the frame
        # compiled is torch/_dynamo/external_utils.py inner - _grouped_core_fn's docstring says the opposite). With
        # fullgraph=True the entry after recompile_limit is not an eager fallback but FailOnRecompileLimitHit. A run
        # whose set of parameters WITH a gradient changes (module_dropout, gated experts, a module skipped in a
        # forward) therefore recompiles on every step (~3 s each) and then dies: measured with CAME's own limit of 64,
        # 80 (4, 4) parameters and a different count each step: flag off 75 steps in 1 s with 1 graph; flag on
        # FailOnRecompileLimitHit at step 63 after 205 s (came_group_shapes_review_recompile.py in the speedtest folder).
        # Here the full cache is simulated by lowering the limit after the first step.
        # Fix: keep a budget of grouped graphs and run a chunk with a new size through the eager grouped core beyond it,
        # and catch (TorchDynamoException, FailOnRecompileLimitHit) around the compiled call - a refused compile has
        # run nothing, the stacked inputs are intact - like library/elementwise_compile.py does.
        import torch._dynamo as dynamo

        shapes = [(4, 4)] * 12
        saved = dynamo.config.recompile_limit
        try:
            pa, oa = _make(dict(BASE, compile_step=True), shapes)
            pb, ob = _make(dict(BASE, compile_step=True, group_shapes=True), shapes)
            for k in range(4):
                skip = set(range(k))  # 12, 11, 10, 9 parameters with a gradient
                for ps, o in ((pa, oa), (pb, ob)):
                    _grads(ps, k, skip=skip)
                    o.step()  # flag on, k = 1: FailOnRecompileLimitHit
                dynamo.config.recompile_limit = 1  # "the cache is full" from the second step on
            self.assertLessEqual(_worst(pa, oa, pb, ob), ULPS)
        finally:
            dynamo.config.recompile_limit = saved
            dynamo.reset()

    @unittest.skipUnless(torch.cuda.is_available(), "the grouped path runs on CUDA parameters only")
    def test_the_grouped_write_back_bumps_the_parameter_version_like_the_per_parameter_path(self):
        # Flag off (fp32 state on the parameter's device) the core updates the Parameter itself in place: p._version
        # goes up every step. The grouped path writes through p.data, which has its own version counter, so p._version
        # never changes. Anything that keys a cache on it stops seeing optimizer steps - the sibling 11-14 item does
        # (sd_scripts_speed1114 library/lycoris_fast.py _ckpt_key) and stays correct only through its optimizer-step
        # hook. Fix: torch._foreach_copy_(list(chunk), ...) instead of [p.data for p in chunk] (allowed under no_grad).
        versions = {}
        for flag in (False, True):
            params, opt = _make(dict(BASE, group_shapes=flag), [(4, 4)] * 3)
            _grads(params, 0)
            opt.step()
            versions[flag] = [p._version for p in params]
        self.assertTrue(all(v > 0 for v in versions[False]))
        self.assertTrue(all(v > 0 for v in versions[True]), versions)

    def test_a_lower_case_false_does_not_switch_the_flag_on(self):
        # sd-scripts (library/train_util.py) keeps an optimizer_args value as a string when ast.literal_eval raises
        # ValueError: "group_shapes=false" / "off" / "no" arrive as 'false' / 'off' / 'no', and bool() of a non-empty
        # string is True - the opt-in path would be ON for someone who wrote it to be off. (compile_step and the other
        # switches of this class behave the same; this one is new and must default to the old path.)
        # Fix: map the strings true/false/1/0/yes/no/on/off, refuse anything else.
        p = torch.nn.Parameter(torch.zeros(2, 2))
        for spelling in ("false", "False", "off", "no", "0"):
            self.assertFalse(CAME([p], group_shapes=spelling).group_shapes, spelling)
        self.assertTrue(CAME([p], group_shapes="true").group_shapes)

    def test_a_chunk_never_holds_more_elements_than_group_shapes_max_numel(self):
        # "Upper bound on elements per stacked chunk": ceil(len / n_chunks) rounds the chunk size up, so parameters a
        # little over cap / 2 are stacked in pairs of almost 2 x cap (default cap: 10 tensors of 30M elements -> chunks
        # of 2 = 1.79 x cap; found worst 1.94 x). The Anima LoKr run stays under it (largest chunk 0.94 x cap).
        # Fix: size = min(ceil(len / n_chunks), cap // numel), with n_chunks = ceil(len / (cap // numel)).
        class Sized:
            def __init__(self, n):
                self.n = n

            def numel(self):
                return self.n

        opt = CAME([torch.nn.Parameter(torch.zeros(2, 2))], group_shapes=True)
        cap = opt.group_shapes_max_numel
        for numel, count in ((30_000_000, 10), (17_000_000, 3), (cap // 2 + 1, 7), (512 * 512, 240)):
            for chunk in opt._grouped_chunks([Sized(numel)] * count):
                self.assertLessEqual(len(chunk) * numel, cap, (numel, count, len(chunk)))


if __name__ == "__main__":
    unittest.main()
