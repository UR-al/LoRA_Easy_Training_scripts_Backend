"""CAME group_shapes: second review (2026-11-14), of the four fixes that followed test_came_group_shapes_review.py
(graph budget + eager fallback, write-back through the Parameter, flag spellings, chunk cap).

Everything here runs WITHOUT a GPU (and does not touch one that is there). The grouped path only takes CUDA parameters,
so these tests use CPU stand-ins:
  * parameters that answer is_cuda=True (the only CUDA question _grouped_eligible asks),
  * opt._use_staging = False - what state_storage_device="cuda" + float32 gives on a GPU,
  * torch.cuda.current_device / torch.cuda.synchronize / Tensor.pin_memory replaced, as in test_came_foreach.py, and
    torch.cuda.is_available -> False so that dynamo does not open a CUDA context,
  * torch.compile(backend="eager"): dynamo traces, guards, caches and enforces recompile_limit exactly as usual; only
    inductor's code generation (it needs a C compiler / Triton) is left out.
What the stand-ins cannot show: the rounding of CUDA reduction kernels and of inductor's fused kernels. That is what the
GPU suites test_came_group_shapes.py / test_came_group_shapes_review.py are for.

CameGroupShapesFixGuardsTest       - the fixes do what they say. Mostly pieces no other test noticed when they were
                                     removed again (checked by mutation: graph budget, TorchDynamoException in the
                                     fallback, the fallback being for good, unknown flag strings, switches in the graph
                                     key), plus CPU twins of GPU-only checks (flag off == backups, refused recompile).
CameGroupShapesReview2OpenIssuesTest - what this review found, each written as the behaviour the code SHOULD have.
                                     They were expectedFailure until the review-2 fixes of 2026-11-14 (came.py:
                                     isolate_recompiles for the grouped core, `del` of the stacked buffers, even chunk
                                     split, duplicated parameters left to the per-parameter loop); now plain tests.
                                     open_issue() is kept for marking the next finding.
                                     Three came from the review fixes (shared compile cache, buffer lifetime, a
                                     left-over parameter); the duplicated-parameter one is older.
                                     None of them changes the numbers of the Anima LoKr run as configured on 2026-11-14
                                     (11 shapes, constant gradient set, recompile_limit 128 from the block compile).
"""

import gc
import importlib.machinery
import importlib.util
import math
import os
import random
import sys
import types
import unittest
import weakref
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
# the optimizer_args of config_anima29.toml (2026-11-14), with the state on "the parameter's device"
BASE = dict(betas=(0.9, 0.999, 0.9999), weight_decay=0.1, weight_decouple=True, fixed_decay=False,
            update_strategy="cautious", clip_threshold=1.0, eps1=1e-30, eps2=1e-16,
            state_storage_device="cpu", state_storage_dtype="float32")
ULPS = 16.0  # "same math" (measured <= 4); a wrong or missing update is off by 1e5 ulps or more
BACKUPS = ("came.py.bak_20261114_before_reviewfix", "came.py.bak_20261114_before_shapebatch",
           "came.py.bak_20261114_before_hotfix")


def open_issue(test):
    """expectedFailure, unless CAME_REVIEW_STRICT=1 (then the test fails in the open)."""
    return test if os.environ.get("CAME_REVIEW_STRICT") == "1" else unittest.expectedFailure(test)


class _CudaLike(torch.nn.Parameter):
    """A CPU Parameter that answers is_cuda=True."""

    @property
    def is_cuda(self):
        return True


class _Sized:
    """Stands in for a parameter in _grouped_chunks (it only asks numel())."""

    def __init__(self, n, i=0):
        self.n, self.i = n, i

    def numel(self):
        return self.n


def _load(fname):
    name = "LoraEasyCustomOptimizer.came_review2_" + fname.replace(".", "_")
    loader = importlib.machinery.SourceFileLoader(name, os.path.join(_PKG_DIR, fname))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    mod.__package__ = "LoraEasyCustomOptimizer"
    loader.exec_module(mod)
    return mod


def _make(kw, shapes, groups=None, cls=None, dtype=torch.float32, cuda_fp32_state=True):
    gen = torch.Generator().manual_seed(0)
    params = [_CudaLike((torch.randn(s, generator=gen) * 0.02).to(dtype)) for s in shapes]
    spec = [dict(params=params, lr=torch.tensor(1e-3))] if groups is None else groups(params)  # LyCORIS: tensor lr
    opt = (cls or CAME)(spec, lr=1e-3, **dict(BASE, **kw))
    if cuda_fp32_state:
        opt._use_staging = False  # str(state_storage_device).startswith("cuda") and float32
    return params, opt


def _grads(params, k, skip=()):
    gen = torch.Generator().manual_seed(1000 + k)
    for i, p in enumerate(params):
        g = (torch.randn(p.shape, generator=gen) * 1e-3).to(p.dtype)
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
    return float(d.max() / (torch.nextafter(m, torch.tensor(float("inf"))) - m))


def _worst(pa, oa, pb, ob):
    """Largest difference (ulps) over parameters and every state tensor; the state layouts must be the same."""
    worst = max(_ulps(a, b) for a, b in zip(pa, pb))
    for a, b in zip(pa, pb):
        sa, sb = oa.state[a], ob.state[b]
        assert sa.keys() == sb.keys()
        for key, v in sa.items():
            if torch.is_tensor(v):
                assert (v.shape, v.dtype) == (sb[key].shape, sb[key].dtype), key
                worst = max(worst, _ulps(v, sb[key]))
    return worst


def _dynamo_entries():
    """Cache entries of the code object every torch.compile'd @torch.no_grad() function is traced through."""
    from torch._dynamo import external_utils
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    return len(_debug_get_cache_entry_list(external_utils.wrap_inline(lambda: None).__code__))


class _CpuStandIns(unittest.TestCase):
    backend = "eager"

    def setUp(self):
        import torch._dynamo as dynamo

        real_compile = torch.compile

        def compile_(fn=None, **kw):
            kw["backend"] = self.backend
            return real_compile(fn, **kw)

        for target, name, stand_in in (
                # also on a PC with a GPU: dynamo then leaves CUDA alone (no context, no VRAM), tensors are CPU anyway
                (torch.cuda, "is_available", lambda: False),
                (torch.cuda, "current_device", lambda: torch.device("cpu")),
                (torch.cuda, "synchronize", lambda *a, **k: None),
                (torch.Tensor, "pin_memory", lambda self_, *a, **k: self_),
                (torch, "compile", compile_)):
            patcher = mock.patch.object(target, name, stand_in)
            patcher.start()
            self.addCleanup(patcher.stop)
        saved = dynamo.config.recompile_limit

        def restore():
            dynamo.config.recompile_limit = saved
            dynamo.reset()

        self.addCleanup(restore)
        dynamo.reset()
        dynamo.config.recompile_limit = 8  # torch's default: a fresh process. CAME raises it to its own 64.


class CameGroupShapesFixGuardsTest(_CpuStandIns):
    def test_the_stand_ins_send_cpu_parameters_through_the_grouped_path(self):
        # without this every other test here could pass through the per-parameter path alone
        shapes = [(4, 4)] * 5 + [(16, 32)] * 3 + [(32, 1)] * 4 + [(7,)] * 2 + [(9, 9)]
        pa, oa = _make(dict(), shapes)
        pb, ob = _make(dict(group_shapes=True), shapes)
        rows = []
        real = CAME._core_factored_grouped_fp32
        with mock.patch.object(CAME, "_core_factored_grouped_fp32", staticmethod(
                lambda grad, *a, **k: (rows.append(grad.shape[0]), real(grad, *a, **k))[1])):
            for k in range(3):
                for ps, o in ((pa, oa), (pb, ob)):
                    _grads(ps, k)
                    o.step()
        self.assertEqual(sorted(rows), [3, 3, 3, 4, 4, 4, 5, 5, 5])
        self.assertLessEqual(_worst(pa, oa, pb, ob), ULPS)
        self.assertTrue(all(p._version > 0 for p in pb))  # review fix 2, on the eager route

    def test_the_graph_budget_leaves_the_per_parameter_functions_room_to_compile(self):
        # Review fix 1 has two halves; test_came_group_shapes_review.py only exercises the except clause. Without the
        # BUDGET (_GROUPED_MAX_GRAPHS) every test passes as well - the grouped path then compiles until the shared cache
        # is full (CAME's own recompile_limit, 64), falls back, and the next NEW shape on the per-parameter path dies
        # with FailOnRecompileLimitHit. Here: the first review's scenario (80 (4, 4) matrices, a different number with
        # a gradient each step = 70 different stack sizes), then one (5, 5) matrix gets its first gradient.
        shapes = [(4, 4)] * 80 + [(5, 5)]
        pa, oa = _make(dict(), shapes)  # eager per-parameter reference
        pb, ob = _make(dict(compile_step=True, group_shapes=True), shapes)
        rnd = random.Random(0)
        for k in range(72):
            active = set(rnd.sample(range(80), 80 - (k % 70) if k else 80))
            skip = (set(range(80)) - active) | ({80} if k < 71 else set())
            for ps, o in ((pa, oa), (pb, ob)):
                _grads(ps, k, skip)
                o.step()
        self.assertEqual(len(ob._grouped_graph_keys), CAME._GROUPED_MAX_GRAPHS)
        self.assertFalse(ob._grouped_compile_off)  # the budget, not a refused compile, kept the cache from filling
        self.assertEqual(_dynamo_entries(), CAME._GROUPED_MAX_GRAPHS + 1)  # + the (5, 5) per-parameter graph
        self.assertLessEqual(_worst(pa, oa, pb, ob), ULPS)  # compiled and eager chunks mixed: still the same math

    def test_a_backend_that_cannot_compile_switches_to_the_eager_core_for_good(self):
        # "the eager core for good if the compiler refuses": a backend failure arrives as BackendCompilerFailed, a
        # TorchDynamoException (on this PC without CUDA: InductorError "cl is not found"). No other test raises one.
        attempts = []

        def broken_backend(gm, example_inputs):
            attempts.append(1)
            raise RuntimeError("no compiler on this machine")

        self.backend = broken_backend
        shapes = [(4, 4)] * 6 + [(16, 32)] * 4 + [(32, 1)] * 5  # no one-of-a-kind / 1-D: only the grouped core is called
        kw = dict(group_shapes=True, weight_decouple=False)      # coupled L2: the step also writes p.grad
        pa, oa = _make(kw, shapes)                               # the eager grouped core
        pb, ob = _make(dict(kw, compile_step=True), shapes)
        with self.assertLogs(came_mod.logger, "WARNING") as logs:
            for k in range(4):
                for ps, o in ((pa, oa), (pb, ob)):
                    _grads(ps, k)
                    o.step()
        self.assertTrue(ob._grouped_compile_off)
        self.assertEqual(len(attempts), 1)  # not once per chunk and step
        self.assertEqual(sum("runs eagerly from here on" in line for line in logs.output), 1)
        self.assertEqual(_worst(pa, oa, pb, ob), 0.0)  # the refused call ran nothing: exactly the eager numbers
        self.assertTrue(all(torch.equal(a.grad, b.grad) for a, b in zip(pa, pb)))
        self.assertTrue(all(p._version > 0 for p in pb))  # review fix 2, on the fallback route

    def test_a_refused_recompile_falls_back_and_the_graphs_already_built_stay_in_use(self):
        # CPU twin of test_a_new_bucket_size_must_not_stop_training_when_the_compile_cache_is_full, plus: the refusal
        # must not switch the SHARED frame to run-only (the per-parameter graphs of another optimizer keep running
        # compiled), and it is tried once, not on every step.
        runs = []

        def counting_backend(gm, example_inputs):
            def run(*a, **k):
                runs.append(1)
                return gm.forward(*a, **k)

            return run

        import torch._dynamo as dynamo

        self.backend = counting_backend
        pa, oa = _make(dict(group_shapes=True), [(4, 4)] * 12)                     # the eager grouped core
        pb, ob = _make(dict(group_shapes=True, compile_step=True), [(4, 4)] * 12)
        pc, oc = _make(dict(compile_step=True), [(4, 4)] * 3 + [(5, 5)] * 2)       # flag off, compiled per parameter
        per_step = []
        for k in range(6):
            skip = set(range(k)) if k < 4 else set()  # 12, 11, 10, 9, 12, 12 parameters with a gradient
            for ps, o in ((pa, oa), (pb, ob)):
                _grads(ps, k, skip)
                o.step()  # k = 1: a new stack size, FailOnRecompileLimitHit inside
            before = len(runs)
            _grads(pc, k)
            oc.step()
            per_step.append(len(runs) - before)
            if k == 0:
                self.assertEqual(_dynamo_entries(), 3)
                # "the cache is full" from here on (1, not 3: also true for a core with a cache region of its own)
                dynamo.config.recompile_limit = 1
        self.assertTrue(ob._grouped_compile_off)
        self.assertEqual(_dynamo_entries(), 3)
        self.assertEqual(per_step, [5] * 6)  # the other optimizer's five parameters: compiled graphs, every step
        self.assertEqual(_worst(pa, oa, pb, ob), 0.0)

    def test_param_groups_with_different_switches_are_separate_graphs_in_the_budget(self):
        # the flags are compile-time constants: (G, shape) alone would count two graphs as one
        def groups(ps):
            half = len(ps) // 2
            return [dict(params=ps[:half], lr=torch.tensor(1e-3)),
                    dict(params=ps[half:], lr=torch.tensor(1e-3), update_strategy="grams")]

        params, opt = _make(dict(compile_step=True, group_shapes=True), [(4, 4)] * 8, groups)
        _grads(params, 0)
        opt.step()
        self.assertEqual(_dynamo_entries(), 2)
        self.assertEqual(len(opt._grouped_graph_keys), 2)

    def test_only_spellings_of_true_and_false_are_accepted(self):
        # review fix 3: sd-scripts hands over what ast.literal_eval refuses as a string
        p = torch.nn.Parameter(torch.zeros(2, 2))
        for bad in ("flase", "bogus", "2", "none", ""):
            with self.assertRaises(ValueError, msg=repr(bad)):
                CAME([p], group_shapes=bad)
        for on in ("true", "True", " TRUE ", "on", "yes", "1", True, 1):
            self.assertIs(CAME([p], group_shapes=on).group_shapes, True, repr(on))
        for off in ("false", "False", " off", "no", "0", False, 0, None):
            self.assertIs(CAME([p], group_shapes=off).group_shapes, False, repr(off))

    def test_chunks_stay_under_the_cap_for_random_buckets(self):
        # review fix 4 as a property: never over group_shapes_max_numel, no chunk of one, at most two chunk sizes
        # (= two graphs per shape), every parameter at most once and in order
        opt = CAME([torch.nn.Parameter(torch.zeros(2, 2))], group_shapes=True)
        rnd = random.Random(0)
        for _ in range(4000):
            cap = rnd.choice([1, 2, 7, 64, 1000, 1 << 22, 1 << 25])
            numel = rnd.choice([1, 16, cap // 3 + 1, cap // 2, cap // 2 + 1, cap, cap + 1, rnd.randint(1, 2 * cap)])
            count = rnd.randint(1, 700)
            opt.group_shapes_max_numel = cap
            chunks = opt._grouped_chunks([_Sized(max(1, numel), i) for i in range(count)])
            label = (cap, numel, count, [len(c) for c in chunks])
            self.assertTrue(all(2 <= len(c) and len(c) * max(1, numel) <= cap for c in chunks), label)
            self.assertLessEqual(len({len(c) for c in chunks}), 2, label)
            order = [x.i for c in chunks for x in c]
            self.assertEqual(order, sorted(set(order)), label)
            if count >= 2 and 2 * max(1, numel) <= cap:
                self.assertGreaterEqual(len(order), count - 1, label)  # what fits is stacked (but see the open issue)

    def test_flag_off_is_bit_identical_to_the_backups(self):
        # CPU twin of the GPU-only flag-off checks, against all three backups of 2026-11-14 that are still there
        shapes = [(4, 4)] * 5 + [(16, 32)] * 3 + [(32, 1)] * 4 + [(3, 4, 5)] * 2 + [(7,)] * 2 + [(40,)] * 2 + [(9, 9), ()]
        cases = (
            (dict(), True, torch.float32),
            (dict(compile_step=True), True, torch.float32),
            # The backups encode BUG-05/07/08. Only unaffected options belong
            # in this identity test; independent CPU oracles cover those fixes
            # in test_came_verified_bugs.py.
            (dict(foreach=True), True, torch.float32),
            (dict(update_strategy="both"), True, torch.float32),
            (dict(state_storage_dtype="bfloat16"), False, torch.float32),   # staging + stochastic rounding
            (dict(), True, torch.bfloat16),
        )

        def run(cls, kw, cuda_fp32_state, dtype):
            torch._dynamo.reset()
            params, opt = _make(kw, shapes, cls=cls, dtype=dtype, cuda_fp32_state=cuda_fp32_state)
            for k in range(4):
                opt.param_groups[0]["lr"].fill_(1e-3 * (k % 3))
                _grads(params, k)
                torch.manual_seed(k)  # stochastic rounding of bf16 write-backs
                opt.step()
            return params, opt

        found = [name for name in BACKUPS if os.path.exists(os.path.join(_PKG_DIR, name))]
        if not found:
            self.skipTest("the came.py.bak_20261114_* backups are gone")
        for name in found:
            old = _load(name)
            for kw, cuda_fp32_state, dtype in cases:
                pa, oa = run(old.CAME, kw, cuda_fp32_state, dtype)
                pb, ob = run(CAME, kw, cuda_fp32_state, dtype)  # group_shapes defaults to False
                self.assertEqual(_worst(pa, oa, pb, ob), 0.0, (name, kw))
                self.assertIsNone(ob._compiled_grouped, (name, kw))  # never built with the flag off
                self.assertEqual(oa.state_dict()["param_groups"][0].keys(), ob.state_dict()["param_groups"][0].keys())


class CameGroupShapesReview2OpenIssuesTest(_CpuStandIns):
    """Found by the second review of 2026-11-14; each test states the behaviour the code should have."""

    def test_the_per_parameter_functions_must_not_stop_training_when_grouped_graphs_took_their_cache(self):
        # Review fix 1 keeps the GROUPED core alive when the shared dynamo cache is full, but the grouped graphs (up to
        # _GROUPED_MAX_GRAPHS = 32 of CAME's own recompile_limit of 64) still sit in the cache of the per-parameter
        # cores, and those calls have no fallback. A run with more than 32 per-parameter graphs left to build therefore
        # still stops with FailOnRecompileLimitHit - where flag off (one graph per shape) runs. Here: 40 LoRA shapes x 3
        # modules, module dropout 0.5 (a shape with ONE active module goes through the per-parameter path):
        # flag off 12 steps with 40 graphs; flag on FailOnRecompileLimitHit in step 4 with 32 + 32 = 64 entries.
        # (Not the Anima run: 11 shapes, no dropout, limit 128. The fixer listed this as the remaining risk.)
        # Fix: give the grouped core its own region - torch.compile(..., isolate_recompiles=True) exists in torch 2.14
        # and made this test pass in a scratch copy (72 entries, no exception) - or give the per-parameter calls the
        # same try / except as the grouped one.
        shapes = [(2 + i // 8, 2 + i % 8) for i in range(40) for _ in range(3)]
        for flag in (False, True):
            torch._dynamo.reset()
            pa, oa = _make(dict(), shapes)  # eager per-parameter reference
            pb, ob = _make(dict(compile_step=True, group_shapes=flag), shapes)
            rnd = random.Random(0)
            for k in range(12):
                skip = {i for i in range(len(shapes)) if k and rnd.random() < 0.5}
                for ps, o in ((pa, oa), (pb, ob)):
                    _grads(ps, k, skip)
                    o.step()  # flag on, k = 4: FailOnRecompileLimitHit from core_factored_fn
            self.assertLessEqual(_worst(pa, oa, pb, ob), ULPS, flag)

    def test_the_stacked_buffers_of_a_chunk_are_released_before_the_next_chunk_is_stacked(self):
        # Review fix 1 collects the arguments of the core in the local `core_args`. The tuple outlives the loop
        # iteration, so the previous chunk's stacked gradient, first moment and parameter copy (3 full-size buffers)
        # stay allocated until the NEXT chunk's three have been stacked as well: 6 buffers at once instead of the 4
        # before the fix (measured here on the 3rd step: 6.1 vs 4.1 full-size buffers; this test passes on
        # came.py.bak_20261114_before_reviewfix). With the default cap that is up to 2 x 128 MiB more transient VRAM
        # between chunks; for the Anima run's two (512, 512) x 120 chunks 720 MiB instead of 480 MiB - above the
        # +601 MiB step peak that was measured for compile_step BEFORE the fix (not re-measured on a GPU; the eager
        # peak of +842 MiB is inside the core and stays the larger one).
        # Fix: `del core_args` (or better: del core_args, grad, p_fp32, stacked) at the end of the loop body.
        shape, per_chunk = (64, 64), 4
        full = per_chunk * 64 * 64 * 4  # bytes of one stacked full-size buffer
        params, opt = _make(dict(group_shapes=True, group_shapes_max_numel=per_chunk * 64 * 64), [shape] * 12)
        for k in range(2):
            _grads(params, k)
            opt.step()
        live, peaks = [], []
        real_stack = torch.stack

        def stack(tensors, *a, **k):
            out = real_stack(tensors, *a, **k)
            live.append((weakref.ref(out), out.numel() * out.element_size()))
            peaks.append(sum(n for ref, n in live if ref() is not None))
            return out

        _grads(params, 2)
        gc.collect()
        gc.disable()
        try:
            with mock.patch.object(torch, "stack", stack):
                opt.step()
        finally:
            gc.enable()
        self.assertLessEqual(max(peaks) / full, 4.5, [round(p / full, 1) for p in peaks])

    def test_a_bucket_is_not_left_with_a_single_parameter_when_an_even_split_fits(self):
        # Review fix 4 cuts a bucket into slices of ceil(len / n_chunks): 10 parameters with room for 3 per chunk give
        # 3 + 3 + 3 and ONE parameter left over for the per-parameter path (one more compiled graph in the shared
        # cache, its own kernel launches every step), although 3 + 3 + 2 + 2 is under the cap, has two chunk sizes and
        # leaves nobody out. Default cap: 10 tensors of 10M elements. ("near-equal chunks" in the docstring.)
        opt = CAME([torch.nn.Parameter(torch.zeros(2, 2))], group_shapes=True)
        numel = 1000
        for count, per_chunk in ((10, 3), (7, 3), (13, 4), (21, 5)):
            opt.group_shapes_max_numel = per_chunk * numel
            sizes = [len(c) for c in opt._grouped_chunks([_Sized(numel)] * count)]
            self.assertGreaterEqual(count, 2 * math.ceil(count / per_chunk))  # an even split has no chunk of one
            self.assertEqual(sum(sizes), count, (count, per_chunk, sizes))
        opt.group_shapes_max_numel = 1 << 25
        self.assertEqual(sum(len(c) for c in opt._grouped_chunks([_Sized(10_000_000)] * 10)), 10)

    def test_a_parameter_listed_twice_in_a_group_is_stepped_as_often_as_on_the_per_parameter_path(self):
        # NOT one of the review fixes: there since the first group_shapes version, found with the CPU stand-ins.
        # torch only warns about a group that lists a parameter twice ("in future, this will cause an error"); the
        # per-parameter loop then steps it twice per step() (shared state). The grouped path stacks both occurrences
        # from the same starting values and writes the same result twice: ONE update. Flag on and flag off differ by a
        # whole update for that parameter (measured: millions of ulps), not by rounding. LyCORIS / sd-scripts do not
        # build such groups today.
        # Fix: a parameter that occurs more than once in group["params"] is not eligible for stacking.
        import warnings

        def groups(ps):
            return [dict(params=ps + [ps[0]], lr=torch.tensor(1e-3))]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "optimizer contains a parameter group with duplicate parameters"
            pa, oa = _make(dict(), [(4, 4)] * 4, groups)
            pb, ob = _make(dict(group_shapes=True), [(4, 4)] * 4, groups)
        for k in range(3):
            for ps, o in ((pa, oa), (pb, ob)):
                _grads(ps, k)
                o.step()
        self.assertLessEqual(_worst(pa, oa, pb, ob), ULPS)


if __name__ == "__main__":
    unittest.main()
