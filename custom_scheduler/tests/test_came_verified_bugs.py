"""CPU regression oracles for BUG-05..09 from the 2026-11-19 review.

Only hardware boundaries are replaced: CUDA transfers/pinning/sync and the
grouped CUDA eligibility gate. torch.compile uses its real fullgraph Dynamo
frontend with the CPU eager backend; this does not test CUDA/Inductor codegen.
"""

import copy
import hashlib
import json
import math
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

try:
    from LoraEasyCustomOptimizer.came import CAME
except ImportError:
    package = types.ModuleType("LoraEasyCustomOptimizer")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "LoraEasyCustomOptimizer")]
    sys.modules["LoraEasyCustomOptimizer"] = package
    from LoraEasyCustomOptimizer.came import CAME


class CPUGroupedCAME(CAME):
    def _grouped_eligible(self, p, group):
        if (p.grad is None or p.grad.is_sparse or p.dtype != torch.float32 or p.ndim < 2
                or self.state_storage_dtype != torch.float32):
            return False
        if not self.state[p]:
            self._init_param_state(p, group, p.shape, True)
        return True


@pytest.fixture(autouse=True)
def cpu_boundaries():
    real_compile = torch.compile
    def cpu_compile(fn, **kwargs):
        return real_compile(fn, backend="eager", **kwargs)
    with mock.patch.object(torch.cuda, "current_device", lambda: torch.device("cpu")), \
         mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None), \
         mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
         mock.patch.object(torch.Tensor, "pin_memory", lambda self, *a, **k: self), \
         mock.patch.object(torch, "compile", cpu_compile):
        yield
    assert not torch.cuda.is_initialized()


PATHS = ["eager", "compiled", "foreach", "grouped", "grouped_compiled"]


def make(path, shape=(16,), dtype=torch.float32, **kwargs):
    cls = CPUGroupedCAME if path.startswith("grouped") else CAME
    params = [torch.nn.Parameter(torch.ones(shape, dtype=dtype)) for _ in range(2)]
    options = dict(lr=0.01, state_storage_dtype="float32", state_storage_device="cpu")
    options.update(kwargs)
    opt = cls(params, foreach=path == "foreach", compile_step="compiled" in path,
              group_shapes=path.startswith("grouped"), **options)
    return params, opt


def step(params, opt, value):
    for p in params:
        p.grad = torch.full_like(p, value)
    opt.step()
    assert_execution_path(opt)


def assert_execution_path(opt):
    if opt.defaults["compile_step"]:
        assert opt._compiled_factored is not opt._core_factored_full_fp32
        assert opt._compiled_unfactored is not opt._core_unfactored_full_fp32
        if opt._compiled_grouped is not None:
            assert opt._compiled_grouped is not opt._core_factored_grouped_fp32
            assert opt._grouped_graph_keys
            assert not opt._grouped_compile_off


def baseline_trace(path, config):
    # Fixed arithmetic inputs, no random normal generation/version-dependent RNG.
    params, opt = make(path, shape=(3, 4), **config)
    params += [torch.nn.Parameter(torch.ones(16)), torch.nn.Parameter(torch.tensor(0.5))]
    opt.add_param_group({"params": params[2:]})
    trace = []
    for index in range(4):
        for i, p in enumerate(params):
            p.grad = ((torch.arange(p.numel()).reshape(p.shape) % 7 - 3).float()
                      * (index + 1) * 0.00025 + (i + 1) * 0.0001)
        torch.manual_seed(120 + index)
        opt.step()
        assert_execution_path(opt)
        snapshot = {}
        for i, p in enumerate(params):
            for key, value in {"parameter": p, **opt.state[p]}.items():
                raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                snapshot[f"{i}/{key}/{value.dtype}"] = hashlib.sha256(raw).hexdigest()
        trace.append(snapshot)
    return trace


BASE_CONFIGS = {
    "user": dict(weight_decay=0.1, weight_decouple=True, fixed_decay=False,
                 update_strategy="cautious", betas=(0.9, 0.999, 0.9999)),
    "fp32_defaults": {},
    "bf16_defaults": dict(state_storage_dtype="bfloat16"),
    "cautious_decay": dict(weight_decay=0.1, weight_decouple=False, cautious_weight_decay=True),
}


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("config_name", BASE_CONFIGS)
def test_unchanged_configuration_matches_pre_fix_bits(path, config_name):
    expected = json.loads(Path(__file__).with_name("came_pre_fix_cpu.json").read_text())
    assert baseline_trace(path, BASE_CONFIGS[config_name]) == expected[f"{path}/{config_name}"]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("shape", [(16,), (3, 4)])
@pytest.mark.parametrize("strategy", ["unmodified", "cautious", "grams", "both"])
def test_bug07_coupled_decay_equals_explicit_l2_gradient(path, shape, strategy):
    params, opt = make(path, shape, weight_decay=0.1, weight_decouple=False, update_strategy=strategy)
    refs, ref_opt = make(path, shape, weight_decay=0.0, update_strategy=strategy)
    for value in (0.0, -0.04, 0.02):
        for p in params:
            p.grad = torch.full_like(p, value)
        for p in refs:
            # Gradient of loss + 0.1 / 2 * ||p||^2.
            p.grad = torch.full_like(p, value).add_(p.detach(), alpha=0.1)
        opt.step()
        ref_opt.step()
        assert_execution_path(opt)
        assert_execution_path(ref_opt)
        for p, ref in zip(params, refs):
            torch.testing.assert_close(p, ref, rtol=0, atol=0)
            for key in opt.state[p]:
                # A stacked add with tensor alpha can differ from the explicit
                # per-tensor host-alpha gradient by one fp32 ULP on CPU.
                torch.testing.assert_close(opt.state[p][key], ref_opt.state[ref][key], rtol=2e-6, atol=0)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("shape", [(), (16,), (3, 4), (2, 3, 4)])
def test_bug08_amsbound_uses_running_max_variance(path, shape):
    params, opt = make(path, shape, ams_bound=True, betas=(0.5, 0.5, 0.5),
                       clip_threshold=100.0, eps1=0.0, eps2=1e-8)
    variance = maximum = momentum = residual = 0.0
    expected_p = 1.0
    for gradient in (2.0, 1.0, 0.5, 3.0):
        variance = 0.5 * variance + 0.5 * gradient ** 2
        maximum = max(maximum, variance)
        preconditioned = gradient / math.sqrt(maximum)
        momentum = 0.5 * momentum + 0.5 * preconditioned
        residual = 0.5 * residual + 0.5 * ((preconditioned - momentum) ** 2 + 1e-8)
        update = momentum / math.sqrt(residual) if len(shape) >= 2 else momentum
        expected_p -= 0.01 * update
        step(params, opt, gradient)
        for p in params:
            state = opt.state[p]
            torch.testing.assert_close(state["exp_avg_sq_hat"], torch.full_like(p, maximum))
            torch.testing.assert_close(state["exp_avg"], torch.full_like(p, momentum))
            torch.testing.assert_close(p, torch.full_like(p, expected_p))


@pytest.mark.parametrize("path", ["eager", "compiled", "grouped", "grouped_compiled"])
def test_bug08_factored_maximum_is_reconstructed_per_parameter(path):
    params, opt = make(path, (2, 2), ams_bound=True, betas=(0.5, 0.5, 0.5),
                       clip_threshold=100.0, eps1=0.0)
    # For g=[[1,2],[3,4]], beta2=.5: row=[1.25,6.25], col=[2.5,5],
    # global mean=3.75. Reconstructed v is their outer product / 3.75.
    raw_variance = torch.tensor([[5 / 6, 5 / 3], [25 / 6, 25 / 3]])
    for index in range(3):
        for scale, p in zip((1.0, 3.0), params):
            p.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) * (scale if index == 0 else 0.0)
        opt.step()
        assert_execution_path(opt)
        for scale, p in zip((1.0, 3.0), params):
            torch.testing.assert_close(opt.state[p]["exp_avg_sq_hat"], raw_variance * scale ** 2)


@pytest.mark.parametrize("path", ["eager", "compiled", "foreach"])
@pytest.mark.parametrize("shape", [(16,), (3, 4)])
@pytest.mark.parametrize("dtype,seed", [(torch.float16, 0), (torch.bfloat16, 0),
                                        (torch.bfloat16, 17), (torch.bfloat16, 93)])
def test_bug05_kahan_preserves_exact_sum_across_rounding(path, shape, dtype, seed):
    params, opt = make(path, shape, dtype, lr=1e-4, kahan_sum=True,
                       betas=(0.0, 0.5, 0.0), eps2=1.0)
    torch.manual_seed(seed)
    # Clipping gives an update of -1 in both factored and unfactored paths;
    # beta1=beta3=0 and eps2=1 make confidence exactly 1.
    for index in range(100):
        step(params, opt, -1.0)
        for p in params:
            exact = p.float() + opt.state[p]["kahan_comp"]
            torch.testing.assert_close(exact, torch.full_like(exact, 1 + (index + 1) * 1e-4),
                                       rtol=0, atol=3e-6)


def test_bug05_fp16_accumulates_one_thousand_sub_ulp_updates():
    params, opt = make("eager", dtype=torch.float16, lr=1e-4, kahan_sum=True,
                       betas=(0.0, 0.5, 0.0))
    for _ in range(1000):
        step(params, opt, -1.0)
    for p in params:
        assert torch.equal(p, torch.full_like(p, 1.099609375))


@pytest.mark.parametrize("path", ["eager", "compiled", "foreach"])
@pytest.mark.parametrize("seed", [0, 17, 93])
def test_bug06_fp32_kahan_buffer_keeps_all_mantissa_bits(path, seed):
    params, opt = make(path, dtype=torch.float16, state_storage_dtype="bfloat16",
                       kahan_sum=True, lr=1e-4, betas=(0.0, 0.5, 0.0))
    torch.manual_seed(seed)
    step(params, opt, -1.0)
    for p in params:
        # fp32(1 + 1e-4) - fp16(1 + 1e-4), exactly representable in fp32.
        expected = torch.full(p.shape, 0.00010001659393310547)
        assert opt.state[p]["kahan_comp"].dtype == torch.float32
        assert torch.equal(opt.state[p]["kahan_comp"], expected)


@pytest.mark.parametrize("path", ["eager", "compiled", "foreach"])
@pytest.mark.parametrize("seed", [0, 17, 93])
def test_bug06_fp32_confidence_buffer_keeps_all_mantissa_bits(path, seed):
    params, opt = make(path, state_storage_dtype="bfloat16", non_factored_confidence=True,
                       betas=(0.5, 0.5, 0.75), eps2=0.01234567)
    torch.manual_seed(seed)
    step(params, opt, 1.0)
    for p in params:
        # Clipped update=1, momentum=0.5; (1-beta3)*((1-0.5)^2+eps2).
        expected = torch.full(p.shape, 0.06558641791343689)
        assert opt.state[p]["exp_avg_res"].dtype == torch.float32
        assert torch.equal(opt.state[p]["exp_avg_res"], expected)


@pytest.mark.parametrize("path", ["eager", "compiled", "foreach"])
@pytest.mark.parametrize("shape", [(), (16,)])
@pytest.mark.parametrize("ams_bound", [False, True])
def test_bug09_fp16_storage_retains_tiny_second_moment_and_resume(path, shape, ams_bound):
    params, opt = make(path, shape, dtype=torch.float16, state_storage_dtype="float16",
                       ams_bound=ams_bound)
    beta2 = float(torch.tensor(0.999))
    expected = maximum = 0.0
    for index in range(40):
        gradient = 1e-5 if index < 20 else 0.0
        actual_gradient = float(torch.tensor(gradient, dtype=torch.float16))
        expected = beta2 * expected + (1.0 - beta2) * (actual_gradient ** 2 + 1e-30)
        maximum = max(maximum, expected)
        step(params, opt, gradient)
        for p in params:
            assert torch.isfinite(p).all()
            torch.testing.assert_close(opt.state[p]["exp_avg_sq"].float(),
                                       torch.full(p.shape, expected), rtol=3e-6, atol=0)
            if ams_bound:
                torch.testing.assert_close(opt.state[p]["exp_avg_sq_hat"].float(),
                                           torch.full(p.shape, maximum), rtol=3e-6, atol=0)
        if index == 19:
            saved = copy.deepcopy(opt.state_dict())
            resumed, resumed_opt = make(path, shape, dtype=torch.float16,
                                        state_storage_dtype="float16", ams_bound=ams_bound)
            for p, q in zip(params, resumed):
                q.data.copy_(p)
            resumed_opt.load_state_dict(saved)
        elif index > 19:
            step(resumed, resumed_opt, gradient)
    for p, q in zip(params, resumed):
        assert torch.equal(p, q)
        for key, value in opt.state[p].items():
            assert torch.equal(value, resumed_opt.state[q][key]), key
        assert opt.state[p]["exp_avg_sq"].dtype == torch.float32
        assert resumed_opt.state[q]["exp_avg_sq"].dtype == torch.float32
    opt.reset()
    for p in params:
        assert opt.state[p]["exp_avg_sq"].dtype == torch.float32
        assert torch.count_nonzero(opt.state[p]["exp_avg_sq"]) == 0


@pytest.mark.parametrize("option,key", [("kahan_sum", "kahan_comp"),
                                        ("ams_bound", "exp_avg_sq_hat")])
def test_legacy_checkpoint_converts_bug05_bug08_buffer_convention(option, key):
    params, opt = make("eager", dtype=torch.float16, **{option: True})
    step(params, opt, 0.3)
    saved = copy.deepcopy(opt.state_dict())
    # Simulate the pre-fix format (no version): rounded-exact compensation,
    # sqrt(variance) AMS maximum. Other moments cannot be retroactively fixed.
    for group in saved["param_groups"]:
        group.pop("came_state_version", None)
    expected = []
    for state in saved["state"].values():
        state[key] = -state[key] if option == "kahan_sum" else state[key].sqrt()
        expected.append(-state[key] if option == "kahan_sum" else state[key].square())
    original = copy.deepcopy(saved)
    restored, restored_opt = make("eager", dtype=torch.float16, **{option: True})
    restored_opt.load_state_dict(saved)
    for p, value in zip(restored, expected):
        assert torch.equal(restored_opt.state[p][key], value)
    # Loading must not mutate the caller's checkpoint, nor convert a new save twice.
    for sid in saved["state"]:
        assert torch.equal(saved["state"][sid][key], original["state"][sid][key])
    again, again_opt = make("eager", dtype=torch.float16, **{option: True})
    again_opt.load_state_dict(copy.deepcopy(restored_opt.state_dict()))
    for p, value in zip(again, expected):
        assert torch.equal(again_opt.state[p][key], value)
