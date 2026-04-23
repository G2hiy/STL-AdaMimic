"""Sanity checks for legged_gym.utils.stl_specs DSL + stl_tasks.far_jump.

Runs locally (CPU) without isaacgym. Verifies:
  1. Predicate / And / Or / Not forward shape, dtype, autograd
  2. Always (G) + Eventually (F) with EventWindowAccumulator:
       - window-outside signals don't affect the running reduce
       - G returns softmin over in-window ρ, F returns softmax
  3. far_jump spec end-to-end on two mock trajectories:
       - one satisfies φ → ρ_final > 0
       - one violates  φ → ρ_final < 0
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Import stl_specs.py and stl_tasks/far_jump.py without triggering
# legged_gym/utils/__init__.py (which imports isaacgym-heavy modules).
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1] / "legged_gym" / "legged_gym" / "utils"


def _spec_load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


stl_specs = _spec_load("stl_specs", _ROOT / "stl_specs.py")

# far_jump imports `from legged_gym.utils.stl_specs import ...`; alias that path
# to the freshly loaded module to short-circuit the package import chain.
_legged_gym_pkg = types.ModuleType("legged_gym")
_legged_gym_pkg.__path__ = []
_utils_pkg = types.ModuleType("legged_gym.utils")
_utils_pkg.__path__ = []
sys.modules.setdefault("legged_gym", _legged_gym_pkg)
sys.modules.setdefault("legged_gym.utils", _utils_pkg)
sys.modules["legged_gym.utils.stl_specs"] = stl_specs

far_jump = _spec_load(
    "stl_tasks_far_jump", _ROOT / "stl_tasks" / "far_jump.py"
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _pred_gt(name: str, threshold: float, field: str):
    return stl_specs.Predicate(lambda sig, f=field, t=threshold: sig[f] - t, name=name)


def _pred_lt(name: str, threshold: float, field: str):
    return stl_specs.Predicate(lambda sig, f=field, t=threshold: t - sig[f], name=name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_dsl_autograd():
    p = stl_specs.Predicate(lambda sig: sig["x"] - 1.0, name="x>1")
    q = stl_specs.Predicate(lambda sig: 2.0 - sig["y"], name="y<2")
    spec = stl_specs.And(p, q)
    x = torch.randn(8, requires_grad=True)
    y = torch.randn(8, requires_grad=True)
    rho = spec.robustness({"x": x, "y": y}, motion_time=torch.zeros(8), beta=10.0)
    rho.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert y.grad is not None and torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0


def test_not_or_shape():
    p = stl_specs.Predicate(lambda sig: sig["x"], name="x")
    q = stl_specs.Predicate(lambda sig: -sig["x"], name="-x")
    rho_or = stl_specs.Or(p, q).robustness({"x": torch.tensor([1.0, -2.0, 0.5])}, torch.zeros(3), beta=5.0)
    assert rho_or.shape == (3,)
    assert (rho_or >= -1e-3).all()  # max(x, -x) = |x| ≥ 0

    rho_not = stl_specs.Not(p).robustness({"x": torch.tensor([2.0, -1.0])}, torch.zeros(2), beta=5.0)
    assert torch.allclose(rho_not, torch.tensor([-2.0, 1.0]))


def test_always_masking_semantics():
    """G_[0.5, 1.5] (x > 0): running softmin over in-window steps only."""
    pred = _pred_gt("x_gt_0", 0.0, "x")
    G = stl_specs.make_always(pred, a=0.5, b=1.5, num_envs=2, device="cpu")

    # Trajectory: t = [0.0, 0.6, 1.0, 1.4, 1.8]
    # env0 in-window signals = [3, 5, 4]  → softmin ≈ 3
    # env1 in-window signals = [3,-1, 4]  → softmin ≈ -1
    times = [0.0, 0.6, 1.0, 1.4, 1.8]
    xs = [[100.0, 100.0], [3.0, 3.0], [5.0, -1.0], [4.0, 4.0], [-100.0, -100.0]]
    for t, x in zip(times, xs):
        G.acc.step(torch.tensor([t, t]), {"x": torch.tensor(x)}, beta=20.0)

    rho = G.robustness({"x": torch.tensor([0.0, 0.0])}, torch.tensor([2.0, 2.0]), beta=20.0)
    assert torch.isfinite(rho).all()
    assert 2.5 < rho[0].item() < 3.5, f"env0 ρ={rho[0].item()}"
    assert -1.5 < rho[1].item() < -0.5, f"env1 ρ={rho[1].item()}"


def test_eventually_masking_semantics():
    """F_[0.5, 1.5] (x > 0): running softmax over in-window steps only."""
    pred = _pred_gt("x_gt_0", 0.0, "x")
    F = stl_specs.make_eventually(pred, a=0.5, b=1.5, num_envs=2, device="cpu")
    times = [0.0, 0.7, 1.0, 1.4, 2.0]
    # env0 in-window = [-1, 2, 0] → softmax ≈ 2
    # env1 in-window = [-3, -2, -1] → softmax ≈ -1
    xs = [[100.0, 100.0], [-1.0, -3.0], [2.0, -2.0], [0.0, -1.0], [100.0, 100.0]]
    for t, x in zip(times, xs):
        F.acc.step(torch.tensor([t, t]), {"x": torch.tensor(x)}, beta=20.0)
    rho = F.robustness({"x": torch.tensor([0.0, 0.0])}, torch.tensor([2.5, 2.5]), beta=20.0)
    assert 1.5 < rho[0].item() < 2.5
    assert -1.5 < rho[1].item() < -0.5


def test_event_window_reset():
    pred = _pred_gt("x_gt_0", 0.0, "x")
    G = stl_specs.make_always(pred, a=0.0, b=1.0, num_envs=3, device="cpu")
    for t, x in [(0.1, [-1.0, 2.0, 3.0]), (0.5, [-2.0, 1.0, 5.0])]:
        G.acc.step(torch.tensor([t, t, t]), {"x": torch.tensor(x)}, beta=20.0)
    G.acc.reset(torch.tensor([0]))
    assert not G.acc.seen[0]
    assert G.acc.seen[1] and G.acc.seen[2]


def _mock_cfg(events=None):
    class _NS:
        pass

    cfg = _NS()
    cfg.dataset = _NS()
    cfg.dataset.keyframe_times = [0.1, 0.3, 0.5, 0.7, 1.0, 1.4, 1.7, 2.0, 2.5, 2.8,
                                  3.0, 3.1, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.3, 5.5]
    cfg.dataset.keyframe_pos_index = [12]
    cfg.dataset.stl_events = events
    cfg.algorithm = _NS()
    cfg.algorithm.stl_far_jump = None
    return cfg


def test_far_jump_spec_satisfying_trajectory():
    cfg = _mock_cfg(events={"takeoff": 3.0, "apex": 3.3, "land": 3.6})
    ctx = far_jump.build(cfg, env_ctx={"device": "cpu", "num_envs": 1, "h_apex_min_default": 0.4})

    # Use a slightly wider mock signal envelope than the STL windows — fp32
    # rounding on motion_time can nudge boundary steps ±ε inside the window,
    # so the mock must supply "airborne" / "landed" values for a bit beyond
    # the exact event times.
    slack = 0.08
    dt = 0.05
    t = 0.0
    while t <= 4.0:
        airborne = (3.0 - slack) <= t <= (3.6 + slack)
        near_apex = abs(t - 3.3) <= (0.1 + slack)
        land_win = (3.6 - slack) <= t <= (3.75 + slack)
        base_z = 0.7 if near_apex else 0.6
        force_max = 1.0 if airborne else 50.0
        force_min = 40.0 if land_win else 30.0
        body_xy_err = 0.05 if land_win else 0.5
        sig = {
            "base_z": torch.tensor([base_z]),
            "feet_force_z_max": torch.tensor([force_max]),
            "feet_force_z_min": torch.tensor([force_min]),
            "body_xy_err": torch.tensor([body_xy_err]),
        }
        for acc in ctx.accumulators:
            acc.step(torch.tensor([t]), sig, beta=10.0)
        t += dt

    final_sig = {
        "base_z": torch.tensor([0.6]),
        "feet_force_z_max": torch.tensor([50.0]),
        "feet_force_z_min": torch.tensor([30.0]),
        "body_xy_err": torch.tensor([0.05]),
    }
    rho = ctx.spec_root.robustness(final_sig, torch.tensor([4.0]), beta=10.0)
    assert rho.item() > 0, f"satisfying trajectory should have ρ > 0, got {rho.item()}"


def test_far_jump_spec_violating_trajectory():
    cfg = _mock_cfg(events={"takeoff": 3.0, "apex": 3.3, "land": 3.6})
    ctx = far_jump.build(cfg, env_ctx={"device": "cpu", "num_envs": 1, "h_apex_min_default": 0.4})

    dt = 0.05
    t = 0.0
    while t <= 4.0:
        sig = {
            "base_z": torch.tensor([0.45]),            # below apex threshold
            "feet_force_z_max": torch.tensor([100.0]),  # fails airborne G predicate
            "feet_force_z_min": torch.tensor([5.0]),    # fails land-contact predicate
            "body_xy_err": torch.tensor([0.4]),         # exceeds d_xy
        }
        for acc in ctx.accumulators:
            acc.step(torch.tensor([t]), sig, beta=10.0)
        t += dt

    final_sig = {
        "base_z": torch.tensor([0.45]),
        "feet_force_z_max": torch.tensor([100.0]),
        "feet_force_z_min": torch.tensor([5.0]),
        "body_xy_err": torch.tensor([0.4]),
    }
    rho = ctx.spec_root.robustness(final_sig, torch.tensor([4.0]), beta=10.0)
    assert rho.item() < 0, f"violating trajectory should have ρ < 0, got {rho.item()}"


def test_far_jump_auto_dispatch():
    """No stl_events → resolver falls back to keyframe_pos_index ±1 auto-dispatch."""
    cfg = _mock_cfg(events=None)
    ctx = far_jump.build(cfg, env_ctx={"device": "cpu", "num_envs": 1})
    # pos_index=12 → kt[11]=3.1 (takeoff), kt[12]=3.6 (apex), kt[13]=3.9 (land)
    assert ctx.events == {"takeoff": 3.1, "apex": 3.6, "land": 3.9}, ctx.events


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    print("[usage] Run `python scripts/check_stl_specs.py` before Stage 1 training with use_stl_reward=true.")
    tests = [
        ("DSL autograd",                     test_dsl_autograd),
        ("Not / Or shape + values",          test_not_or_shape),
        ("Always masking semantics",         test_always_masking_semantics),
        ("Eventually masking semantics",     test_eventually_masking_semantics),
        ("EventWindowAccumulator.reset",     test_event_window_reset),
        ("far_jump satisfying ρ>0",          test_far_jump_spec_satisfying_trajectory),
        ("far_jump violating  ρ<0",          test_far_jump_spec_violating_trajectory),
        ("far_jump event auto-dispatch",     test_far_jump_auto_dispatch),
    ]
    for name, fn in tests:
        fn()
        print(f"[OK] {name}")
    print("[OK] STL specs self-check passed.")


if __name__ == "__main__":
    main()
