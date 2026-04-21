"""STL specification for the `far_jump` task.

Spec (conjunction):
    φ_far_jump =
        G[takeoff, land]   (feet_force_z_max < f_air)          # airborne phase: low contact force
      ∧ F[apex-δ, apex+δ]  (base_z ≥ h_apex_min)               # reach apex height
      ∧ F[land,  land+δ_l] (||body_xy - ref_body_xy|| ≤ d_xy)  # land with body near ref
      ∧ F[land,  land+δ_l] (feet_force_z_min > f_land)         # feet in solid contact after land

Signals are produced by `MotionTrackingEnv._collect_stl_signals()`:
    base_z             : (N,)     — self.base_pos[:, 2]
    feet_force_z_max   : (N,)     — contact_forces[:, feet_contact_indices, 2].max(-1)
    feet_force_z_min   : (N,)     — contact_forces[:, feet_contact_indices, 2].min(-1)
    body_xy_err        : (N,)     — per-env mean L2 distance of keyframe bodies (xy only)
                                     between current and reference (with env_origin_offset).

Event times (takeoff/apex/land) are resolved in this order:
    1. cfg.dataset.stl_events (manual override, dict of floats in seconds)  — 方式 A
    2. auto-dispatch from cfg.dataset.keyframe_times + keyframe_pos_index    — 方式 C
"""
from __future__ import annotations

from typing import Any, Dict, List

from legged_gym.utils.stl_specs import (
    And,
    Eventually,
    Predicate,
    STLContext,
    STLNode,
    make_always,
    make_eventually,
)


# ---------------------------------------------------------------------------
# Event-time resolution
# ---------------------------------------------------------------------------
def _resolve_events(cfg) -> Dict[str, float]:
    manual = getattr(cfg.dataset, "stl_events", None)
    if manual is not None:
        out = {k: float(v) for k, v in dict(manual).items()}
        for key in ("takeoff", "apex", "land"):
            if key not in out:
                raise KeyError(f"cfg.dataset.stl_events missing required key '{key}': {out}")
        return out

    kt = list(cfg.dataset.keyframe_times)
    pi_list = list(cfg.dataset.keyframe_pos_index)
    if len(pi_list) == 0:
        raise ValueError("keyframe_pos_index is empty; cannot auto-dispatch STL events")
    pi = int(pi_list[0])
    return {
        "takeoff": float(kt[max(pi - 1, 0)]),
        "apex":    float(kt[pi]),
        "land":    float(kt[min(pi + 1, len(kt) - 1)]),
    }


# ---------------------------------------------------------------------------
# Predicate factories (closures over constants defined per build)
# ---------------------------------------------------------------------------
def _pred_feet_force_upper(f_air: float) -> Predicate:
    """ρ = f_air − feet_force_z_max  (positive ⇔ airborne)."""
    def fn(sig):
        return f_air - sig["feet_force_z_max"]
    return Predicate(fn, name=f"feet_force_max<{f_air}")


def _pred_feet_force_lower(f_land: float) -> Predicate:
    """ρ = feet_force_z_min − f_land  (positive ⇔ both feet in solid contact)."""
    def fn(sig):
        return sig["feet_force_z_min"] - f_land
    return Predicate(fn, name=f"feet_force_min>{f_land}")


def _pred_base_z_min(h_apex_min: float) -> Predicate:
    """ρ = base_z − h_apex_min."""
    def fn(sig):
        return sig["base_z"] - h_apex_min
    return Predicate(fn, name=f"base_z>{h_apex_min:.3f}")


def _pred_body_xy_err(d_xy: float) -> Predicate:
    """ρ = d_xy − body_xy_err  (positive ⇔ body xy close to ref)."""
    def fn(sig):
        return d_xy - sig["body_xy_err"]
    return Predicate(fn, name=f"body_xy_err<{d_xy:.3f}")


# ---------------------------------------------------------------------------
# Spec builder
# ---------------------------------------------------------------------------
def build(cfg, env_ctx: Dict[str, Any]) -> STLContext:
    """Compile the far_jump STL spec. Returns an STLContext to be stored on the env."""
    events = _resolve_events(cfg)
    takeoff = events["takeoff"]
    apex = events["apex"]
    land = events["land"]
    if not (takeoff < apex <= land):
        raise ValueError(f"invalid event ordering: takeoff={takeoff}, apex={apex}, land={land}")

    # Hyperparameters (overridable via cfg.algorithm.stl_far_jump.*)
    fj_cfg = getattr(cfg.algorithm, "stl_far_jump", None)
    def _get(name, default):
        return float(getattr(fj_cfg, name, default)) if fj_cfg is not None else float(default)

    f_air = _get("f_air", 5.0)            # N
    f_land = _get("f_land", 20.0)         # N
    d_xy = _get("d_xy", 0.15)             # m
    delta_apex = _get("delta_apex", 0.10) # s
    delta_land = _get("delta_land", 0.15) # s
    h_apex_min = _get("h_apex_min", float(env_ctx.get("h_apex_min_default", 0.55)))  # m

    num_envs = int(env_ctx["num_envs"])
    device = env_ctx["device"]

    # --- build sub-specs (each window has its own accumulator) --------------
    g_airborne = make_always(
        _pred_feet_force_upper(f_air),
        a=takeoff, b=land, num_envs=num_envs, device=device,
    )
    f_apex = make_eventually(
        _pred_base_z_min(h_apex_min),
        a=max(apex - delta_apex, 0.0), b=apex + delta_apex,
        num_envs=num_envs, device=device,
    )
    f_land_xy = make_eventually(
        _pred_body_xy_err(d_xy),
        a=land, b=land + delta_land,
        num_envs=num_envs, device=device,
    )
    f_land_contact = make_eventually(
        _pred_feet_force_lower(f_land),
        a=land, b=land + delta_land,
        num_envs=num_envs, device=device,
    )

    spec_root: STLNode = And(g_airborne, f_apex, f_land_xy, f_land_contact)
    accumulators: List = [g_airborne.acc, f_apex.acc, f_land_xy.acc, f_land_contact.acc]

    meta = {
        "f_air": f_air,
        "f_land": f_land,
        "d_xy": d_xy,
        "delta_apex": delta_apex,
        "delta_land": delta_land,
        "h_apex_min": h_apex_min,
    }
    return STLContext(spec_root=spec_root, accumulators=accumulators, events=events, meta=meta)
