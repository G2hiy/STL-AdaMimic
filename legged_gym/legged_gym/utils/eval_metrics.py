"""Per-episode metric collector for ablation evaluation.

Usage:
    collector = RolloutMetricCollector(env, stl_events=cfg.dataset.stl_events,
                                       save_trace_k=16)
    env.reset()
    for _ in range(max_steps):
        # read env state BEFORE step advances reset_idx ...
        actions = policy(obs)
        obs, *_, infos, *_ = env.step(actions)
        collector.step(env, infos)
        if collector.done(target_episodes):
            break
    collector.save(out_dir)

Design notes:
  * `env.post_physics_step()` calls `reset_idx(env_ids)` internally, so by the
    time `step()` returns, the terminated envs are already re-initialized.
  * `extras['success']` / `extras['completions']` are written _inside_
    `reset_idx` before `episode_failed_buf` is zeroed, so for the just-reset
    envs those tensors carry the pre-reset verdict.
  * `reset_buf` is set to 1 for the reset envs at the end of `reset_idx`, so it
    is the cleanest signal to detect episode boundaries from outside.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


@dataclass
class _EnvTrace:
    """Rolling per-step buffer for one env, cleared on episode reset."""
    t: List[float] = field(default_factory=list)
    base_z: List[float] = field(default_factory=list)
    body_err_mm: List[float] = field(default_factory=list)
    joint_err_rad: List[float] = field(default_factory=list)
    feet_force_max: List[float] = field(default_factory=list)
    feet_force_min: List[float] = field(default_factory=list)
    stl_rho: List[float] = field(default_factory=list)
    body_xy_err: List[float] = field(default_factory=list)

    def append(self, **kw):
        for k, v in kw.items():
            getattr(self, k).append(float(v))

    def clear(self):
        for f in self.__dataclass_fields__:
            getattr(self, f).clear()

    def as_np(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(getattr(self, k), dtype=np.float32) for k in self.__dataclass_fields__}


class RolloutMetricCollector:
    """Gathers per-episode metrics + a handful of full-resolution traces."""

    def __init__(
        self,
        env,
        stl_events: Optional[Dict[str, float]] = None,
        save_trace_k: int = 16,
        trace_stride: int = 1,
    ):
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.dt = float(env.dt)
        self.task_id = str(env.cfg.dataset.task_id)

        self.stl_events = dict(stl_events) if stl_events is not None else None
        self.save_trace_k = int(save_trace_k)
        self.trace_stride = int(max(1, trace_stride))

        self._step = 0
        self._traces: List[_EnvTrace] = [_EnvTrace() for _ in range(self.num_envs)]
        self._episodes: List[Dict[str, Any]] = []
        self._saved_traces: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def done(self, target: int) -> bool:
        return len(self._episodes) >= target

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    # ------------------------------------------------------------------
    def _signals(self, env) -> Dict[str, torch.Tensor]:
        """Read current-step signals after env.step() returns."""
        body_err_mm = torch.norm(env.dif_global_body_pos, dim=-1).mean(dim=-1) * 1000.0
        joint_err_rad = torch.norm(env.dif_joint_angles, dim=-1)
        base_z = env.base_pos[:, 2]
        feet_z = env.contact_forces[:, env.feet_contact_indices, 2]
        feet_force_max = feet_z.max(dim=-1).values
        feet_force_min = feet_z.min(dim=-1).values
        body_xy_err = torch.norm(env.dif_global_body_pos[:, :, :2], dim=-1).mean(dim=-1)

        rho = getattr(env, "_stl_rho_cache", None)
        if rho is None:
            rho = torch.full((self.num_envs,), float("nan"), device=self.device)

        t = env.motion_time

        return {
            "t": t,
            "base_z": base_z,
            "body_err_mm": body_err_mm,
            "joint_err_rad": joint_err_rad,
            "feet_force_max": feet_force_max,
            "feet_force_min": feet_force_min,
            "stl_rho": rho,
            "body_xy_err": body_xy_err,
        }

    def step(self, env, infos: Dict[str, Any]) -> None:
        """Call after each env.step() — records signals, then closes episodes
        for envs whose reset_buf==1."""
        sig = self._signals(env)
        sig_cpu = {k: v.detach().cpu() for k, v in sig.items()}

        if self._step % self.trace_stride == 0:
            for i in range(self.num_envs):
                self._traces[i].append(
                    t=sig_cpu["t"][i].item(),
                    base_z=sig_cpu["base_z"][i].item(),
                    body_err_mm=sig_cpu["body_err_mm"][i].item(),
                    joint_err_rad=sig_cpu["joint_err_rad"][i].item(),
                    feet_force_max=sig_cpu["feet_force_max"][i].item(),
                    feet_force_min=sig_cpu["feet_force_min"][i].item(),
                    stl_rho=sig_cpu["stl_rho"][i].item(),
                    body_xy_err=sig_cpu["body_xy_err"][i].item(),
                )

        reset_ids = env.reset_buf.nonzero(as_tuple=False).flatten().tolist()
        if reset_ids:
            success = infos.get("success", None)
            completions = infos.get("completions", None)
            success_cpu = success.detach().cpu() if isinstance(success, torch.Tensor) else None
            comp_cpu = completions.detach().cpu() if isinstance(completions, torch.Tensor) else None

            for env_id in reset_ids:
                self._close_episode(
                    env_id=env_id,
                    success=bool(success_cpu[env_id].item()) if success_cpu is not None else False,
                    completion=float(comp_cpu[env_id].item()) if comp_cpu is not None else float("nan"),
                )

        self._step += 1

    # ------------------------------------------------------------------
    def _close_episode(self, env_id: int, success: bool, completion: float) -> None:
        tr = self._traces[env_id].as_np()
        if tr["t"].size == 0:
            self._traces[env_id].clear()
            return

        record: Dict[str, Any] = {
            "episode_idx": len(self._episodes),
            "env_id": env_id,
            "success": bool(success),
            "completion": float(completion),
            "episode_steps": int(tr["t"].size),
            "episode_duration_s": float(tr["t"].size * self.dt),
            # tracking errors
            "body_err_mean_mm": float(tr["body_err_mm"].mean()),
            "body_err_peak_mm": float(tr["body_err_mm"].max()),
            "joint_err_mean_rad": float(tr["joint_err_rad"].mean()),
            "joint_err_peak_rad": float(tr["joint_err_rad"].max()),
            # jump-related summary (useful for far_jump but harmless elsewhere)
            "base_z_peak_m": float(tr["base_z"].max()),
            # STL (NaN → collector output is "nan" which pandas drops in mean)
            "stl_rho_mean": float(np.nanmean(tr["stl_rho"])) if not np.all(np.isnan(tr["stl_rho"])) else float("nan"),
            "stl_rho_min": float(np.nanmin(tr["stl_rho"])) if not np.all(np.isnan(tr["stl_rho"])) else float("nan"),
        }

        # Task-specific windowed metrics (far_jump: airborne / apex / landing).
        if self.stl_events is not None:
            record.update(self._far_jump_window_metrics(tr))

        self._episodes.append(record)

        if len(self._saved_traces) < self.save_trace_k:
            self._saved_traces.append({
                "episode_idx": record["episode_idx"],
                "env_id": env_id,
                "success": record["success"],
                "trace": tr,
            })

        self._traces[env_id].clear()

    def _far_jump_window_metrics(self, tr: Dict[str, np.ndarray]) -> Dict[str, float]:
        t = tr["t"]
        events = self.stl_events
        takeoff, apex, land = events.get("takeoff"), events.get("apex"), events.get("land")
        if any(v is None for v in (takeoff, apex, land)):
            return {}
        delta_apex = 0.10
        delta_land = 0.15
        out: Dict[str, float] = {}

        m_air = (t >= takeoff) & (t <= land)
        if m_air.any():
            out["airborne_feet_force_peak_N"] = float(tr["feet_force_max"][m_air].max())
        else:
            out["airborne_feet_force_peak_N"] = float("nan")

        m_apex = (t >= apex - delta_apex) & (t <= apex + delta_apex)
        if m_apex.any():
            out["apex_base_z_peak_m"] = float(tr["base_z"][m_apex].max())
        else:
            out["apex_base_z_peak_m"] = float("nan")

        m_land = (t >= land) & (t <= land + delta_land)
        if m_land.any():
            out["land_body_xy_err_min_m"] = float(tr["body_xy_err"][m_land].min())
            out["land_feet_force_min_N"] = float(tr["feet_force_min"][m_land].max())
        else:
            out["land_body_xy_err_min_m"] = float("nan")
            out["land_feet_force_min_N"] = float("nan")
        return out

    # ------------------------------------------------------------------
    def save(self, out_dir: str, tag: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(self._episodes)
        df.insert(0, "tag", tag)
        df.to_csv(os.path.join(out_dir, "episodes.csv"), index=False)

        summary = self._summarize(df)
        summary["tag"] = tag
        summary["task_id"] = self.task_id
        summary["n_episodes"] = int(len(df))
        if extra_meta:
            summary["meta"] = extra_meta
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        torch.save({"tag": tag, "traces": self._saved_traces}, os.path.join(out_dir, "traces.pt"))

        print(f"[eval] saved {len(df)} episodes → {out_dir}")
        print(f"[eval] success={summary['success_rate']:.3f}  "
              f"comp={summary['completion_mean']:.3f}  "
              f"body_err={summary['body_err_mean_mm_mean']:.1f}mm  "
              f"stl_rho={summary.get('stl_rho_mean_mean', float('nan')):.3f}")

    @staticmethod
    def _summarize(df: pd.DataFrame) -> Dict[str, float]:
        if len(df) == 0:
            return {}
        out: Dict[str, float] = {
            "success_rate": float(df["success"].mean()),
            "completion_mean": float(df["completion"].mean()),
            "completion_std": float(df["completion"].std(ddof=0)),
        }
        numeric_cols = [
            "body_err_mean_mm", "body_err_peak_mm", "joint_err_mean_rad", "joint_err_peak_rad",
            "base_z_peak_m", "stl_rho_mean", "stl_rho_min",
            "airborne_feet_force_peak_N", "apex_base_z_peak_m",
            "land_body_xy_err_min_m", "land_feet_force_min_N",
        ]
        for c in numeric_cols:
            if c in df.columns:
                vals = df[c].astype(float)
                out[f"{c}_mean"] = float(vals.mean(skipna=True))
                out[f"{c}_std"] = float(vals.std(ddof=0, skipna=True))
        return out
