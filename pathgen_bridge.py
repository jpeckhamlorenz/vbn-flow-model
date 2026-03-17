"""pathgen_bridge.py — Convert vbn_pathgen trajectories to iLQR format and run optimization.

Part A: Conversion utilities
    pathgen_to_traj()  — DataFrame → dict (SI units, resampled)
    pathgen_to_npz()   — DataFrame → .npz file
    Trajectory registry for CLI lookup

Part B: High-level iLQR runner
    run_ilqr_on_pathgen()  — full pipeline: pathgen DF → iLQR → results dict
    load_ilqr_results()    — load pre-computed controls .npz
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ── Ensure inputs/ is importable ─────────────────────────────────────────────
_INPUTS_DIR = str(Path(__file__).resolve().parent / "inputs")
if _INPUTS_DIR not in sys.path:
    sys.path.insert(0, _INPUTS_DIR)


# ======================================================================
#  Part A — Conversion utilities
# ======================================================================

def pathgen_to_traj(
    df: pd.DataFrame,
    dt_target: float = 0.01,
) -> dict:
    """Convert a pathgen DataFrame to the iLQR trajectory dict format.

    Args:
        df: DataFrame from PathGenerator.generate_path() or a trajectory
            function (e.g., m_VBN).  Expected columns: ``t``, ``q``, ``w_pred``.
            Units: t [s], q [mm^3/s], w_pred [mm].
        dt_target: Target timestep [s].  If the pathgen dt differs,
            the trajectory is resampled via linear interpolation.

    Returns:
        dict with keys:
            t      [N]  float64  — uniform time axis [s]
            dt     float         — actual timestep [s]
            N      int           — number of samples
            Q_com  [N]  float64  — commanded flowrate [m^3/s]
            W_com  [N]  float64  — predicted bead width [m]
            Q_vbn  None          — not available from pathgen
            Q_res  None          — not available from pathgen
            Q_tru  None          — not available from pathgen
    """
    for col in ("t", "q", "w_pred"):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column '{col}'. "
                             f"Available: {list(df.columns)}")

    t_raw = df["t"].to_numpy().astype(np.float64)
    q_raw = df["q"].to_numpy().astype(np.float64)
    w_raw = df["w_pred"].to_numpy().astype(np.float64)

    # Convert units: mm^3/s → m^3/s, mm → m
    Q_com_raw = q_raw * 1e-9
    W_com_raw = w_raw * 1e-3

    # Detect source dt and resample if needed
    dt_raw = float(np.median(np.diff(t_raw)))
    ratio = dt_raw / dt_target

    if abs(ratio - round(ratio)) < 0.05 and round(ratio) >= 2:
        # Clean integer downsample (e.g., 0.02 → 0.01 by interp, or 0.01 → 0.01 no-op)
        # Since dt_raw > dt_target, we need to upsample via interpolation
        t_new = np.arange(t_raw[0], t_raw[-1], dt_target)
        Q_com = np.interp(t_new, t_raw, Q_com_raw)
        W_com = np.interp(t_new, t_raw, W_com_raw)
        dt = dt_target
        print(f"  Resampled: dt {dt_raw:.4f} s → {dt:.4f} s  "
              f"({len(t_raw)} → {len(t_new)} samples)")
    elif abs(ratio - 1.0) < 0.05:
        # Already at target dt
        t_new = t_raw
        Q_com = Q_com_raw
        W_com = W_com_raw
        dt = dt_raw
    else:
        # General interpolation
        t_new = np.arange(t_raw[0], t_raw[-1], dt_target)
        Q_com = np.interp(t_new, t_raw, Q_com_raw)
        W_com = np.interp(t_new, t_raw, W_com_raw)
        dt = dt_target
        print(f"  Resampled: dt {dt_raw:.4f} s → {dt:.4f} s  "
              f"({len(t_raw)} → {len(t_new)} samples)")

    N = len(t_new)
    # Re-create uniform time axis from 0
    t_uniform = np.arange(N, dtype=np.float64) * dt

    _SCALE = 6e7  # m^3/s → mL/min
    print(f"  N = {N} samples  ({N * dt:.2f} s total)")
    print(f"  Q_com: [{Q_com.min() * _SCALE:.3f}, {Q_com.max() * _SCALE:.3f}] mL/min")
    print(f"  W_com: [{W_com.min() * 1e3:.2f}, {W_com.max() * 1e3:.2f}] mm")

    return dict(
        t=t_uniform, dt=dt, N=N,
        Q_com=Q_com, W_com=W_com,
        Q_vbn=None, Q_res=None, Q_tru=None,
    )


def pathgen_to_npz(
    df: pd.DataFrame,
    path: Union[str, Path],
    dt_target: float = 0.01,
) -> Path:
    """Convert a pathgen DataFrame and save as .npz for run_ilqr_deploy.py.

    Args:
        df: pathgen DataFrame (same as pathgen_to_traj).
        path: Output .npz path.
        dt_target: Target timestep [s].

    Returns:
        Resolved path to the saved .npz file.
    """
    traj = pathgen_to_traj(df, dt_target=dt_target)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        time=traj["t"],
        Q_com=traj["Q_com"],
        W_com=traj["W_com"],
    )
    print(f"  Saved trajectory: {out}")
    return out.resolve()


# ── Trajectory registry ──────────────────────────────────────────────────────

TRAJECTORY_REGISTRY: Dict[str, Callable] = {}


def _populate_registry() -> None:
    """Import trajectory functions from vbn_pathgen and register them."""
    try:
        from vbn_pathgen import (
            m_VBN,
            test_line_gen,
            line_width_step,
            corner_flowmatch,
            bead_flow_constant,
            sealant_static,
            sealant_VBN,
            sealant_scan,
            bed_leveling,
            dotted_line_columns,
            dashed_line_columns,
        )
        for name, func in [
            ("m_VBN", m_VBN),
            ("test_line_gen", test_line_gen),
            ("line_width_step", line_width_step),
            ("corner_flowmatch", corner_flowmatch),
            ("bead_flow_constant", bead_flow_constant),
            ("sealant_static", sealant_static),
            ("sealant_VBN", sealant_VBN),
            ("sealant_scan", sealant_scan),
            ("bed_leveling", bed_leveling),
            ("dotted_line_columns", dotted_line_columns),
            ("dashed_line_columns", dashed_line_columns),
        ]:
            TRAJECTORY_REGISTRY[name] = func
    except ImportError as e:
        print(f"  Warning: could not import vbn_pathgen: {e}")


def get_trajectory_func(name: str) -> Callable:
    """Look up a trajectory function by name.

    Raises KeyError with available names if not found.
    """
    if not TRAJECTORY_REGISTRY:
        _populate_registry()
    if name not in TRAJECTORY_REGISTRY:
        avail = ", ".join(sorted(TRAJECTORY_REGISTRY.keys()))
        raise KeyError(f"Unknown trajectory '{name}'. Available: {avail}")
    return TRAJECTORY_REGISTRY[name]


# ======================================================================
#  Part B — High-level iLQR runner
# ======================================================================

_SCALE = 6e7  # m^3/s → mL/min


def _resolve_checkpoint(sweep_id: str = "mnywg829",
                        ckpt_path: Optional[str] = None,
                        config_path: Optional[str] = None):
    """Resolve checkpoint path and run config.

    Returns (ckpt_path, run_config).
    """
    if ckpt_path is not None:
        cp = Path(ckpt_path)
        if not cp.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cp}")
        if config_path is not None:
            cfg_path = Path(config_path)
        else:
            candidates = [Path("config.json"), cp.parent.parent / "config.json"]
            cfg_path = next((c for c in candidates if c.exists()), None)
            if cfg_path is None:
                raise FileNotFoundError(
                    "--ckpt_path given without --config_path; no config.json found.")
        with open(cfg_path) as f:
            cfg = json.load(f)
        if "huber_delta" not in cfg:
            cfg["huber_delta"] = 1.0
        from models.traj_WALR import DictToObject
        return cp, DictToObject(cfg)
    else:
        print(f"  Resolving best checkpoint from WandB sweep '{sweep_id}'...")
        from models.traj_WALR import get_best_run
        run_id, run_config = get_best_run(sweep_id=sweep_id)
        ckpt_dir = Path("VBN-modeling") / run_id / "checkpoints"
        ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
        ckpt_path_resolved = ckpt_files[-1]
        print(f"  Checkpoint: {ckpt_path_resolved}")
        return ckpt_path_resolved, run_config


def load_ilqr_results(path: Union[str, Path]) -> dict:
    """Load a previously saved controls .npz and return a results dict.

    Args:
        path: Path to controls .npz file (from run_ilqr_on_pathgen or
              run_ilqr_deploy.py).

    Returns:
        dict with keys matching run_ilqr_on_pathgen() output.
    """
    d = np.load(path)
    return dict(
        t=d["t"],
        Q_cmd_opt=d["Q_cmd_opt"],
        w_cmd_opt=d["w_cmd_opt"],
        Q_cmd_naive=d["Q_cmd_naive"],
        w_cmd_naive=d["w_cmd_naive"],
        Q_out_opt=d["Q_out_stepmode_opt"],
        Q_out_naive=d["Q_out_stepmode_naive"],
        Q_com=d["Q_com"],
        cost_hist=[],  # not stored in .npz
    )


def run_ilqr_on_pathgen(
    df: pd.DataFrame,
    ilqr_kwargs: Optional[Dict[str, Any]] = None,
    dt_target: float = 0.01,
    save_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Run the full iLQR FFEC pipeline on a pathgen trajectory.

    Args:
        df: DataFrame from PathGenerator.generate_path() or a trajectory
            function.  Expected columns: t, q, w_pred (pathgen units).
        ilqr_kwargs: Override iLQR parameters.  Supported keys:
            G, G_f, R_diag, S_diag, segment_len, max_iter, n_samples,
            w_delta_plus, w_delta_minus, Q_min, Q_max, w_min, w_max,
            analytical_only, sweep_id, ckpt_path, config_path,
            device, tol, eps_ode, eps_ctrl_rel, eps_ctrl_floor_Q,
            eps_ctrl_floor_w.
        dt_target: Target timestep [s] for resampling.
        save_path: If provided, save controls .npz to this path.

    Returns:
        dict with keys:
            t              [N]  time axis [s]
            Q_cmd_opt      [N]  optimized flowrate command [m^3/s]
            w_cmd_opt      [N]  optimized width command [m]
            Q_cmd_naive    [N]  naive flowrate command [m^3/s]
            w_cmd_naive    [N]  naive width command [m]
            Q_out_opt      [N]  predicted output under opt controls [m^3/s]
            Q_out_naive    [N]  predicted output under naive controls [m^3/s]
            Q_com          [N]  reference/tracking target [m^3/s]
            cost_hist      list of cost values per iteration
    """
    from ilqr_config import ILQRConfig

    kw = ilqr_kwargs or {}

    # ── Build config ─────────────────────────────────────────────────────
    cfg = ILQRConfig()
    # Apply overrides from ilqr_kwargs
    _cfg_map = {
        "G":                 ("cost",       "G"),
        "G_f":               ("cost",       "G_f"),
        "R_diag":            ("cost",       "R_diag"),
        "S_diag":            ("cost",       "S_diag"),
        "max_iter":          ("solver",     "max_iter"),
        "tol":               ("solver",     "tol"),
        "eps_ode":           ("solver",     "eps_ode"),
        "eps_ctrl_rel":      ("solver",     "eps_ctrl_rel"),
        "eps_ctrl_floor_Q":  ("solver",     "eps_ctrl_floor_Q"),
        "eps_ctrl_floor_w":  ("solver",     "eps_ctrl_floor_w"),
        "Q_min":             ("bounds",     "Q_min"),
        "Q_max":             ("bounds",     "Q_max"),
        "w_min":             ("bounds",     "w_min"),
        "w_max":             ("bounds",     "w_max"),
        "w_delta_plus":      ("bounds",     "w_delta_plus"),
        "w_delta_minus":     ("bounds",     "w_delta_minus"),
        "Q_delta_plus":      ("bounds",     "Q_delta_plus"),
        "Q_delta_minus":     ("bounds",     "Q_delta_minus"),
        "segment_len":       ("bounds",     "segment_len"),
        "n_samples":         ("run",        "n_samples"),
        "analytical_only":   ("run",        "analytical_only"),
        "sweep_id":          ("checkpoint", "sweep_id"),
        "ckpt_path":         ("checkpoint", "ckpt_path"),
        "config_path":       ("checkpoint", "config_path"),
        "device":            ("display",    "device"),
    }
    for key, value in kw.items():
        if key in _cfg_map:
            group_name, field_name = _cfg_map[key]
            sub = getattr(cfg, group_name)
            setattr(sub, field_name, value)
        else:
            print(f"  Warning: unknown ilqr_kwarg '{key}' — ignored")

    if cfg.cost.G_f is None:
        cfg.cost.G_f = cfg.cost.G

    # ── Convert trajectory ───────────────────────────────────────────────
    print("\n== Converting pathgen trajectory ==")
    traj = pathgen_to_traj(df, dt_target=dt_target)
    t_full = traj["t"]
    dt     = traj["dt"]
    N_full = traj["N"]
    Q_com  = traj["Q_com"]
    W_com  = traj["W_com"]

    N_ilqr = min(cfg.run.n_samples, N_full)
    t_ilqr = t_full[:N_ilqr]
    print(f"  iLQR horizon: {N_ilqr} steps ({N_ilqr * dt:.2f} s)")

    # ── Checkpoint ───────────────────────────────────────────────────────
    print("\n== Checkpoint resolution ==")
    ckpt_path, run_config = _resolve_checkpoint(
        sweep_id=cfg.checkpoint.sweep_id,
        ckpt_path=cfg.checkpoint.ckpt_path,
        config_path=cfg.checkpoint.config_path,
    )
    print(f"  num_layers={run_config.num_layers}  hidden_size={run_config.hidden_size}")

    # ── Initialise MATLAB bridge + LSTM ──────────────────────────────────
    print("\n== Initialising pipeline components ==")
    from matlab_bridge import MatlabBridge
    bridge = MatlabBridge(
        fluid=cfg.physics.fluid,
        mixer=cfg.physics.mixer,
        pump=cfg.physics.pump,
    )

    from lstm_step import LSTMStepWrapper
    lstm = LSTMStepWrapper(
        ckpt_path=ckpt_path,
        run_config=run_config,
        train_list_path=cfg.data.train_list,
        data_folder=cfg.data.data_folder,
        device=cfg.display.device,
    )

    # ── Hybrid dynamics + naive rollout ───────────────────────────────────
    from dynamics import HybridDynamics
    dyn    = HybridDynamics(bridge=bridge, lstm=lstm, dt=dt)
    state0 = dyn.make_initial_state(t0=float(t_full[0]))

    if not cfg.run.analytical_only:
        dyn.use_lstm = True

    U_naive = np.stack([Q_com[:N_ilqr], W_com[:N_ilqr]], axis=1)
    print(f"\n  Rolling out HybridDynamics (naive) for {N_ilqr} steps ...")
    _, Q_out_naive = dyn.rollout(state0, U_naive)

    rmse_naive = np.sqrt(np.mean((Q_out_naive - Q_com[:N_ilqr]) ** 2)) * _SCALE
    print(f"  Naive step-mode RMSE vs Q_com: {rmse_naive:.4f} mL/min")

    # ── Build control bounds ─────────────────────────────────────────────
    bnd = cfg.bounds
    _use_rel = bnd.w_delta_plus is not None or bnd.w_delta_minus is not None
    if _use_rel:
        dp = bnd.w_delta_plus  if bnd.w_delta_plus  is not None else np.inf
        dm = bnd.w_delta_minus if bnd.w_delta_minus is not None else np.inf
        w_lo = np.maximum(bnd.w_min, W_com[:N_ilqr] - dm)
        w_hi = np.minimum(bnd.w_max, W_com[:N_ilqr] + dp)

        # Also handle Q bounds if specified
        if bnd.Q_delta_plus is not None or bnd.Q_delta_minus is not None:
            q_dp = bnd.Q_delta_plus  if bnd.Q_delta_plus  is not None else np.inf
            q_dm = bnd.Q_delta_minus if bnd.Q_delta_minus is not None else np.inf
            q_lo = np.maximum(bnd.Q_min, Q_com[:N_ilqr] - q_dm)
            q_hi = np.minimum(bnd.Q_max, Q_com[:N_ilqr] + q_dp)
        else:
            q_lo = np.full(N_ilqr, bnd.Q_min)
            q_hi = np.full(N_ilqr, bnd.Q_max)

        u_min = np.column_stack([q_lo, w_lo])  # [N, 2]
        u_max = np.column_stack([q_hi, w_hi])  # [N, 2]
    else:
        u_min = np.array([bnd.Q_min, bnd.w_min])
        u_max = np.array([bnd.Q_max, bnd.w_max])

    # ── iLQR solve ───────────────────────────────────────────────────────
    print(f"\n== iLQR solve ==")
    print(f"  G={cfg.cost.G:.2e}  G_f={cfg.cost.G_f:.2e}  R=diag{cfg.cost.R_diag}")
    if cfg.cost.S_diag is not None:
        print(f"  S=diag{cfg.cost.S_diag}  (soft rate penalty)")

    R_mat = np.diag(cfg.cost.R_diag)
    S_mat = np.diag(cfg.cost.S_diag) if cfg.cost.S_diag is not None else None
    q_ref = Q_com[:N_ilqr].copy()

    from ilqr import ILQRSolver
    solver = ILQRSolver(
        dynamics=dyn,
        G=cfg.cost.G,
        R=R_mat,
        G_f=cfg.cost.G_f,
        S=S_mat,
        max_iter=cfg.solver.max_iter,
        tol=cfg.solver.tol,
        eps_ode=cfg.solver.eps_ode,
        eps_ctrl=getattr(cfg.solver, "eps_ctrl", 1e-7),
        eps_ctrl_rel=cfg.solver.eps_ctrl_rel,
        eps_ctrl_floor_Q=cfg.solver.eps_ctrl_floor_Q,
        eps_ctrl_floor_w=cfg.solver.eps_ctrl_floor_w,
        verbose=True,
    )

    # ---- Segmented iLQR loop ----
    seg_len = bnd.segment_len or N_ilqr
    n_segs  = math.ceil(N_ilqr / seg_len)
    if n_segs > 1:
        print(f"  Segmented iLQR: {n_segs} x {seg_len}-step segments  "
              f"({seg_len * dt:.2f} s each, LSTM h,c reset per segment)")

    U_opt      = np.copy(U_naive)
    Q_out_iLQR = np.zeros(N_ilqr)
    cost_hist: List[float] = []

    from tqdm import tqdm
    seg_iter = (tqdm(range(n_segs), desc="Segments", leave=True)
                if n_segs > 1 else range(n_segs))

    for s_idx in seg_iter:
        s0 = s_idx * seg_len
        s1 = min(s0 + seg_len, N_ilqr)

        if n_segs > 1:
            dyn.use_lstm = not cfg.run.analytical_only
            mode_str = "hybrid" if dyn.use_lstm else "analytical-only"
            print(f"\n  --- Segment {s_idx + 1}/{n_segs}: "
                  f"steps [{s0}:{s1}]  ({s0*dt:.2f}-{s1*dt:.2f} s) "
                  f"[{mode_str}] ---")

        # Slice bounds
        um  = (u_min[s0:s1] if isinstance(u_min, np.ndarray) and u_min.ndim > 1
               else u_min)
        umx = (u_max[s0:s1] if isinstance(u_max, np.ndarray) and u_max.ndim > 1
               else u_max)

        # Build prefix from optimised controls of all previous segments
        U_prefix = U_opt[:s0].copy() if s0 > 0 else None
        state0_global = state0 if s0 > 0 else None

        U_seg, Q_seg, ch = solver.solve(
            state0=state0,
            q_ref=q_ref[s0:s1],
            U_init=U_naive[s0:s1].copy(),
            u_min=um, u_max=umx,
            w_rate_max=(bnd.w_rate_max * 1e-3 * dt if bnd.w_rate_max is not None
                        else None),
            q_rate_max=(bnd.q_rate_max * (1e-6 / 60) * dt if bnd.q_rate_max is not None
                        else None),
            U_prefix=U_prefix,
            state0_global=state0_global,
        )

        U_opt[s0:s1]      = U_seg
        Q_out_iLQR[s0:s1] = Q_seg
        cost_hist.extend(ch)

    rmse_ilqr = np.sqrt(np.mean((Q_out_iLQR - q_ref) ** 2)) * _SCALE
    if len(cost_hist) >= 2:
        reduction = (cost_hist[0] - cost_hist[-1]) / abs(cost_hist[0]) * 100
        print(f"\n  iLQR RMSE vs Q_com: {rmse_ilqr:.4f} mL/min  "
              f"(cost reduction {reduction:.1f}%,  {len(cost_hist)} iters)")
    print(f"  Naive RMSE:  {rmse_naive:.4f}  ->  iLQR RMSE:  {rmse_ilqr:.4f}  "
          f"({'improved' if rmse_ilqr < rmse_naive else 'no improvement'})")

    # ── Build results dict ───────────────────────────────────────────────
    results = dict(
        t=t_ilqr,
        Q_cmd_opt=U_opt[:, 0],
        w_cmd_opt=U_opt[:, 1],
        Q_cmd_naive=U_naive[:, 0],
        w_cmd_naive=U_naive[:, 1],
        Q_out_opt=Q_out_iLQR,
        Q_out_naive=Q_out_naive,
        Q_com=q_ref,
        cost_hist=cost_hist,
    )

    # ── Save controls .npz ───────────────────────────────────────────────
    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            sp,
            t=t_ilqr,
            Q_cmd_opt=U_opt[:, 0],
            w_cmd_opt=U_opt[:, 1],
            Q_cmd_naive=U_naive[:, 0],
            w_cmd_naive=U_naive[:, 1],
            Q_com=q_ref,
            Q_out_stepmode_opt=Q_out_iLQR,
            Q_out_stepmode_naive=Q_out_naive,
        )
        print(f"\n  Controls saved: {sp}")

    # ── Quit MATLAB bridge ───────────────────────────────────────────────
    bridge.quit()

    return results

def smooth_segment_boundaries(
    t_cmd_opt: np.ndarray,
    Q_cmd_opt: np.ndarray,
    w_cmd_opt: np.ndarray,
    segment_len: int = 450,
    dt_original: float = 0.01,
    pre_boundary_steps: int = 10,
    post_boundary_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace segment-boundary spike artifacts with linear interpolation.

    The segmented iLQR solver resets LSTM hidden state at each segment
    boundary, producing transient spikes in Q_cmd and w_cmd near the
    junction.  This function identifies the affected samples and replaces
    them with a linear interpolation between the nearest clean points on
    either side.

    Boundary locations are found in *time* (multiples of
    ``segment_len * dt_original``) rather than by index, so the function
    remains correct even if the trajectories have been resampled to a
    different timestep or made non-uniform after iLQR.

    The ``pre/post_boundary_steps`` parameters refer to steps in the
    *current* time vector ``t_cmd_opt`` (not the original iLQR grid).

    Args:
        t_cmd_opt:  [N] time axis [s] corresponding to Q/w arrays.
        Q_cmd_opt:  [N] optimised flowrate command [m^3/s].
        w_cmd_opt:  [N] optimised bead-width command [m].
        segment_len: iLQR segment length in original timesteps
            (default 450).
        dt_original: Original iLQR timestep [s] (default 0.01).
            Used with *segment_len* to compute the boundary time
            interval: ``T_seg = segment_len * dt_original``.
        pre_boundary_steps: Number of steps in *t_cmd_opt* before
            each boundary index to replace (default 1).
        post_boundary_steps: Number of steps in *t_cmd_opt* after
            each boundary index to replace (default 5).

    Returns:
        (Q_cmd_smoothed, w_cmd_smoothed) — same length as inputs.
        Originals are NOT modified (copies are returned).
    """
    N = len(Q_cmd_opt)
    Q_out = Q_cmd_opt.copy()
    w_out = w_cmd_opt.copy()

    T_seg = segment_len * dt_original  # boundary time interval [s]

    # Find boundary times: T_seg, 2*T_seg, ... up to max(t)
    t_max = t_cmd_opt[-1]
    boundary_times = np.arange(T_seg, t_max + T_seg * 0.5, T_seg)

    for t_boundary in boundary_times:
        # Index of the sample nearest to this boundary time
        b = int(np.argmin(np.abs(t_cmd_opt - t_boundary)))

        # Faulty range indices [lo, hi)
        lo = max(b - pre_boundary_steps, 0)
        hi = min(b + post_boundary_steps, N)
        n_faulty = hi - lo
        if n_faulty <= 0:
            continue

        # Anchor indices: last clean point before, first clean point after
        left_anchor = max(lo - 1, 0)
        right_anchor = min(hi, N - 1)
        if left_anchor == right_anchor:
            continue

        # Interpolate excluding endpoints so the anchors themselves stay
        # untouched and the replacement values sit strictly between them.
        for arr in (Q_out, w_out):
            interp_vals = np.linspace(
                arr[left_anchor], arr[right_anchor], n_faulty + 2
            )[1:-1]
            arr[lo:hi] = interp_vals

    return Q_out, w_out


def smooth_Q_cmd(
    Q_cmd: np.ndarray,
    window_length: int = 15,
    polyorder: int = 3,
) -> np.ndarray:
    """Lightly smooth Q_cmd to reduce high-acceleration transients.

    Applies a Savitzky-Golay filter which fits a local polynomial of
    degree *polyorder* over a sliding window of *window_length* samples.
    This preserves ramps and steady-state regions (up to the polynomial
    order) while attenuating sharp spikes that would demand unrealistic
    pump acceleration.

    Typical usage:
        Q_smooth = smooth_Q_cmd(Q_cmd, window_length=15, polyorder=3)

    Tuning guidance:
        - Larger *window_length* → more aggressive smoothing (must be odd
          and > polyorder).
        - Higher *polyorder* → preserves sharper features.  polyorder=3
          is a good default for piecewise-linear-ish commands.
        - For dt=0.01 s trajectories, window_length=15 smooths over a
          0.15 s neighbourhood.  For resampled (e.g. dt=0.1 s) data,
          reduce window_length proportionally.

    Args:
        Q_cmd:         [N] flowrate command array [m^3/s].
        window_length: Number of samples in the local fitting window.
                       Must be a positive odd integer > polyorder
                       (default 15).
        polyorder:     Polynomial order for the local fit (default 3).

    Returns:
        Q_cmd_smoothed — same length as input.  A copy; the original is
        NOT modified.
    """
    from scipy.signal import savgol_filter

    # Enforce odd window_length (S-G requirement)
    if window_length % 2 == 0:
        window_length += 1

    # Clamp window_length to array length (must be <= N and > polyorder)
    if window_length > len(Q_cmd):
        window_length = len(Q_cmd) if len(Q_cmd) % 2 == 1 else len(Q_cmd) - 1
    if window_length <= polyorder:
        # Not enough points to filter — return unchanged copy
        return Q_cmd.copy()

    return savgol_filter(Q_cmd, window_length, polyorder)