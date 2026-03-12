"""
run_ilqr_deploy.py — End-to-end iLQR FFEC deployment for arbitrary trajectories.

Takes any base trajectory .npz (not restricted to the test split), runs the full
iLQR pipeline, saves the optimized control arrays, and optionally validates them
via the original windowed prediction pipeline.

Usage::

    # Basic run — saves controls + figures to --out_dir
    python run_ilqr_deploy.py \\
        --traj_npz dataset/LSTM_sim_samples/corner_60100_1000.npz \\
        --G 1e15 --R_diag 1e-3 1e-3 \\
        --n_samples 450 --max_iter 50

    # With relative width bounds + windowed validation
    python run_ilqr_deploy.py \\
        --traj_npz dataset/LSTM_sim_samples/corner_60100_1000.npz \\
        --G 1e15 --R_diag 1e-3 1e-3 \\
        --n_samples 450 --max_iter 50 \\
        --w_delta_plus 2e-4 --w_delta_minus 4e-4 \\
        --validate --out_dir ilqr_deploy_results/

    # With JSON config file
    python run_ilqr_deploy.py \\
        --traj_npz dataset/LSTM_sim_samples/corner_60100_1000.npz \\
        --ilqr_config my_experiment.json --max_iter 50

Outputs (in --out_dir)::

    <traj_name>_controls_G<G>_R<R>.npz    ← deploy artifact: optimized Q_cmd, w_cmd
    <traj_name>_A_G<G>_R<R>.png           ← flowrate comparison figure
    <traj_name>_B_G<G>_R<R>.png           ← iLQR diagnostics figure
    <traj_name>_windowed_val_*.png         ← windowed validation (if --validate)

The controls .npz contains:
    t                    [N]  seconds  — time axis
    Q_cmd_opt            [N]  m³/s    — optimized pump flow command  (SEND TO HARDWARE)
    w_cmd_opt            [N]  m       — optimized width command       (SEND TO HARDWARE)
    Q_cmd_naive          [N]  m³/s    — naive command (= Q_com)
    w_cmd_naive          [N]  m       — naive width   (= W_com)
    Q_com                [N]  m³/s    — tracking target
    Q_out_stepmode_opt   [N]  m³/s    — hybrid model prediction under opt controls
    Q_out_stepmode_naive [N]  m³/s    — hybrid model prediction under naive controls
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ilqr_config import ILQRConfig, add_ilqr_args, load_config

# ── Shared display scale ─────────────────────────────────────────────────────
_SCALE    = 6e7        # m³/s → mL/min
_DT_TRAIN = 0.01       # training timestep [s]
_W_DEFAULT = 0.0029    # fallback bead width [m] if W_com absent in file


# ── Argument parsing ─────────────────────────────────────────────────────────

def _parse_args() -> tuple[ILQRConfig, argparse.Namespace]:
    """Parse CLI arguments with three-tier config precedence.

    Returns ``(cfg, args)`` where *cfg* is the merged :class:`ILQRConfig`
    and *args* holds script-specific values (``traj_npz``, ``dt_target``,
    ``out_dir``, ``validate``).
    """
    p = argparse.ArgumentParser(
        description="iLQR FFEC deployment for arbitrary trajectory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Script-specific args (NOT part of ILQRConfig) -------------------
    p.add_argument("--traj_npz", required=True,
                   help="Path to any base trajectory .npz "
                        "(keys: time, Q_com, Q_vbn, Q_res, and Q_sim/Q_exp/Q_tru; "
                        "optionally W_com).")
    p.add_argument("--dt_target", type=float, default=_DT_TRAIN,
                   help="Target timestep after downsampling [s].")
    p.add_argument("--out_dir", default="ilqr_deploy_results",
                   help="Output directory for controls .npz + figures.")
    p.add_argument("--validate", action="store_true",
                   help="After saving controls, run validate_ilqr_windowed.py "
                        "to check improvement via the original windowed pipeline.")
    p.add_argument("--ilqr_config", default=None,
                   help="Path to iLQR config JSON file. Overrides dataclass defaults; "
                        "any CLI flags further override the JSON values.")

    # -- Shared iLQR args (default=SUPPRESS; merged by load_config) ------
    add_ilqr_args(p)

    raw = p.parse_args()
    cfg = load_config(raw, ilqr_config_path=getattr(raw, "ilqr_config", None))
    return cfg, raw


# ── Checkpoint resolution ────────────────────────────────────────────────────

def _resolve_checkpoint(ckpt_cfg):
    """Return (ckpt_path, run_config) from CheckpointConfig."""
    if ckpt_cfg.ckpt_path is not None:
        ckpt_path = Path(ckpt_cfg.ckpt_path)
        if not ckpt_path.exists():
            sys.exit(f"Checkpoint not found: {ckpt_path}")
        if ckpt_cfg.config_path is not None:
            cfg_path = Path(ckpt_cfg.config_path)
        else:
            candidates = [Path("config.json"), ckpt_path.parent.parent / "config.json"]
            cfg_path = next((c for c in candidates if c.exists()), None)
            if cfg_path is None:
                sys.exit("--ckpt_path given without --config_path; no config.json found.")
        with open(cfg_path) as f:
            cfg = json.load(f)
        if "huber_delta" not in cfg:
            cfg["huber_delta"] = 1.0
        from models.traj_WALR import DictToObject
        return ckpt_path, DictToObject(cfg)
    else:
        print(f"  Resolving best checkpoint from WandB sweep '{ckpt_cfg.sweep_id}'...")
        from models.traj_WALR import get_best_run
        run_id, run_config = get_best_run(sweep_id=ckpt_cfg.sweep_id)
        ckpt_dir = Path("VBN-modeling") / run_id / "checkpoints"
        ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpt_files:
            sys.exit(f"No .ckpt files found in {ckpt_dir}")
        ckpt_path = ckpt_files[-1]
        print(f"  Checkpoint: {ckpt_path}")
        return ckpt_path, run_config


# ── Trajectory loading ────────────────────────────────────────────────────────

def _load_from_path(npz_path: Path, dt_target: float) -> dict:
    """
    Load a base trajectory .npz from an explicit path, downsample to dt_target.

    Supports the same key conventions as test_ilqr_test_set._load_base():
      - Ground truth: Q_sim (simulated), Q_exp (experimental), or Q_tru (windowed)
      - Width: W_com (m); if absent, falls back to _W_DEFAULT

    Returns dict with keys: t, dt, N, Q_com, Q_vbn, Q_res, Q_tru, W_com
    All flowrates in m³/s; widths in m; time in s.
    """
    print(f"  Loading: {npz_path}")
    d = np.load(npz_path)

    for key in ("Q_sim", "Q_exp", "Q_tru"):
        if key in d:
            q_tru_raw = d[key].ravel().astype(np.float64)
            print(f"  Ground truth key: '{key}'")
            break
    else:
        raise KeyError(
            f"No ground-truth key (Q_sim / Q_exp / Q_tru) in {npz_path.name}. "
            f"Found: {list(d.keys())}"
        )

    time_raw  = d["time"].ravel().astype(np.float64)
    Q_com_raw = d["Q_com"].ravel().astype(np.float64)
    Q_vbn_raw = d["Q_vbn"].ravel().astype(np.float64)
    Q_res_raw = d["Q_res"].ravel().astype(np.float64)
    W_com_raw = (d["W_com"].ravel().astype(np.float64) if "W_com" in d
                 else np.full(len(time_raw), _W_DEFAULT, dtype=np.float64))

    dt_raw = float(np.median(np.diff(time_raw)))
    ds = max(1, int(round(dt_target / dt_raw)))
    dt = dt_raw * ds
    print(f"  Raw dt = {dt_raw:.5f} s  →  downsample ×{ds}  →  dt = {dt:.5f} s")

    sl = slice(None, None, ds)
    N = len(time_raw[sl])
    t = np.arange(N, dtype=np.float64) * dt

    print(f"  N = {N} samples  ({N * dt:.2f} s total)")
    print(f"  Q_com: [{Q_com_raw[sl].min() * _SCALE:.3f}, "
          f"{Q_com_raw[sl].max() * _SCALE:.3f}] mL/min")
    print(f"  W_com: [{W_com_raw[sl].min() * 1e3:.2f}, "
          f"{W_com_raw[sl].max() * 1e3:.2f}] mm")

    return dict(
        t=t, dt=dt, N=N,
        Q_com=Q_com_raw[sl].copy(),
        Q_vbn=Q_vbn_raw[sl].copy(),
        Q_res=Q_res_raw[sl].copy(),
        Q_tru=q_tru_raw[sl].copy(),
        W_com=W_com_raw[sl].copy(),
    )


# ── Separator ─────────────────────────────────────────────────────────────────

def _sep(title: str, width: int = 62) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import matplotlib.pyplot as plt
    import torch

    cfg, args = _parse_args()
    if cfg.cost.G_f is None:
        cfg.cost.G_f = cfg.cost.G

    traj_path = Path(args.traj_npz)
    if not traj_path.exists():
        sys.exit(f"Trajectory file not found: {traj_path}")

    parent_id = traj_path.stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"G{cfg.cost.G:.0e}_R{cfg.cost.R_diag[0]:.0e}"

    # ── Load trajectory ───────────────────────────────────────────────────────
    _sep(f"Loading trajectory: {parent_id}")
    traj = _load_from_path(traj_path, dt_target=args.dt_target)
    t_full, dt  = traj["t"], traj["dt"]
    N_full      = traj["N"]
    Q_com       = traj["Q_com"]
    Q_vbn       = traj["Q_vbn"]
    Q_res       = traj["Q_res"]
    Q_tru       = traj["Q_tru"]
    W_com       = traj["W_com"]

    N_ilqr = min(cfg.run.n_samples, N_full)
    t_ilqr = t_full[:N_ilqr]
    print(f"\n  iLQR horizon: {N_ilqr} steps ({N_ilqr * dt:.2f} s)")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    _sep("Checkpoint resolution")
    ckpt_path, run_config = _resolve_checkpoint(cfg.checkpoint)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  num_layers={run_config.num_layers}  hidden_size={run_config.hidden_size}")

    # ── Initialise MATLAB bridge + LSTM ───────────────────────────────────────
    _sep("Initialising pipeline components")
    from matlab_bridge import MatlabBridge
    bridge = MatlabBridge(fluid=cfg.physics.fluid, mixer=cfg.physics.mixer,
                          pump=cfg.physics.pump)
    bridge.start()

    from lstm_step import LSTMStepWrapper
    lstm = LSTMStepWrapper(
        ckpt_path=ckpt_path,
        run_config=run_config,
        train_list_path=cfg.data.train_list,
        data_folder=cfg.data.data_folder,
        device=cfg.display.device,
    )

    # ── Windowed LSTM reference (full trajectory) ─────────────────────────────
    from test_ilqr_test_set import _run_windowed_lstm
    print(f"\n  Running windowed LSTM reference on full trajectory ({N_full} samples)…")
    Q_pred_windowed_full = _run_windowed_lstm(
        Q_com=Q_com, Q_vbn=Q_vbn, W_com=W_com,
        lstm=lstm, N=N_full, dt=dt,
        window_len_s=cfg.run.window_len_s, window_step_s=cfg.run.window_step_s,
    )

    # ── Hybrid dynamics naive rollout ─────────────────────────────────────────
    from dynamics import HybridDynamics
    dyn    = HybridDynamics(bridge=bridge, lstm=lstm, dt=dt)
    state0 = dyn.make_initial_state(t0=float(t_full[0]))

    U_naive = np.stack([Q_com[:N_ilqr], W_com[:N_ilqr]], axis=1)
    print(f"\n  Rolling out HybridDynamics (naive) for {N_ilqr} steps …")
    states_naive, Q_out_naive = dyn.rollout(state0, U_naive)

    rmse_naive = np.sqrt(np.mean((Q_out_naive - Q_com[:N_ilqr]) ** 2)) * _SCALE
    print(f"  Naive step-mode RMSE vs Q_com: {rmse_naive:.4f} mL/min")

    # ── Build control bounds ──────────────────────────────────────────────────
    bnd = cfg.bounds
    _use_rel = bnd.w_delta_plus is not None or bnd.w_delta_minus is not None
    if _use_rel:
        dp = bnd.w_delta_plus  if bnd.w_delta_plus  is not None else np.inf
        dm = bnd.w_delta_minus if bnd.w_delta_minus is not None else np.inf
        w_lo = np.maximum(bnd.w_min, W_com[:N_ilqr] - dm)
        w_hi = np.minimum(bnd.w_max, W_com[:N_ilqr] + dp)
        u_min = np.column_stack([np.full(N_ilqr, bnd.Q_min), w_lo])  # [N, 2]
        u_max = np.column_stack([np.full(N_ilqr, bnd.Q_max), w_hi])  # [N, 2]
        print(f"  Width bounds: ±[+{dp*1e3:.2f}, -{dm*1e3:.2f}] mm around W_com")
    else:
        u_min = np.array([bnd.Q_min, bnd.w_min])
        u_max = np.array([bnd.Q_max, bnd.w_max])

    # ── iLQR solve ────────────────────────────────────────────────────────────
    _sep(f"iLQR solve — {parent_id}")
    print(f"  G={cfg.cost.G:.2e}  G_f={cfg.cost.G_f:.2e}  R=diag{cfg.cost.R_diag}")
    if cfg.cost.S_diag is not None:
        print(f"  S = diag{cfg.cost.S_diag}  (soft rate penalty)")
    print(f"  max_iter={cfg.solver.max_iter}  tol={cfg.solver.tol}")
    print(f"  Horizon: {N_ilqr} steps  (~{10 * N_ilqr} MATLAB calls/iter)")

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
        eps_ctrl=cfg.solver.eps_ctrl,
        eps_ctrl_rel=cfg.solver.eps_ctrl_rel,
        eps_ctrl_floor_Q=cfg.solver.eps_ctrl_floor_Q,
        eps_ctrl_floor_w=cfg.solver.eps_ctrl_floor_w,
        verbose=True,
    )
    U_opt, Q_out_iLQR, cost_hist = solver.solve(
        state0=state0,
        q_ref=q_ref,
        U_init=U_naive.copy(),
        u_min=u_min,
        u_max=u_max,
    )

    rmse_ilqr = np.sqrt(np.mean((Q_out_iLQR - q_ref) ** 2)) * _SCALE
    if len(cost_hist) >= 2:
        reduction = (cost_hist[0] - cost_hist[-1]) / abs(cost_hist[0]) * 100
        print(f"\n  iLQR RMSE vs Q_com: {rmse_ilqr:.4f} mL/min  "
              f"(cost reduction {reduction:.1f}%,  {len(cost_hist)} iters)")
    print(f"  Naive RMSE:  {rmse_naive:.4f}  →  iLQR RMSE:  {rmse_ilqr:.4f}  "
          f"({'✓ improved' if rmse_ilqr < rmse_naive else '⚠ no improvement'})")

    # ── Save controls .npz ────────────────────────────────────────────────────
    ctrl_path = out_dir / f"{parent_id}_controls_{tag}.npz"
    np.savez(
        ctrl_path,
        t                    = t_ilqr,
        Q_cmd_opt            = U_opt[:, 0],
        w_cmd_opt            = U_opt[:, 1],
        Q_cmd_naive          = U_naive[:, 0],
        w_cmd_naive          = U_naive[:, 1],
        Q_com                = Q_com[:N_ilqr],
        Q_out_stepmode_opt   = Q_out_iLQR,
        Q_out_stepmode_naive = Q_out_naive,
    )
    print(f"\n  ✓ Controls saved: {ctrl_path}")

    # ── Quit MATLAB bridge ────────────────────────────────────────────────────
    bridge.quit()

    # ── Figures ───────────────────────────────────────────────────────────────
    from test_ilqr_test_set import _plot_flowrate_comparison, _plot_ilqr_diagnostics

    # Step-mode residual breakdown (for diagnostics panel)
    from constants.pump_viscotec_outdated import EXTRUSION_RATIO as ER
    Q_anal_stepmode = np.array(
        [float(states_naive[k].theta_dot) * ER for k in range(N_ilqr)],
        dtype=np.float64,
    )
    Q_res_stepmode = Q_out_naive - Q_anal_stepmode
    Q_res_windowed = Q_pred_windowed_full[:N_ilqr] - Q_vbn[:N_ilqr]
    rmse_win = (np.sqrt(np.mean(
        (Q_pred_windowed_full[:N_ilqr] - Q_com[:N_ilqr]) ** 2)) * _SCALE)

    fig_a = _plot_flowrate_comparison(
        parent_id=parent_id,
        t_full=t_full,
        Q_com=Q_com,
        Q_vbn=Q_vbn,
        Q_tru=Q_tru,
        W_com=W_com,
        Q_pred_windowed=Q_pred_windowed_full,
        t_ilqr=t_ilqr,
        Q_out_naive=Q_out_naive,
        Q_out_iLQR=Q_out_iLQR,
        rmse_windowed=rmse_win,
        rmse_naive=rmse_naive,
        rmse_iLQR=rmse_ilqr,
    )
    fig_b = _plot_ilqr_diagnostics(
        parent_id=parent_id,
        t_ilqr=t_ilqr,
        Q_res_npz=Q_res[:N_ilqr],
        Q_res_stepmode=Q_res_stepmode,
        Q_res_windowed=Q_res_windowed,
        Q_cmd_naive=U_naive[:, 0],
        Q_cmd_opt=U_opt[:, 0],
        W_cmd_naive=U_naive[:, 1],
        W_cmd_opt=U_opt[:, 1],
        cost_hist=cost_hist,
    )

    path_a = out_dir / f"{parent_id}_A_{tag}.png"
    path_b = out_dir / f"{parent_id}_B_{tag}.png"
    fig_a.savefig(path_a, dpi=150, bbox_inches="tight")
    fig_b.savefig(path_b, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path_a.name}")
    print(f"  Saved: {path_b.name}")

    if not cfg.display.no_show:
        plt.show()
    else:
        plt.close("all")

    # ── Optional windowed validation ──────────────────────────────────────────
    if args.validate:
        _sep("Running windowed pipeline validation")
        val_cmd = [
            sys.executable, "validate_ilqr_windowed.py",
            str(ctrl_path),
            "--save_dir", str(out_dir),
            "--no_show",
            "--device", cfg.display.device,
        ]
        if cfg.checkpoint.ckpt_path:
            val_cmd += ["--ckpt_path", str(ckpt_path)]
            if cfg.checkpoint.config_path:
                val_cmd += ["--config_path", cfg.checkpoint.config_path]
        else:
            val_cmd += ["--sweep_id", cfg.checkpoint.sweep_id]
        val_cmd += ["--data_folder", cfg.data.data_folder]

        print("  CMD:", " ".join(val_cmd))
        subprocess.run(val_cmd, check=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    _sep("Deploy summary")
    print(f"  Trajectory  : {parent_id}")
    print(f"  Horizon     : {N_ilqr} steps  ({N_ilqr * dt:.2f} s)")
    print(f"  G={cfg.cost.G:.1e}  R={cfg.cost.R_diag[0]:.1e}  iters={len(cost_hist)}")
    print(f"  Naive RMSE  : {rmse_naive:.4f} mL/min")
    print(f"  iLQR  RMSE  : {rmse_ilqr:.4f} mL/min")
    print(f"\n  Controls saved to: {ctrl_path.resolve()}")
    print(f"  Figures  saved to: {out_dir.resolve()}/")
    print("\n  Done.")


if __name__ == "__main__":
    main()
