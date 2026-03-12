"""
run_ffec.py — Top-level CLI script for iLQR FFEC (feedforward error compensation).

Given a reference flowrate trajectory q_ref(t), finds optimal controls
[Q_cmd*(t), w_cmd*(t)] that minimise tracking error through the hybrid
ODE+LSTM model using iLQR with finite-difference + autograd linearization.

Usage::

    python run_ffec.py \\
        --ckpt_path   VBN-modeling/<run_id>/checkpoints/epoch=X.ckpt \\
        --config_path config.json \\
        --input_path  dataset/LSTM_sim_samples/corner_4065_1000.npz \\
        --output_path outputs/ffec_result.npz \\
        --G 1e18 --R_diag 1e-3 1e-3 --G_f 1e18 \\
        --dt 0.01 --max_iter 30 \\
        --w_nom 0.0029 \\
        --device cpu

config.json should be a JSON file containing run_config fields:
  {"hidden_size": 128, "num_layers": 3, "lr": 1e-3, "huber_delta": 1.0}

Tip: to get checkpoint path and config from a WandB sweep, run::

    from models.traj_WALR import get_best_run
    run_id, run_config = get_best_run(sweep_id='mnywg829')
    # checkpoint: VBN-modeling/<run_id>/checkpoints/epoch=*.ckpt

Output .npz contains:
  t            [s]    time vector
  q_ref        [m³/s] reference trajectory
  Q_cmd_init   [m³/s] initial (nominal) commanded flowrate
  Q_cmd_opt    [m³/s] optimised commanded flowrate
  w_cmd_opt    [m]    optimised bead width command
  Q_out_init   [m³/s] model output with initial controls
  Q_out_opt    [m³/s] model output with optimised controls
  cost_hist    []     cost per iLQR iteration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ilqr_config import ILQRConfig, add_ilqr_args, load_config


# ======================================================================
# Argument parsing
# ======================================================================

def _parse_args() -> tuple[ILQRConfig, argparse.Namespace]:
    """Parse CLI arguments with three-tier config precedence.

    Returns ``(cfg, args)`` where *cfg* is the merged :class:`ILQRConfig`
    and *args* holds script-specific values (``input_path``, ``output_path``,
    ``plot``).
    """
    p = argparse.ArgumentParser(
        description="iLQR FFEC: find optimal [Q_cmd, w_cmd] to track q_ref"
    )

    # -- Script-specific args (NOT part of ILQRConfig) -------------------
    p.add_argument("--input_path", required=True, type=str,
                   help="Path to .npz file containing the reference trajectory. "
                        "Must contain Q_com (used as q_ref) and time vector.")
    p.add_argument("--output_path", default="outputs/ffec_result.npz", type=str,
                   help="Output .npz path (will confirm before writing)")
    p.add_argument("--plot", action="store_true",
                   help="Show matplotlib plot after optimisation")
    p.add_argument("--ilqr_config", default=None,
                   help="Path to iLQR config JSON file. Overrides dataclass defaults; "
                        "any CLI flags further override the JSON values.")

    # -- Shared iLQR args (default=SUPPRESS; merged by load_config) ------
    add_ilqr_args(p)

    raw = p.parse_args()
    cfg = load_config(raw, ilqr_config_path=getattr(raw, "ilqr_config", None))
    return cfg, raw


# ======================================================================
# Helper: resolve checkpoint from CheckpointConfig
# ======================================================================

def _resolve_checkpoint(ckpt_cfg):
    """Return (ckpt_path, run_config) from CheckpointConfig."""
    from ilqr_config import CheckpointConfig
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
        ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt"))
        if not ckpts:
            sys.exit(f"No checkpoints found in {ckpt_dir}")
        ckpt_path = ckpts[-1]
        print(f"  Using checkpoint: {ckpt_path}")
        return ckpt_path, run_config


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    cfg, args = _parse_args()

    # ---- Resolve checkpoint
    ckpt_path, run_config = _resolve_checkpoint(cfg.checkpoint)

    # ---- Load input trajectory
    data = np.load(args.input_path)
    if "Q_com" not in data:
        sys.exit(f"Input .npz must contain 'Q_com'. Keys found: {list(data.keys())}")

    q_ref = data["Q_com"].ravel().astype(np.float64)  # [m³/s] — use Q_com as reference
    T = len(q_ref)
    dt = cfg.physics.dt
    t_vec = np.arange(T) * dt

    print(f"Input trajectory: T={T} timesteps, dt={dt}s")
    print(f"q_ref range: [{q_ref.min():.3e}, {q_ref.max():.3e}] m³/s")

    # ---- Initial control guess: Q_cmd = q_ref, w_cmd = nominal
    Q_cmd_init = q_ref.copy()
    w_cmd_init = np.full(T, cfg.physics.w_nom)
    U_init = np.stack([Q_cmd_init, w_cmd_init], axis=1)  # [T, 2]

    # ---- Cost matrices
    R = np.diag(cfg.cost.R_diag)

    # ---- Control bounds
    u_min = np.array([cfg.bounds.Q_min, cfg.bounds.w_min])
    u_max = np.array([cfg.bounds.Q_max, cfg.bounds.w_max])

    # ---- Initialise dynamics (this starts MATLAB engine)
    print("\nInitialising MatlabBridge...")
    from matlab_bridge import MatlabBridge
    bridge = MatlabBridge(
        fluid=cfg.physics.fluid,
        mixer=cfg.physics.mixer,
        pump=cfg.physics.pump,
    )

    print("Initialising LSTMStepWrapper...")
    from lstm_step import LSTMStepWrapper
    lstm = LSTMStepWrapper(
        ckpt_path=ckpt_path,
        run_config=run_config,
        train_list_path=cfg.data.train_list,
        data_folder=cfg.data.data_folder,
        device=cfg.display.device,
    )

    from dynamics import HybridDynamics
    dyn = HybridDynamics(bridge=bridge, lstm=lstm, dt=dt)

    # ---- Initial rollout (baseline)
    print("\nRunning initial rollout...")
    state0 = dyn.make_initial_state(t0=0.0)
    _, Q_out_init = dyn.rollout(state0, U_init)
    cost_init = sum(
        cfg.cost.G * (Q_out_init[k] - q_ref[k]) ** 2
        + U_init[k] @ R @ U_init[k]
        for k in range(T)
    )
    print(f"Initial cost: {cost_init:.6e}")
    print(f"Initial tracking RMSE: {np.sqrt(np.mean((Q_out_init - q_ref)**2)):.3e} m³/s")

    # ---- Run iLQR
    print("\nRunning iLQR optimisation...")
    from ilqr import ILQRSolver
    solver = ILQRSolver(
        dynamics=dyn,
        G=cfg.cost.G,
        R=R,
        G_f=cfg.cost.G_f,
        max_iter=cfg.solver.max_iter,
        tol=cfg.solver.tol,
        eps_ode=cfg.solver.eps_ode,
        eps_ctrl=cfg.solver.eps_ctrl,
        eps_ctrl_rel=cfg.solver.eps_ctrl_rel,
        eps_ctrl_floor_Q=cfg.solver.eps_ctrl_floor_Q,
        eps_ctrl_floor_w=cfg.solver.eps_ctrl_floor_w,
        verbose=True,
    )

    U_opt, Q_out_opt, cost_hist = solver.solve(
        state0=state0,
        q_ref=q_ref,
        U_init=U_init,
        u_min=u_min,
        u_max=u_max,
    )

    print(f"\nFinal cost:     {cost_hist[-1]:.6e}")
    print(f"Tracking RMSE (opt):  {np.sqrt(np.mean((Q_out_opt - q_ref)**2)):.3e} m³/s")
    print(f"Tracking RMSE (init): {np.sqrt(np.mean((Q_out_init - q_ref)**2)):.3e} m³/s")

    # ---- Save results (confirm before writing)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nAbout to write results to: {out_path}")
    confirm = input("Confirm write? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Write cancelled.")
    else:
        np.savez(
            out_path,
            t=t_vec,
            q_ref=q_ref,
            Q_cmd_init=Q_cmd_init,
            w_cmd_init=w_cmd_init,
            Q_cmd_opt=U_opt[:, 0],
            w_cmd_opt=U_opt[:, 1],
            Q_out_init=Q_out_init,
            Q_out_opt=Q_out_opt,
            cost_hist=np.array(cost_hist),
        )
        print(f"Saved to {out_path}")

    # ---- Optional plot
    if args.plot:
        _plot_results(t_vec, q_ref, Q_out_init, Q_out_opt, U_opt, cost_hist)

    # ---- Cleanup
    bridge.quit()
    print("Done.")


# ======================================================================
# Plotting
# ======================================================================

def _plot_results(
    t: np.ndarray,
    q_ref: np.ndarray,
    Q_out_init: np.ndarray,
    Q_out_opt: np.ndarray,
    U_opt: np.ndarray,
    cost_hist: list[float],
) -> None:
    import matplotlib.pyplot as plt

    scale = 6e7  # m³/s → mL/min

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Flowrate tracking
    ax = axes[0]
    ax.plot(t, q_ref * scale, "k--", linewidth=2, label="q_ref")
    ax.plot(t, Q_out_init * scale, "b", linewidth=1.5, alpha=0.6, label="q_out (init)")
    ax.plot(t, Q_out_opt * scale, "r", linewidth=2, label="q_out (opt)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Flowrate [mL/min]")
    ax.set_title("iLQR FFEC: Flowrate Tracking")
    ax.legend()
    ax.grid(True, linewidth=0.5)

    # Optimal controls
    ax = axes[1]
    ax.plot(t, U_opt[:, 0] * scale, "r", linewidth=2, label="Q_cmd* [mL/min]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Q_cmd [mL/min]")
    ax.set_title("Optimal Commanded Flowrate")
    ax.legend()
    ax.grid(True, linewidth=0.5)

    ax2 = ax.twinx()
    ax2.plot(t, U_opt[:, 1] * 1000, "g--", linewidth=1.5, label="w_cmd* [mm]")
    ax2.set_ylabel("w_cmd [mm]")
    ax2.legend(loc="lower right")

    # Cost history
    ax = axes[2]
    ax.semilogy(range(len(cost_hist)), cost_hist, "k-o", markersize=4)
    ax.set_xlabel("iLQR Iteration")
    ax.set_ylabel("Total Cost")
    ax.set_title("iLQR Cost Convergence")
    ax.grid(True, linewidth=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
