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


# ======================================================================
# Argument parsing
# ======================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="iLQR FFEC: find optimal [Q_cmd, w_cmd] to track q_ref"
    )

    # Model
    p.add_argument("--ckpt_path", required=True, type=str,
                   help="Path to .ckpt checkpoint file")
    p.add_argument("--config_path", required=True, type=str,
                   help="Path to JSON file with run_config fields "
                        "(hidden_size, num_layers, lr, huber_delta)")

    # Data
    p.add_argument("--input_path", required=True, type=str,
                   help="Path to .npz file containing the reference trajectory. "
                        "Must contain Q_com (used as q_ref) and time vector.")
    p.add_argument("--train_list", default="splits/train.txt", type=str,
                   help="Path to train split .txt file (for norm stats)")
    p.add_argument("--data_folder", default="dataset/recursive_samples", type=str,
                   help="Folder containing training .npz files "
                        "(must contain all files listed in splits/train.txt)")

    # Output
    p.add_argument("--output_path", default="outputs/ffec_result.npz", type=str,
                   help="Output .npz path (will confirm before writing)")

    # Cost weights
    p.add_argument("--G", type=float, default=1e18,
                   help="Tracking cost weight G (scalar)")
    p.add_argument("--R_diag", nargs=2, type=float, default=[1e-3, 1e-3],
                   metavar=("R_Q", "R_w"),
                   help="Diagonal of R matrix [R_Q_cmd, R_w_cmd]")
    p.add_argument("--G_f", type=float, default=1e18,
                   help="Terminal tracking cost weight")

    # Dynamics
    p.add_argument("--dt", type=float, default=0.01,
                   help="Timestep [s] (should match data sampling rate)")
    p.add_argument("--w_nom", type=float, default=0.0029,
                   help="Nominal bead width [m] for initial control guess")
    p.add_argument("--fluid", default="fluid_DOW121")
    p.add_argument("--mixer", default="mixer_ISSM50nozzle")
    p.add_argument("--pump", default="pump_viscotec_outdated")

    # Control bounds
    p.add_argument("--Q_min", type=float, default=0.0,
                   help="Minimum Q_cmd [m³/s]")
    p.add_argument("--Q_max", type=float, default=1e-6,
                   help="Maximum Q_cmd [m³/s]")
    p.add_argument("--w_min", type=float, default=0.0005,
                   help="Minimum w_cmd [m]")
    p.add_argument("--w_max", type=float, default=0.005,
                   help="Maximum w_cmd [m]")

    # iLQR
    p.add_argument("--max_iter", type=int, default=30,
                   help="Maximum iLQR iterations")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Relative cost tolerance for convergence")

    # FD perturbation
    p.add_argument("--eps_ode", type=float, default=1e-5,
                   help="FD perturbation for ODE state dims")
    p.add_argument("--eps_ctrl", type=float, default=1e-7,
                   help="FD perturbation for control dims")

    # Misc
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="PyTorch device for LSTM")
    p.add_argument("--plot", action="store_true",
                   help="Show matplotlib plot after optimisation")

    return p.parse_args()


# ======================================================================
# Helper: load config from JSON
# ======================================================================

class _DictObj:
    """Convert a dict to an attribute-access object (mirrors DictToObject in traj_WALR.py)."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = _parse_args()

    # ---- Load run_config
    with open(args.config_path) as f:
        cfg_dict = json.load(f)
    # Ensure required fields present
    for field in ("hidden_size", "num_layers", "lr"):
        if field not in cfg_dict:
            sys.exit(f"config_path JSON is missing required field: '{field}'")
    if "huber_delta" not in cfg_dict:
        cfg_dict["huber_delta"] = 1.0
    run_config = _DictObj(cfg_dict)

    # ---- Load input trajectory
    data = np.load(args.input_path)
    if "Q_com" not in data:
        sys.exit(f"Input .npz must contain 'Q_com'. Keys found: {list(data.keys())}")

    q_ref = data["Q_com"].ravel().astype(np.float64)  # [m³/s] — use Q_com as reference
    T = len(q_ref)
    dt = args.dt
    t_vec = np.arange(T) * dt

    print(f"Input trajectory: T={T} timesteps, dt={dt}s")
    print(f"q_ref range: [{q_ref.min():.3e}, {q_ref.max():.3e}] m³/s")

    # ---- Initial control guess: Q_cmd = q_ref, w_cmd = nominal
    Q_cmd_init = q_ref.copy()
    w_cmd_init = np.full(T, args.w_nom)
    U_init = np.stack([Q_cmd_init, w_cmd_init], axis=1)  # [T, 2]

    # ---- Cost matrices
    R = np.diag(args.R_diag)

    # ---- Control bounds
    u_min = np.array([args.Q_min, args.w_min])
    u_max = np.array([args.Q_max, args.w_max])

    # ---- Initialise dynamics (this starts MATLAB engine)
    print("\nInitialising MatlabBridge...")
    from matlab_bridge import MatlabBridge
    bridge = MatlabBridge(fluid=args.fluid, mixer=args.mixer, pump=args.pump)

    print("Initialising LSTMStepWrapper...")
    from lstm_step import LSTMStepWrapper
    lstm = LSTMStepWrapper(
        ckpt_path=args.ckpt_path,
        run_config=run_config,
        train_list_path=args.train_list,
        data_folder=args.data_folder,
        device=args.device,
    )

    from dynamics import HybridDynamics
    dyn = HybridDynamics(bridge=bridge, lstm=lstm, dt=dt)

    # ---- Initial rollout (baseline)
    print("\nRunning initial rollout...")
    state0 = dyn.make_initial_state(t0=0.0)
    _, Q_out_init = dyn.rollout(state0, U_init)
    cost_init = sum(
        args.G * (Q_out_init[k] - q_ref[k]) ** 2
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
        G=args.G,
        R=R,
        G_f=args.G_f,
        max_iter=args.max_iter,
        tol=args.tol,
        eps_ode=args.eps_ode,
        eps_ctrl=args.eps_ctrl,
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
