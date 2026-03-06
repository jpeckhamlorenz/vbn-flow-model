"""
test_ilqr_pipeline.py — End-to-end iLQR FFEC pipeline test with diagnostic plots.

Two sequential checks:

  1. Model Integrity Check
     Runs HybridDynamics.rollout() with Q_com as the naive control and compares
     the output against the original simulation data stored in the .npz file
     (Q_sim, Q_vbn, Q_res).  Confirms that the new step-mode MATLAB bridge +
     LSTM wrapper reproduce the same predictions as the training pipeline.

  2. iLQR Optimisation
     Runs ILQRSolver on the same trajectory with q_ref = Q_com and compares
     the optimised output against the naive (open-loop feedforward) baseline.

Usage (minimal — auto-resolves best checkpoint from WandB sweep)::

    python test_ilqr_pipeline.py

Usage (explicit checkpoint)::

    python test_ilqr_pipeline.py \\
        --ckpt_path  VBN-modeling/<run_id>/checkpoints/epoch=X.ckpt \\
        --config_path config.json \\
        --data_path  dataset/LSTM_sim_samples/corner_4065_1000.npz \\
        --n_samples  300 \\
        --max_iter   10

Runtime note:
  Each iLQR iteration linearizes T timesteps (10 MATLAB calls each) and may
  run several forward passes during line search.  With n_samples=300 and
  max_iter=10 expect ~5-20 min on a laptop.  Use --n_samples 100 for a
  quicker smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# m³/s → mL/min  (1 m³/s = 6e7 mL/min)
_SCALE: float = 6e7


# ======================================================================
# CLI
# ======================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end iLQR FFEC pipeline test with diagnostic plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument(
        "--data_path",
        default="dataset/LSTM_sim_samples/corner_4065_1000.npz",
        help="Path to .npz file with keys: time, Q_com, Q_vbn, Q_res, Q_sim",
    )
    p.add_argument(
        "--n_samples", type=int, default=300,
        help="Number of timesteps to use from the start of the trajectory. "
             "Lower = faster.  Default: 300 (~30 s at dt=0.1 s).",
    )

    # Checkpoint resolution
    grp = p.add_argument_group("Checkpoint (use --sweep_id OR --ckpt_path + --config_path)")
    grp.add_argument(
        "--sweep_id", default="mnywg829",
        help="WandB sweep ID passed to get_best_run() when --ckpt_path is absent. "
             "Default: mnywg829 (active sweep).",
    )
    grp.add_argument(
        "--ckpt_path", default=None,
        help="Explicit path to .ckpt checkpoint file.",
    )
    grp.add_argument(
        "--config_path", default=None,
        help="JSON config file (required with --ckpt_path). "
             "Fields: hidden_size, num_layers, lr [, huber_delta, batch_size].",
    )

    # Norm stats
    p.add_argument("--train_list", default="splits/train.txt",
                   help="Train split .txt for norm stats recomputation")
    p.add_argument("--data_folder", default="dataset/recursive_samples",
                   help="Folder containing training .npz files listed in splits/train.txt")

    # Dynamics
    p.add_argument("--w_nom", type=float, default=0.0029,
                   help="Nominal bead width [m].  Default: 0.0029 m (2.9 mm).")
    p.add_argument("--dt", type=float, default=None,
                   help="Timestep [s].  Inferred from time vector if not given.")

    # Windowed inference (integrity check reference)
    p.add_argument("--window_len_s", type=float, default=4.5,
                   help="Window length [s] for the windowed LSTM reference in the "
                        "integrity check (matches WindowParams default).  Default: 4.5 s.")
    p.add_argument("--window_step_s", type=float, default=0.1,
                   help="Window step [s] for the windowed LSTM reference.  "
                        "Default: 0.1 s (dense overlap).")
    p.add_argument("--fluid",  default="fluid_DOW121")
    p.add_argument("--mixer",  default="mixer_ISSM50nozzle")
    p.add_argument("--pump",   default="pump_viscotec_outdated")

    # iLQR cost
    p.add_argument("--G",    type=float, default=1e14,
                   help="Tracking cost weight G (default 1e14)")
    p.add_argument("--G_f",  type=float, default=None,
                   help="Terminal cost weight (default: same as --G)")
    p.add_argument("--R_diag", nargs=2, type=float, default=[1e-3, 1e-3],
                   metavar=("R_Q", "R_w"),
                   help="Diagonal of R matrix [R_Q_cmd, R_w_cmd]")

    # iLQR solver
    p.add_argument("--max_iter", type=int, default=10,
                   help="Max iLQR iterations (default 10 for a test run)")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Relative cost convergence tolerance")
    p.add_argument("--eps_ode",  type=float, default=1e-5)
    p.add_argument("--eps_ctrl", type=float, default=1e-7)

    # Control bounds
    p.add_argument("--Q_min", type=float, default=0.0)
    p.add_argument("--Q_max", type=float, default=1e-6)
    p.add_argument("--w_min", type=float, default=0.0005)
    p.add_argument("--w_max", type=float, default=0.005)

    # Misc
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--no_show", action="store_true",
                   help="Skip plt.show() (useful for non-interactive runs)")

    return p.parse_args()


# ======================================================================
# Helpers
# ======================================================================

def _resolve_checkpoint(args: argparse.Namespace):
    """
    Return (ckpt_path: Path, run_config) from either:
      - explicit --ckpt_path + --config_path, or
      - WandB get_best_run(--sweep_id)
    """
    if args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            sys.exit(f"Checkpoint not found: {ckpt_path}")

        if args.config_path is not None:
            cfg_path = Path(args.config_path)
        else:
            # Fall back to config.json in run directory or project root
            candidates = [
                Path("config.json"),
                ckpt_path.parent.parent / "config.json",
            ]
            cfg_path = next((c for c in candidates if c.exists()), None)
            if cfg_path is None:
                sys.exit(
                    "--ckpt_path supplied without --config_path and no config.json found.\n"
                    "Please supply --config_path."
                )
            print(f"  Config: {cfg_path}")

        with open(cfg_path) as f:
            cfg = json.load(f)
        if "huber_delta" not in cfg:
            cfg["huber_delta"] = 1.0

        from models.traj_WALR import DictToObject
        return ckpt_path, DictToObject(cfg)

    else:
        print(f"  Resolving best checkpoint from WandB sweep '{args.sweep_id}'...")
        from models.traj_WALR import get_best_run
        run_id, run_config = get_best_run(sweep_id=args.sweep_id)
        ckpt_dir = Path("VBN-modeling") / run_id / "checkpoints"
        ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpt_files:
            sys.exit(f"No .ckpt files found in {ckpt_dir}")
        ckpt_path = ckpt_files[-1]
        print(f"  Checkpoint: {ckpt_path}")
        return ckpt_path, run_config


def _sep(title: str, width: int = 62) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ======================================================================
# Windowed LSTM reference (replicates training-time inference paradigm)
# ======================================================================

@torch.no_grad()
def _run_windowed_lstm(
    Q_com: np.ndarray,
    Q_vbn: np.ndarray,
    w_nom_m: float,
    lstm,               # LSTMStepWrapper
    N: int,
    dt: float,
    window_len_s: float = 4.5,
    window_step_s: float = 0.1,
) -> np.ndarray:
    """
    Replicate the original windowed LSTM inference paradigm used during
    training and in flow_predictor_lstm_windowed(), using the already-loaded
    LSTMStepWrapper.  No second MATLAB call is needed — Q_vbn comes from the
    .npz file.

    For each window the hidden state is reset to zero (matching the training
    convention). Overlapping window predictions are averaged.

    Args:
        Q_com:         commanded flowrate [m³/s], shape (N,)
        Q_vbn:         analytical ODE flowrate [m³/s], shape (N,) — from .npz
        w_nom_m:       nominal bead width [m] (uniform across all timesteps)
        lstm:          LSTMStepWrapper (model already loaded and in eval mode)
        N:             number of timesteps to cover
        dt:            timestep [s]
        window_len_s:  window length [s]   (default 4.5 s — matches WindowParams)
        window_step_s: window step [s]     (default 0.1 s — matches WindowParams)

    Returns:
        Q_out_windowed: Q_vbn + averaged_residual  [m³/s], shape (N,)
    """
    import torch
    from lstm_step import _FLOW_SCALE, _BEAD_SCALE, _NORM_EPS

    window_len  = max(1, int(round(window_len_s  / dt)))
    window_step = max(1, int(round(window_step_s / dt)))

    # Precompute raw feature matrix [N, 3] in the same physical-scaled units
    # used by LSTMStepWrapper.step()
    w_arr = np.full(N, w_nom_m, dtype=np.float64)
    feats_raw = np.stack([
        Q_com * _FLOW_SCALE,    # command  [m³/s → normalised units]
        w_arr * _BEAD_SCALE,    # bead     [m → mm]
        Q_vbn * _FLOW_SCALE,    # analytical [m³/s → normalised units]
    ], axis=1)  # [N, 3]

    # Normalise using the same per-channel stats as LSTMStepWrapper.step()
    in_mu = lstm._in_mu.cpu().numpy()   # [3]
    in_sd = lstm._in_sd.cpu().numpy()   # [3]
    feats_norm = (feats_raw - in_mu[None, :]) / (in_sd[None, :] + _NORM_EPS)

    tgt_mu = lstm._tgt_mu
    tgt_sd = lstm._tgt_sd

    # Accumulate residuals from every overlapping window, then average
    acc   = np.zeros(N, dtype=np.float64)
    count = np.zeros(N, dtype=np.float64)

    for s in range(0, N, window_step):
        e = min(s + window_len, N)
        win_len = e - s
        if win_len == 0:
            break

        # Input tensor [1, win_len, 3]  (batch=1, seq_len, features)
        # WalrLSTM.lstm is batch_first=True, so batch dim comes first.
        x_win = torch.tensor(
            feats_norm[s:e, :],
            dtype=torch.float32,
            device=lstm.device,
        ).unsqueeze(0)  # [1, win_len, 3]

        # Fresh h, c per window — THIS is the key difference from step-mode
        h0, c0 = lstm.init_state()

        # LSTM + FC forward
        out, _ = lstm._net.lstm(x_win, (h0, c0))   # [1, win_len, hidden_size]
        y_hat_norm = lstm._net.fc(out).squeeze(0).squeeze(-1)  # [win_len]

        # Denormalise → physical residual [m³/s]
        res = (
            y_hat_norm.cpu().numpy().astype(np.float64)
            * (tgt_sd + _NORM_EPS)
            + tgt_mu
        ) / _FLOW_SCALE

        acc[s:e]   += res
        count[s:e] += 1.0

    # Divide by overlap count; leave zeros where no window covered (shouldn't happen)
    valid = count > 0
    avg_res = np.zeros(N, dtype=np.float64)
    avg_res[valid] = acc[valid] / count[valid]

    return Q_vbn + avg_res


# ======================================================================
# Plot 1 — Model Integrity Check
# ======================================================================

def _plot_integrity(
    t: np.ndarray,
    Q_com: np.ndarray,
    Q_vbn_npz: np.ndarray,
    Q_sim_npz: np.ndarray,
    Q_res_npz: np.ndarray,
    Q_out_hybrid: np.ndarray,
    Q_analytical_hybrid: np.ndarray,
    Q_res_predicted: np.ndarray,
    Q_out_windowed: np.ndarray | None = None,
) -> plt.Figure:
    """
    Two-panel figure confirming the new hybrid pipeline is intact.

    Panel 1 (flowrate):
      Q_com          [black dashed] — commanded flowrate (model input)
      Q_sim_npz      [red]          — ground-truth simulation from .npz (reference)
      Q_vbn_npz      [green]        — analytical ODE from .npz
      Q_out_windowed [purple]       — windowed LSTM (original training paradigm) — optional
      Q_out_hybrid   [blue]         — our new HybridDynamics.rollout() output

      ✓ Purple ≈ Red  →  windowed LSTM matches ground truth (sanity check on model quality)
      ✓ Blue ≈ Purple for first ~window_len steps, then may diverge (h,c threading mismatch)

    Panel 2 (residuals):
      Q_res_npz       [red]  — residual stored in .npz  (LSTM training target)
      Q_res_predicted [blue] — residual predicted by step-mode LSTMStepWrapper

      ✓ Blue ≈ Red  →  LSTM step-mode matches training-time batch-mode
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        "Model Integrity Check\n"
        "Confirms the new step-mode pipeline produces the same output as the original training pipeline",
        fontsize=12, fontweight="bold",
    )

    # ---- Panel 1: Flowrates ------------------------------------------------
    ax = axes[0]
    S = _SCALE
    ax.plot(t, Q_com          * S, color="black",   ls="--", lw=1.5, alpha=0.7,
            label="Q_com (commanded, model input)")
    ax.plot(t, Q_sim_npz      * S, color="red",     ls="-",  lw=2.0,
            label="Q_sim (ground-truth from .npz)  ← reference")
    ax.plot(t, Q_vbn_npz      * S, color="green",   ls="-",  lw=1.5,
            label="Q_vbn (analytical ODE from .npz)")
    if Q_out_windowed is not None:
        rmse_windowed = np.sqrt(np.mean((Q_out_windowed - Q_sim_npz) ** 2)) * S
        ax.plot(t, Q_out_windowed * S, color="purple", ls="-",  lw=1.5, alpha=0.85,
                label=f"Q_out windowed (original training paradigm)  RMSE {rmse_windowed:.4f}")
    ax.plot(t, Q_out_hybrid   * S, color="blue",    ls="-",  lw=1.5, alpha=0.85,
            label="Q_out hybrid (step-mode rollout)  ← new")

    rmse_hybrid      = np.sqrt(np.mean((Q_out_hybrid - Q_sim_npz) ** 2)) * S
    rmse_analytical  = np.sqrt(np.mean((Q_vbn_npz    - Q_sim_npz) ** 2)) * S
    ax.set_ylabel("Flowrate [mL/min]")
    title_parts = [
        f"hybrid RMSE: {rmse_hybrid:.4f} mL/min",
        f"analytical RMSE: {rmse_analytical:.4f} mL/min",
    ]
    if Q_out_windowed is not None:
        title_parts.insert(0, f"windowed RMSE: {rmse_windowed:.4f} mL/min")
    ax.set_title("Flowrates  —  " + "   |   ".join(title_parts), fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, color="0.75", lw=0.5)

    # ---- Panel 2: Residuals ------------------------------------------------
    ax = axes[1]
    ax.plot(t, Q_res_npz       * S, color="red",   ls="-",  lw=2.0,
            label="Q_res from .npz  (LSTM training target)")
    ax.plot(t, Q_res_predicted * S, color="blue",  ls="-",  lw=1.5, alpha=0.85,
            label="Q_res step-mode LSTM  ← new")

    # Also overlay the analytical-output agreement as a sanity check
    delta_anal = Q_analytical_hybrid - Q_vbn_npz
    ax_twin = ax.twinx()
    ax_twin.plot(t, delta_anal * S, color="grey", ls=":", lw=1.0, alpha=0.6,
                 label="Q_analytical hybrid − Q_vbn_npz  (should ≈ 0)")
    ax_twin.set_ylabel("Analytical delta [mL/min]", fontsize=8, color="grey")
    ax_twin.tick_params(axis="y", labelcolor="grey")
    ax_twin.legend(loc="lower right", fontsize=7)

    rmse_res = np.sqrt(np.mean((Q_res_predicted - Q_res_npz) ** 2)) * S
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Residual flowrate [mL/min]")
    ax.set_title(f"Residuals  —  step-mode LSTM vs .npz target RMSE: {rmse_res:.5f} mL/min",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, color="0.75", lw=0.5)

    fig.tight_layout()
    return fig


# ======================================================================
# Plot 2 — iLQR Results
# ======================================================================

def _plot_ilqr(
    t: np.ndarray,
    q_ref: np.ndarray,
    Q_out_naive: np.ndarray,
    Q_out_opt: np.ndarray,
    U_naive: np.ndarray,
    U_opt: np.ndarray,
    cost_hist: list[float],
) -> plt.Figure:
    """
    Three-panel figure showing iLQR optimisation results.

    Panel 1: Flowrate tracking — q_ref, naive model output, iLQR output
    Panel 2: Control signals  — Q_cmd and w_cmd (naive vs optimised)
    Panel 3: iLQR cost convergence (semilogy vs iteration)
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    S = _SCALE
    fig.suptitle("iLQR FFEC Optimisation Results", fontsize=12, fontweight="bold")

    rmse_naive = np.sqrt(np.mean((Q_out_naive - q_ref) ** 2)) * S
    rmse_opt   = np.sqrt(np.mean((Q_out_opt   - q_ref) ** 2)) * S

    # ---- Panel 1: Tracking ------------------------------------------------
    ax = axes[0]
    ax.plot(t, q_ref       * S, color="black",    ls="--", lw=1.5, alpha=0.8,
            label="q_ref (target)")
    ax.plot(t, Q_out_naive * S, color="royalblue", ls="-",  lw=1.5, alpha=0.7,
            label=f"Q_out naive  (RMSE {rmse_naive:.4f} mL/min)")
    ax.plot(t, Q_out_opt   * S, color="firebrick", ls="-",  lw=2.0,
            label=f"Q_out iLQR   (RMSE {rmse_opt:.4f} mL/min)")
    ax.set_ylabel("Flowrate [mL/min]")
    ax.set_title(
        f"Tracking Performance  —  naive RMSE: {rmse_naive:.4f} mL/min"
        f"   |   iLQR RMSE: {rmse_opt:.4f} mL/min",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, color="0.75", lw=0.5)
    ax.set_xlabel("Time [s]")

    # ---- Panel 2: Controls ------------------------------------------------
    ax = axes[1]
    ax.plot(t, U_naive[:, 0] * S, color="royalblue", ls="--", lw=1.5, alpha=0.7,
            label="Q_cmd naive")
    ax.plot(t, U_opt[:, 0]   * S, color="firebrick",  ls="-",  lw=1.5,
            label="Q_cmd iLQR")
    ax.set_ylabel("Q_cmd [mL/min]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, color="0.75", lw=0.5)
    ax.set_title("Optimised Controls", fontsize=9)
    ax.set_xlabel("Time [s]")

    ax2 = ax.twinx()
    ax2.plot(t, U_naive[:, 1] * 1000, color="grey",        ls="--", lw=1.0, alpha=0.5,
             label="w_cmd naive [mm]")
    ax2.plot(t, U_opt[:, 1]   * 1000, color="darkorange",  ls="-",  lw=1.5,
             label="w_cmd iLQR [mm]")
    ax2.set_ylabel("w_cmd [mm]")
    ax2.legend(loc="upper right", fontsize=8)

    # ---- Panel 3: Cost convergence ----------------------------------------
    ax = axes[2]
    iters = np.arange(len(cost_hist))
    ax.semilogy(iters, cost_hist, "k-o", markersize=5, lw=1.5)
    ax.set_xlabel("iLQR Iteration")
    ax.set_ylabel("Total Cost")
    ax.set_title("iLQR Cost Convergence", fontsize=9)
    ax.grid(True, which="both", color="0.75", lw=0.5)
    if len(iters) <= 20:
        ax.set_xticks(iters)

    fig.tight_layout()
    return fig


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = _parse_args()
    if args.G_f is None:
        args.G_f = args.G

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    _sep("Loading data")
    data_path = Path(args.data_path)
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}")

    d = np.load(data_path)
    required = {"time", "Q_com", "Q_vbn", "Q_res", "Q_sim"}
    missing  = required - set(d.keys())
    if missing:
        sys.exit(f"Data file missing keys: {missing}  (found: {list(d.keys())})")

    T_full = len(d["time"])
    N = min(args.n_samples, T_full)
    print(f"  File: {data_path.name}  ({T_full} samples total, using first {N})")

    time_raw  = d["time"].ravel()[:N].astype(np.float64)
    Q_com     = d["Q_com"].ravel()[:N].astype(np.float64)
    Q_vbn_npz = d["Q_vbn"].ravel()[:N].astype(np.float64)
    Q_res_npz = d["Q_res"].ravel()[:N].astype(np.float64)
    Q_sim_npz = d["Q_sim"].ravel()[:N].astype(np.float64)

    dt = args.dt if args.dt is not None else float(np.median(np.diff(time_raw)))
    t  = np.arange(N, dtype=np.float64) * dt   # uniform time axis from 0
    print(f"  dt = {dt:.5f} s   →   trajectory length = {N * dt:.2f} s")
    print(f"  Q_com range: [{Q_com.min() * _SCALE:.3f}, {Q_com.max() * _SCALE:.3f}] mL/min")

    # ------------------------------------------------------------------
    # Resolve checkpoint
    # ------------------------------------------------------------------
    _sep("Checkpoint resolution")
    ckpt_path, run_config = _resolve_checkpoint(args)
    print(f"  num_layers  = {run_config.num_layers}")
    print(f"  hidden_size = {run_config.hidden_size}")

    # ------------------------------------------------------------------
    # Initialize pipeline components
    # ------------------------------------------------------------------
    _sep("Initialising pipeline")

    print("  Starting MatlabBridge (persistent engine)...")
    from matlab_bridge import MatlabBridge
    bridge = MatlabBridge(fluid=args.fluid, mixer=args.mixer, pump=args.pump)

    print("  Initialising LSTMStepWrapper (recomputing norm stats from train split)...")
    from lstm_step import LSTMStepWrapper
    lstm = LSTMStepWrapper(
        ckpt_path=ckpt_path,
        run_config=run_config,
        train_list_path=args.train_list,
        data_folder=args.data_folder,
        device=args.device,
    )

    from dynamics import HybridDynamics
    dyn = HybridDynamics(bridge=bridge, lstm=lstm, dt=dt)

    state0 = dyn.make_initial_state(t0=float(time_raw[0]))
    ER = bridge.extrusion_ratio
    print(f"  State dim = {dyn.state_dim}  (3 ODE + 2×{lstm._num_layers}×{lstm._hidden_size} LSTM)")

    # ==================================================================
    # Section 1: Model Integrity Check
    # ==================================================================
    _sep("Section 1 / 2  —  Model Integrity Check")
    print("  Rolling out HybridDynamics.rollout() with U = [Q_com, w_nom]...")

    w_nom_full = np.full(N, args.w_nom)
    U_naive    = np.stack([Q_com, w_nom_full], axis=1)  # [N, 2]

    states_naive, Q_out_naive = dyn.rollout(state0, U_naive)

    # Extract analytical and residual components from our rollout
    # states[k].theta_dot = theta_dot_traj[k] for all k (see dynamics.py)
    Q_analytical_hybrid = np.array(
        [float(states_naive[k].theta_dot) * ER for k in range(N)],
        dtype=np.float64,
    )
    Q_res_predicted = Q_out_naive - Q_analytical_hybrid

    # ------------------------------------------------------------------
    # Windowed LSTM reference (replicates training-time inference paradigm)
    # ------------------------------------------------------------------
    print("\n  Running windowed LSTM inference (original training paradigm)...")
    print(f"  window_len = {args.window_len_s} s  ({int(round(args.window_len_s / dt))} steps),  "
          f"window_step = {args.window_step_s} s  ({int(round(args.window_step_s / dt))} steps)")
    Q_out_windowed = _run_windowed_lstm(
        Q_com=Q_com,
        Q_vbn=Q_vbn_npz,
        w_nom_m=args.w_nom,
        lstm=lstm,
        N=N,
        dt=dt,
        window_len_s=args.window_len_s,
        window_step_s=args.window_step_s,
    )

    # Metrics
    rmse_hybrid     = np.sqrt(np.mean((Q_out_naive      - Q_sim_npz) ** 2)) * _SCALE
    rmse_windowed   = np.sqrt(np.mean((Q_out_windowed   - Q_sim_npz) ** 2)) * _SCALE
    rmse_analytical = np.sqrt(np.mean((Q_vbn_npz        - Q_sim_npz) ** 2)) * _SCALE
    rmse_res        = np.sqrt(np.mean((Q_res_predicted  - Q_res_npz) ** 2)) * _SCALE
    delta_anal_max  = np.max(np.abs(Q_analytical_hybrid - Q_vbn_npz)) * _SCALE

    print(f"\n  [Flowrate agreement vs Q_sim (ground truth)]")
    print(f"    Windowed LSTM RMSE        : {rmse_windowed:.5f} mL/min  ← original training paradigm")
    print(f"    Step-mode hybrid RMSE     : {rmse_hybrid:.5f} mL/min  ← iLQR pipeline (new)")
    print(f"    Analytical-only RMSE      : {rmse_analytical:.5f} mL/min")
    print(f"    Analytical delta (max|Δ|) : {delta_anal_max:.2e} mL/min  "
          f"(should be ≈0 if ODE inputs are identical)")

    window_len_steps = int(round(args.window_len_s / dt))
    if N > window_len_steps:
        rmse_early = np.sqrt(np.mean(
            (Q_out_naive[:window_len_steps] - Q_out_windowed[:window_len_steps]) ** 2
        )) * _SCALE
        rmse_late = np.sqrt(np.mean(
            (Q_out_naive[window_len_steps:] - Q_out_windowed[window_len_steps:]) ** 2
        )) * _SCALE
        print(f"\n  [Step-mode vs windowed divergence  (window_len = {window_len_steps} steps)]")
        print(f"    Within first window  RMSE : {rmse_early:.5f} mL/min  (should be ≈0)")
        print(f"    Beyond first window  RMSE : {rmse_late:.5f} mL/min  "
              f"(h,c threading mismatch — grows with horizon)")

    print(f"\n  [Residual agreement]")
    print(f"    LSTM residual RMSE        : {rmse_res:.6f} mL/min")

    if rmse_hybrid < rmse_analytical:
        print("\n  ✓ Hybrid model outperforms analytical alone — LSTM correction active.")
    else:
        print("\n  ⚠ Hybrid model does NOT outperform analytical — check norm stats / checkpoint.")

    if rmse_windowed <= rmse_hybrid:
        print(f"  ℹ  Windowed RMSE ({rmse_windowed:.4f}) ≤ step-mode RMSE ({rmse_hybrid:.4f}) — "
              f"expected: training mismatch grows beyond ~{window_len_steps} steps.")
    else:
        print(f"  ✓ Step-mode RMSE ({rmse_hybrid:.4f}) ≤ windowed RMSE ({rmse_windowed:.4f}) — "
              f"step-mode matches or outperforms windowed on this horizon.")

    fig1 = _plot_integrity(
        t, Q_com, Q_vbn_npz, Q_sim_npz, Q_res_npz,
        Q_out_naive, Q_analytical_hybrid, Q_res_predicted,
        Q_out_windowed=Q_out_windowed,
    )

    # ==================================================================
    # Section 2: iLQR Optimisation
    # ==================================================================
    _sep("Section 2 / 2  —  iLQR Optimisation")

    q_ref  = Q_com.copy()
    R_mat  = np.diag(args.R_diag)
    u_min  = np.array([args.Q_min, args.w_min])
    u_max  = np.array([args.Q_max, args.w_max])

    print(f"  G = {args.G:.2e},  G_f = {args.G_f:.2e},  R = diag{args.R_diag}")
    print(f"  Q bounds: [{args.Q_min:.2e}, {args.Q_max:.2e}] m³/s")
    print(f"  w bounds: [{args.w_min*1e3:.1f}, {args.w_max*1e3:.1f}] mm")
    print(f"  max_iter = {args.max_iter},  tol = {args.tol}")
    print(f"\n  Starting iLQR solve (T={N} steps, ~{10*N} MATLAB calls/iter)...")

    from ilqr import ILQRSolver
    solver = ILQRSolver(
        dynamics=dyn,
        G=args.G,
        R=R_mat,
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
        U_init=U_naive.copy(),
        u_min=u_min,
        u_max=u_max,
    )

    rmse_naive_track = np.sqrt(np.mean((Q_out_naive - q_ref) ** 2)) * _SCALE
    rmse_opt_track   = np.sqrt(np.mean((Q_out_opt   - q_ref) ** 2)) * _SCALE
    print(f"\n  [iLQR tracking results]")
    print(f"    Naive tracking RMSE : {rmse_naive_track:.5f} mL/min")
    print(f"    iLQR tracking RMSE  : {rmse_opt_track:.5f} mL/min")
    print(f"    Cost iterations     : {len(cost_hist)}")
    if len(cost_hist) >= 2:
        reduction = (cost_hist[0] - cost_hist[-1]) / abs(cost_hist[0]) * 100
        print(f"    Cost reduction      : {reduction:.1f}%")

    if rmse_opt_track < rmse_naive_track:
        print("\n  ✓ iLQR improved tracking.")
    else:
        print("\n  ⚠ iLQR did NOT improve tracking — try more iterations or tune G/R.")

    fig2 = _plot_ilqr(
        t, q_ref, Q_out_naive, Q_out_opt,
        U_naive, U_opt, cost_hist,
    )

    # ------------------------------------------------------------------
    # Cleanup and display
    # ------------------------------------------------------------------
    bridge.quit()

    if not args.no_show:
        plt.show()

    print("\n  Done.")


if __name__ == "__main__":
    main()
