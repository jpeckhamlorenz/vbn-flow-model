"""
test_ilqr_test_set.py — Run iLQR FFEC on all parent trajectories in splits/test.txt.

Reads splits/test.txt, identifies the unique parent trajectories (e.g.
corner_60100_1000 and flowrate_pattern_03_averaged_smoothed), loads their full
base files from dataset/, downsamples to dt = 0.01 s (matching LSTM training),
and for each trajectory:

  1. Runs windowed LSTM stitching over the FULL trajectory — the exact same
     inference paradigm used by traj_WALR_test.py, enabling direct comparison.
  2. Runs HybridDynamics.rollout() (naive) + ILQRSolver on the first
     --n_samples steps.

Two figures per trajectory:

  Figure A  "Flowrate Comparison"  (mirrors traj_WALR_test.py output)
    Panel 1 — flowrate time series:
      Q_tru/Q_sim/Q_exp  [red]          ground truth (full trajectory)
      Q_vbn              [green]        analytical ODE (full trajectory)
      Q_pred_windowed    [purple]       windowed LSTM (full trajectory)
      Q_com              [black dashed] commanded input (full trajectory)
      Q_out_naive        [blue]         step-mode hybrid rollout  (first n steps)
      Q_out_iLQR         [orange]       iLQR-optimised output     (first n steps)
      Grey shading shows the iLQR horizon [0, n_samples * dt].
    Panel 2 — bead width W_com (mirrors traj_WALR_test.py Figure 2)

  Figure B  "iLQR Diagnostics"
    Panel 1 — residuals on [0, n_samples]: Q_res from .npz (red),
              step-mode LSTM (blue), windowed LSTM (purple)
    Panel 2 — controls on [0, n_samples]: Q_cmd and w_cmd, naive vs iLQR
    Panel 3 — iLQR cost convergence

Usage (minimal — resolves best checkpoint from WandB sweep automatically)::

    python test_ilqr_test_set.py

Usage (quick smoke test)::

    python test_ilqr_test_set.py --n_samples 100 --max_iter 3

Usage (longer run)::

    python test_ilqr_test_set.py --n_samples 400 --max_iter 15

Runtime note:
  Windowed LSTM runs fast (pure PyTorch).  iLQR needs ~10 * n_samples MATLAB
  calls per iteration.  With n_samples=200, max_iter=10 expect ~5-15 min.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# m³/s → mL/min
_SCALE: float = 6e7
# Target timestep matching LSTM training (LSTM_data_splitter downsample target)
_DT_TRAIN: float = 0.01
# Fallback bead width [m] when W_com absent from base file
_W_DEFAULT: float = 0.0029
# Regex to extract _window_<N> suffix
_WINDOW_RE = re.compile(r"_window_(\d+)\.npz$")
# Ordered list of dirs to search for base trajectory files
_BASE_SEARCH_DIRS = [
    "dataset/LSTM_sim_samples",
    "dataset/LSTM_exp_samples/smoothed_samples",
    "dataset/LSTM_exp_samples/averaged_samples",
    "dataset/LSTM_exp_samples/all_samples",
    "dataset/LSTM_exp_samples",
]


# ======================================================================
# CLI
# ======================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="iLQR FFEC evaluation on test-set parent trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--test_list", default="splits/test.txt",
                   help="Path to test split .txt listing windowed .npz filenames")
    p.add_argument("--parents", nargs="+", default=None,
                   help="Override: run only these parent IDs instead of all from test.txt")
    p.add_argument("--n_samples", type=int, default=200,
                   help="Number of timesteps for iLQR horizon (default 200 = 2 s at dt=0.01 s).")
    p.add_argument("--dt_train", type=float, default=_DT_TRAIN,
                   help="Target dt [s] for downsampling base trajectories (default 0.01 s).")

    # Checkpoint
    grp = p.add_argument_group("Checkpoint")
    grp.add_argument("--sweep_id", default="mnywg829",
                     help="WandB sweep ID for get_best_run() when --ckpt_path absent.")
    grp.add_argument("--ckpt_path", default=None, help="Explicit .ckpt path.")
    grp.add_argument("--config_path", default=None, help="JSON config (with --ckpt_path).")

    # Norm stats
    p.add_argument("--train_list", default="splits/train.txt")
    p.add_argument("--data_folder", default="dataset/recursive_samples",
                   help="Folder with training .npz files listed in splits/train.txt")

    # Windowed LSTM
    p.add_argument("--window_len_s", type=float, default=4.5,
                   help="Window length [s] for windowed LSTM reference (default 4.5 s).")
    p.add_argument("--window_step_s", type=float, default=0.1,
                   help="Window step [s] for windowed LSTM reference (default 0.1 s).")

    # Physical constants
    p.add_argument("--fluid",  default="fluid_DOW121")
    p.add_argument("--mixer",  default="mixer_ISSM50nozzle")
    p.add_argument("--pump",   default="pump_viscotec_outdated")

    # iLQR cost
    p.add_argument("--G",    type=float, default=1e14)
    p.add_argument("--G_f",  type=float, default=None,
                   help="Terminal cost weight (default: same as --G)")
    p.add_argument("--R_diag", nargs=2, type=float, default=[1e-3, 1e-3],
                   metavar=("R_Q", "R_w"))
    p.add_argument("--S_diag", nargs=2, type=float, default=None,
                   metavar=("S_Q", "S_w"),
                   help="Rate penalty diagonal [S_Q, S_w]. Penalises control rate of change: "
                        "Σ (u[k]-u[k-1])^T diag(S) (u[k]-u[k-1]). "
                        "Controls are in SI (Q_cmd~1e-8 m^3/s, w_cmd~3e-3 m) so S must be "
                        "large to compete with tracking (G*e^2~1e-5 at G=1e15): "
                        "S_Q ~ 1e16+ (dQ/step~1e-10), S_w ~ 1e8+ (dw/step~1e-5). "
                        "Default: None.")

    # iLQR solver
    p.add_argument("--max_iter", type=int, default=10)
    p.add_argument("--tol",      type=float, default=1e-4)
    p.add_argument("--eps_ode",  type=float, default=1e-5)
    p.add_argument("--eps_ctrl", type=float, default=1e-7,
                   help="(legacy) Absolute FD perturbation for controls. Superseded by "
                        "relative eps below.")
    p.add_argument("--eps_ctrl_rel", type=float, default=1e-3,
                   help="Relative FD perturbation for controls: eps = max(rel*|u|, floor). "
                        "Default: 1e-3 (0.1%%).")
    p.add_argument("--eps_ctrl_floor_Q", type=float, default=1e-12,
                   help="Floor FD perturbation for Q_cmd [m^3/s]. Default: 1e-12.")
    p.add_argument("--eps_ctrl_floor_w", type=float, default=1e-7,
                   help="Floor FD perturbation for w_cmd [m]. Default: 1e-7.")
    p.add_argument("--use_windowed_cost", action="store_true",
                   help="Use windowed LSTM cost for iLQR (adjusts q_ref by windowed "
                        "residual each iteration so the optimizer targets the deployed "
                        "windowed pipeline). Requires no extra MATLAB calls.")
    p.add_argument("--analytical_only", action="store_true",
                   help="Run iLQR on the analytical ODE model only (no LSTM). "
                        "Finds Q_cmd_opt such that Q_vbn(Q_cmd_opt) ≈ Q_com. "
                        "The full windowed prediction is still run separately for "
                        "display. Much faster (~30× fewer PyTorch calls/iter) and "
                        "better-conditioned at long horizons.")

    # Control bounds
    p.add_argument("--Q_min", type=float, default=-1e-7)
    p.add_argument("--Q_max", type=float, default=1e-6)
    p.add_argument("--w_min", type=float, default=0.0007)
    p.add_argument("--w_max", type=float, default=0.0029)
    p.add_argument("--w_delta_plus",  type=float, default=0.0003,
                   help="Max bead width ABOVE W_com[k] at each timestep [m]. "
                        "E.g. 2e-4 (= 0.2 mm). Intersected with --w_max. "
                        "None = no relative upper bound (only --w_max applies).")
    p.add_argument("--w_delta_minus", type=float, default=0.0015,
                   help="Max bead width BELOW W_com[k] at each timestep [m]. "
                        "E.g. 4e-4 (= 0.4 mm). Intersected with --w_min. "
                        "None = no relative lower bound (only --w_min applies).")
    p.add_argument("--Q_delta_plus",  type=float, default=None,
                   help="Max flow command ABOVE Q_com[k] at each timestep [m³/s]. "
                        "E.g. 5e-9 (= 0.3 mL/min). Intersected with --Q_max. "
                        "Prevents chattering: iLQR cannot push Q_cmd far above Q_com. "
                        "None = no relative upper bound (only --Q_max applies).")
    p.add_argument("--Q_delta_minus", type=float, default=None,
                   help="Max flow command BELOW Q_com[k] at each timestep [m³/s]. "
                        "E.g. 5e-9 (= 0.3 mL/min). Intersected with --Q_min. "
                        "None = no relative lower bound (only --Q_min applies).")
    p.add_argument("--segment_len", type=int, default=None,
                   help="Split the iLQR horizon into sequential segments of this many steps. "
                        "ODE state (theta, theta_dot, motor_pos) threads between segments; "
                        "LSTM h,c resets to 0 at each segment start — matching the deployed "
                        "windowed pipeline which also resets per 4.5s window. "
                        "At dt=0.01s: --segment_len 450 = 4.5s segments, exactly one LSTM "
                        "training window, so step-mode ≡ windowed pipeline per segment. "
                        "Default: None = one segment (full N_ilqr horizon, current behaviour).")
    p.add_argument("--w_rate_max", type=float, default=None,
                   help="Max bead width change rate [mm/s]. E.g. 1.0 limits the nozzle to "
                        "1 mm/s. Prevents bang-bang chattering of w_cmd by enforcing "
                        "|w[k]-w[k-1]| <= w_rate_max * dt per step. "
                        "Converted internally to m/step = w_rate_max * 1e-3 * dt. "
                        "Default: None = no rate limit (only absolute/relative bounds apply).")
    p.add_argument("--q_rate_max", type=float, default=None,
                   help="Max flow rate change [mL/min/s]. E.g. 10.0 limits flow acceleration "
                        "to 10 mL/min per second. Prevents Q_cmd chattering by enforcing "
                        "|Q[k]-Q[k-1]| <= q_rate_max * dt per step. "
                        "Converted internally to m³/s/step = q_rate_max * (1e-6/60) * dt. "
                        "Default: None = no rate limit.")

    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--no_show", action="store_true",
                   help="Skip plt.show() (non-interactive / batch runs)")
    p.add_argument("--save_dir", default=None,
                   help="Directory to save figures as PNG files (created if absent). "
                        "Filenames: <parent_id>_{A,B}_G<G>_R<R>.png")

    return p.parse_args()


# ======================================================================
# Utilities
# ======================================================================

def _sep(title: str, width: int = 62) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _resolve_checkpoint(args: argparse.Namespace):
    """Return (ckpt_path, run_config) from sweep_id or explicit paths."""
    if args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            sys.exit(f"Checkpoint not found: {ckpt_path}")
        if args.config_path is not None:
            cfg_path = Path(args.config_path)
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


# ======================================================================
# Parse test.txt → unique parent IDs
# ======================================================================

def _parse_parents(test_list: Path) -> list[str]:
    """
    Extract and sort unique parent IDs from a windowed test split file.
    Each line is expected to be of the form  <parent_id>_window_<N>.npz.
    """
    parents: set[str] = set()
    with open(test_list) as f:
        for line in f:
            name = line.strip()
            if not name or not name.endswith(".npz"):
                continue
            m = _WINDOW_RE.search(name)
            if m:
                parents.add(name[: m.start()])
    return sorted(parents)


# ======================================================================
# Load and downsample base trajectory
# ======================================================================

def _find_base_file(parent_id: str, repo_root: Path) -> Path:
    """Search standard dataset directories for <parent_id>.npz."""
    for rel_dir in _BASE_SEARCH_DIRS:
        candidate = repo_root / rel_dir / f"{parent_id}.npz"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find base trajectory file for '{parent_id}'.\n"
        f"Searched in: {[rel_dir for rel_dir in _BASE_SEARCH_DIRS]}"
    )


def _load_base(
    parent_id: str,
    repo_root: Path,
    dt_target: float = _DT_TRAIN,
    w_default: float = _W_DEFAULT,
) -> dict:
    """
    Load a base trajectory, downsample to dt_target, and return a uniform dict.

    Returns:
        dict with keys: t, dt, N, Q_com, Q_vbn, Q_res, Q_tru, W_com
            t       [s]     uniform time axis starting at 0
            Q_com   [m³/s]  commanded flowrate
            Q_vbn   [m³/s]  analytical ODE flowrate
            Q_res   [m³/s]  residual (Q_tru - Q_vbn) — LSTM training target
            Q_tru   [m³/s]  ground truth (Q_sim for simulated, Q_exp for experimental)
            W_com   [m]     bead width (from file, or constant w_default if absent)
    """
    path = _find_base_file(parent_id, repo_root)
    print(f"  Base file: {path.relative_to(repo_root)}")
    d = np.load(path)

    # Ground truth key detection
    for key in ("Q_sim", "Q_exp", "Q_tru"):
        if key in d:
            q_tru_raw = d[key].ravel().astype(np.float64)
            print(f"  Ground truth: '{key}'")
            break
    else:
        raise KeyError(f"No ground-truth key (Q_sim / Q_exp / Q_tru) in {path.name}. "
                       f"Found: {list(d.keys())}")

    time_raw = d["time"].ravel().astype(np.float64)
    Q_com_raw = d["Q_com"].ravel().astype(np.float64)
    Q_vbn_raw = d["Q_vbn"].ravel().astype(np.float64)
    Q_res_raw = d["Q_res"].ravel().astype(np.float64)
    W_com_raw = (d["W_com"].ravel().astype(np.float64) if "W_com" in d
                 else np.full(len(time_raw), w_default, dtype=np.float64))

    dt_raw = float(np.median(np.diff(time_raw)))
    ds = max(1, int(round(dt_target / dt_raw)))
    dt = dt_raw * ds
    print(f"  Raw dt = {dt_raw:.5f} s  →  downsample ×{ds}  →  dt = {dt:.5f} s")

    sl = slice(None, None, ds)
    time_ds = time_raw[sl]
    N = len(time_ds)
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


# ======================================================================
# Windowed LSTM inference (mirrors traj_WALR_test.py stitch logic)
# ======================================================================

@torch.no_grad()
def _run_windowed_lstm(
    Q_com: np.ndarray,
    Q_vbn: np.ndarray,
    W_com: np.ndarray,
    lstm,               # LSTMStepWrapper
    N: int,
    dt: float,
    window_len_s: float = 4.5,
    window_step_s: float = 0.1,
) -> np.ndarray:
    """
    Replicate the windowed LSTM inference paradigm from traj_WALR_test.py.

    For each sliding window: h, c are reset to zero (matching training convention).
    Overlapping window predictions are averaged (matching stitch_parent()).

    Args:
        Q_com:   commanded flowrate [m³/s], shape (N,)
        Q_vbn:   analytical ODE flowrate [m³/s], shape (N,)
        W_com:   bead width [m], shape (N,)  (may be constant or variable)
        lstm:    LSTMStepWrapper (model loaded, in eval mode)
        N:       number of timesteps
        dt:      timestep [s]
        window_len_s:  window length [s] — default matches WindowParams (4.5 s)
        window_step_s: window step [s]  — default matches WindowParams (0.1 s)

    Returns:
        Q_out_windowed: Q_vbn + averaged residual [m³/s], shape (N,)
    """
    from lstm_step import _FLOW_SCALE, _BEAD_SCALE, _NORM_EPS

    window_len  = max(1, int(round(window_len_s  / dt)))
    window_step = max(1, int(round(window_step_s / dt)))

    feats_raw = np.stack([
        Q_com * _FLOW_SCALE,
        W_com * _BEAD_SCALE,
        Q_vbn * _FLOW_SCALE,
    ], axis=1)  # [N, 3]

    in_mu = lstm._in_mu.cpu().numpy()
    in_sd = lstm._in_sd.cpu().numpy()
    feats_norm = (feats_raw - in_mu[None, :]) / (in_sd[None, :] + _NORM_EPS)

    tgt_mu = lstm._tgt_mu
    tgt_sd = lstm._tgt_sd

    acc   = np.zeros(N, dtype=np.float64)
    count = np.zeros(N, dtype=np.float64)

    for s in range(0, N, window_step):
        e = min(s + window_len, N)
        if e == s:
            break
        # [1, win, 3]  — batch_first=True in WalrLSTM
        x_win = torch.tensor(feats_norm[s:e], dtype=torch.float32,
                              device=lstm.device).unsqueeze(0)
        h0, c0 = lstm.init_state()
        out, _ = lstm._net.lstm(x_win, (h0, c0))            # [1, win, hidden]
        y_norm = lstm._net.fc(out).squeeze(0).squeeze(-1)   # [win]
        res = (y_norm.cpu().numpy().astype(np.float64) * (tgt_sd + _NORM_EPS) + tgt_mu) / _FLOW_SCALE
        acc[s:e]   += res
        count[s:e] += 1.0

    valid = count > 0
    avg_res = np.zeros(N, dtype=np.float64)
    avg_res[valid] = acc[valid] / count[valid]
    return Q_vbn + avg_res


# ======================================================================
# Figure A — Flowrate Comparison  (mirrors traj_WALR_test.py)
# ======================================================================

def _plot_flowrate_comparison(
    parent_id: str,
    t_full: np.ndarray,
    Q_com: np.ndarray,
    Q_vbn: np.ndarray,
    Q_tru: np.ndarray,
    W_com: np.ndarray,
    Q_pred_windowed: np.ndarray,
    t_ilqr: np.ndarray,
    Q_out_naive: np.ndarray,
    Q_out_iLQR: np.ndarray,
    rmse_windowed: float,
    rmse_naive: float,
    rmse_iLQR: float,
) -> plt.Figure:
    """
    Two-panel figure matching the style of traj_WALR_test.py.

    Panel 1: Flowrate time series — full trajectory for windowed reference,
             overlaid with step-mode and iLQR outputs for the iLQR horizon.
    Panel 2: Bead width W_com — mirrors traj_WALR_test.py Figure 2.
    """
    S = _SCALE
    ilqr_end = t_ilqr[-1] if len(t_ilqr) > 0 else 0.0

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )
    fig.suptitle(
        f"Flowrate Comparison — {parent_id}\n"
        f"Purple (windowed) RMSE: {rmse_windowed:.4f} mL/min  |  "
        f"Blue (naive) RMSE: {rmse_naive:.4f} mL/min  |  "
        f"Orange (iLQR) RMSE: {rmse_iLQR:.4f} mL/min  "
        f"[vs Q_com, first {len(t_ilqr)} steps]",
        fontsize=10, fontweight="bold",
    )

    # ---- Panel 1: Flowrates ------------------------------------------------
    ax = axes[0]

    # Shade the iLQR horizon
    ax.axvspan(0.0, ilqr_end, alpha=0.06, color="royalblue", zorder=0,
               label=f"iLQR horizon  ({len(t_ilqr)} steps)")

    # Full-trajectory lines (matching traj_WALR_test.py colours)
    ax.plot(t_full, Q_tru            * S, color="red",    lw=2.0, ls="-",
            label="Truth  (Q_sim / Q_exp)")
    ax.plot(t_full, Q_vbn            * S, color="green",  lw=1.5, ls="-",
            label="Analytical ODE (Q_vbn)")
    ax.plot(t_full, Q_pred_windowed  * S, color="purple", lw=1.5, ls="-", alpha=0.85,
            label=f"Windowed LSTM  (RMSE {rmse_windowed:.4f} mL/min)")
    ax.plot(t_full, Q_com            * S, color="black",  lw=1.5, ls="--", alpha=0.7,
            label="Commanded  (Q_com)")

    # iLQR-horizon overlays
    ax.plot(t_ilqr, Q_out_naive * S, color="royalblue", lw=1.8, ls="-", alpha=0.85,
            label=f"Step-mode hybrid  (RMSE {rmse_naive:.4f} mL/min)")
    ax.plot(t_ilqr, Q_out_iLQR  * S, color="darkorange", lw=2.0, ls="-",
            label=f"iLQR optimised  (RMSE {rmse_iLQR:.4f} mL/min)")

    ax.set_ylabel("Flowrate [mL/min]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, color="0.75", lw=0.5)
    ax.set_title(
        f"Full trajectory  ({t_full[-1]:.1f} s)  |  "
        f"iLQR horizon: first {len(t_ilqr)} steps ({ilqr_end:.1f} s)  [blue shading]",
        fontsize=9,
    )

    # ---- Panel 2: Bead width -----------------------------------------------
    ax2 = axes[1]
    ax2.plot(t_full, W_com * 1e3, color="steelblue", lw=1.5, label="W_com")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Bead width [mm]")
    ax2.set_title("Bead width", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, color="0.75", lw=0.5)

    fig.tight_layout()
    return fig


# ======================================================================
# Figure B — iLQR Diagnostics
# ======================================================================

def _plot_ilqr_diagnostics(
    parent_id: str,
    t_ilqr: np.ndarray,
    Q_res_npz: np.ndarray,
    Q_res_stepmode: np.ndarray,
    Q_res_windowed: np.ndarray,
    Q_cmd_naive: np.ndarray,
    Q_cmd_opt: np.ndarray,
    W_cmd_naive: np.ndarray,
    W_cmd_opt: np.ndarray,
    cost_hist: list[float],
) -> plt.Figure:
    """
    Three-panel iLQR diagnostics figure (over the iLQR horizon only).

    Panel 1: Residuals — .npz target vs step-mode LSTM vs windowed LSTM
    Panel 2: Controls  — Q_cmd and w_cmd, naive vs iLQR-optimised
    Panel 3: iLQR cost convergence
    """
    S = _SCALE
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.suptitle(f"iLQR Diagnostics — {parent_id}", fontsize=11, fontweight="bold")

    # ---- Panel 1: Residuals ------------------------------------------------
    ax = axes[0]
    ax.plot(t_ilqr, Q_res_npz      * S, color="red",    lw=2.0, ls="-",
            label="Q_res from .npz  (LSTM training target)")
    ax.plot(t_ilqr, Q_res_stepmode * S, color="blue",   lw=1.5, ls="-", alpha=0.85,
            label="Q_res step-mode LSTM  (iLQR pipeline)")
    ax.plot(t_ilqr, Q_res_windowed * S, color="purple", lw=1.5, ls="-", alpha=0.8,
            label="Q_res windowed LSTM  (training paradigm)")

    rmse_res = np.sqrt(np.mean((Q_res_stepmode - Q_res_npz) ** 2)) * S
    ax.set_ylabel("Residual flowrate [mL/min]")
    ax.set_title(f"LSTM Residuals  —  step-mode vs .npz RMSE: {rmse_res:.5f} mL/min",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, color="0.75", lw=0.5)

    # ---- Panel 2: Controls -------------------------------------------------
    ax = axes[1]
    ax.plot(t_ilqr, Q_cmd_naive * S, color="royalblue",  lw=1.5, ls="--", alpha=0.7,
            label="Q_cmd naive")
    ax.plot(t_ilqr, Q_cmd_opt   * S, color="darkorange", lw=1.8, ls="-",
            label="Q_cmd iLQR")
    ax.set_ylabel("Q_cmd [mL/min]")
    ax.set_title("Optimised Controls", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, color="0.75", lw=0.5)

    ax_w = ax.twinx()
    ax_w.plot(t_ilqr, W_cmd_naive * 1e3, color="grey",       lw=1.0, ls="--", alpha=0.5,
              label="w_cmd naive [mm]")
    ax_w.plot(t_ilqr, W_cmd_opt   * 1e3, color="firebrick",  lw=1.5, ls="-",
              label="w_cmd iLQR [mm]")
    ax_w.set_ylabel("w_cmd [mm]")
    ax_w.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time [s]")

    # ---- Panel 3: Cost convergence -----------------------------------------
    ax = axes[2]
    iters = np.arange(len(cost_hist))
    ax.semilogy(iters, cost_hist, "k-o", markersize=5, lw=1.5)
    ax.set_xlabel("iLQR Iteration")
    ax.set_ylabel("Total Cost")
    ax.set_title("iLQR Cost Convergence", fontsize=9)
    ax.grid(True, which="both", color="0.75", lw=0.5)
    if len(iters) <= 25:
        ax.set_xticks(iters)

    fig.tight_layout()
    return fig


# ======================================================================
# Per-parent RMSE summary table
# ======================================================================

def _print_summary_table(results: list[dict]) -> None:
    """Print a side-by-side RMSE comparison table for all processed trajectories."""
    print("\n" + "=" * 75)
    print("  SUMMARY — RMSE vs Q_com tracking target  [mL/min]")
    print("=" * 75)
    print(f"  {'Parent':50s}  {'Windowed':>10}  {'Naive':>10}  {'iLQR':>10}")
    print("  " + "-" * 73)
    for r in results:
        print(f"  {r['parent']:50s}  {r['rmse_win']:>10.4f}  "
              f"{r['rmse_naive']:>10.4f}  {r['rmse_ilqr']:>10.4f}")
    print("=" * 75)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = _parse_args()
    if args.G_f is None:
        args.G_f = args.G

    repo_root = Path(__file__).resolve().parent

    # Output directory for figures (--save_dir)
    out_dir: Path | None = None
    if args.save_dir is not None:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Figures will be saved to: {out_dir.resolve()}")

    # ------------------------------------------------------------------
    # Identify parent trajectories
    # ------------------------------------------------------------------
    _sep("Identifying test-set parent trajectories")
    if args.parents:
        parents = args.parents
        print(f"  Using --parents override: {parents}")
    else:
        test_list = Path(args.test_list)
        if not test_list.exists():
            sys.exit(f"Test list not found: {test_list}")
        parents = _parse_parents(test_list)
        print(f"  Parsed {test_list}  →  {len(parents)} parent(s):")
        for p in parents:
            print(f"    - {p}")

    # ------------------------------------------------------------------
    # Resolve checkpoint (once, shared across trajectories)
    # ------------------------------------------------------------------
    _sep("Checkpoint resolution")
    ckpt_path, run_config = _resolve_checkpoint(args)
    print(f"  num_layers  = {run_config.num_layers}")
    print(f"  hidden_size = {run_config.hidden_size}")

    # ------------------------------------------------------------------
    # Initialise shared pipeline components
    # ------------------------------------------------------------------
    _sep("Initialising shared pipeline components")

    print("  Starting MatlabBridge (persistent engine)...")
    from matlab_bridge import MatlabBridge
    bridge = MatlabBridge(fluid=args.fluid, mixer=args.mixer, pump=args.pump)
    ER = bridge.extrusion_ratio

    print("  Initialising LSTMStepWrapper (computing norm stats from train split)...")
    from lstm_step import LSTMStepWrapper
    lstm = LSTMStepWrapper(
        ckpt_path=ckpt_path,
        run_config=run_config,
        train_list_path=args.train_list,
        data_folder=args.data_folder,
        device=args.device,
    )

    from dynamics import HybridDynamics

    # iLQR cost / absolute bounds (shared across trajectories)
    R_mat = np.diag(args.R_diag)
    Q_min_abs, Q_max_abs = args.Q_min, args.Q_max
    w_min_abs, w_max_abs = args.w_min, args.w_max
    _use_rel_bounds = (args.w_delta_plus  is not None or args.w_delta_minus is not None or
                       args.Q_delta_plus  is not None or args.Q_delta_minus is not None)

    # Accumulate per-trajectory metrics for final summary table
    results: list[dict] = []

    # ==================================================================
    # Per-trajectory loop
    # ==================================================================
    for parent_idx, parent_id in enumerate(parents):
        _sep(f"Trajectory {parent_idx + 1} / {len(parents)}  —  {parent_id}")

        # ---- Load base trajectory ----------------------------------------
        traj = _load_base(parent_id, repo_root, dt_target=args.dt_train)
        t_full   = traj["t"]
        dt       = traj["dt"]
        N_full   = traj["N"]
        Q_com    = traj["Q_com"]
        Q_vbn    = traj["Q_vbn"]
        Q_res    = traj["Q_res"]
        Q_tru    = traj["Q_tru"]
        W_com    = traj["W_com"]

        N_ilqr = min(args.n_samples, N_full)
        t_ilqr = t_full[:N_ilqr]

        print(f"\n  iLQR horizon: {N_ilqr} steps ({N_ilqr * dt:.2f} s)")

        # ---- Windowed LSTM over FULL trajectory (traj_WALR_test.py reference) ----
        print("\n  Running windowed LSTM on full trajectory "
              f"({N_full} samples, {N_full * dt:.1f} s)...")
        Q_pred_windowed_full = _run_windowed_lstm(
            Q_com=Q_com, Q_vbn=Q_vbn, W_com=W_com,
            lstm=lstm, N=N_full, dt=dt,
            window_len_s=args.window_len_s, window_step_s=args.window_step_s,
        )
        rmse_win_full = (np.sqrt(np.mean((Q_pred_windowed_full - Q_tru) ** 2))
                         * _SCALE)
        print(f"  Windowed LSTM RMSE vs Q_tru (full): {rmse_win_full:.4f} mL/min")

        # ---- HybridDynamics rollout (naive, iLQR horizon) -------------------
        dyn = HybridDynamics(bridge=bridge, lstm=lstm, dt=dt,
                             use_lstm=not args.analytical_only)
        state0 = dyn.make_initial_state(t0=float(t_full[0]))

        U_naive = np.stack([Q_com[:N_ilqr], W_com[:N_ilqr]], axis=1)  # [N_ilqr, 2]

        print(f"\n  Rolling out HybridDynamics (naive) for {N_ilqr} steps...")
        states_naive, Q_out_naive = dyn.rollout(state0, U_naive)

        # Extract analytical and residual from step-mode rollout
        Q_anal_stepmode = np.array(
            [float(states_naive[k].theta_dot) * ER for k in range(N_ilqr)],
            dtype=np.float64,
        )
        Q_res_stepmode = Q_out_naive - Q_anal_stepmode

        # Windowed residual on iLQR horizon (for diagnostics panel)
        Q_res_windowed_ilqr = Q_pred_windowed_full[:N_ilqr] - Q_vbn[:N_ilqr]

        rmse_naive = (np.sqrt(np.mean((Q_out_naive - Q_com[:N_ilqr]) ** 2))
                      * _SCALE)
        rmse_win_ilqr = (np.sqrt(np.mean(
            (Q_pred_windowed_full[:N_ilqr] - Q_com[:N_ilqr]) ** 2)) * _SCALE)

        print(f"  Step-mode RMSE vs Q_com (iLQR horizon): {rmse_naive:.4f} mL/min")
        print(f"  Windowed  RMSE vs Q_com (iLQR horizon): {rmse_win_ilqr:.4f} mL/min")

        # ---- iLQR optimisation -----------------------------------------------
        _sep(f"iLQR — {parent_id}")
        q_ref = Q_com[:N_ilqr].copy()

        # Build control bounds — time-varying if any relative deltas requested
        if _use_rel_bounds:
            # Width bounds
            dp_w = args.w_delta_plus  if args.w_delta_plus  is not None else np.inf
            dm_w = args.w_delta_minus if args.w_delta_minus is not None else np.inf
            w_lo = np.maximum(w_min_abs, W_com[:N_ilqr] - dm_w)  # [N_ilqr]
            w_hi = np.minimum(w_max_abs, W_com[:N_ilqr] + dp_w)  # [N_ilqr]
            # Flow bounds — relative to Q_com[k] to prevent chattering
            dp_q = args.Q_delta_plus  if args.Q_delta_plus  is not None else np.inf
            dm_q = args.Q_delta_minus if args.Q_delta_minus is not None else np.inf
            q_lo = np.maximum(Q_min_abs, Q_com[:N_ilqr] - dm_q)  # [N_ilqr]
            q_hi = np.minimum(Q_max_abs, Q_com[:N_ilqr] + dp_q)  # [N_ilqr]
            u_min = np.column_stack([q_lo, w_lo])  # [N, 2]
            u_max = np.column_stack([q_hi, w_hi])  # [N, 2]
        else:
            u_min = np.array([Q_min_abs, w_min_abs])  # [2] constant
            u_max = np.array([Q_max_abs, w_max_abs])  # [2] constant

        print(f"  G = {args.G:.2e},  G_f = {args.G_f:.2e},  R = diag{args.R_diag}")
        if args.S_diag is not None:
            print(f"  S = diag{args.S_diag}  (soft rate penalty)")
        if args.S_diag is not None and (args.w_rate_max is not None or args.q_rate_max is not None):
            print("  NOTE: Soft penalty (--S_diag) + hard rate limits both active. "
                  "Consider using only --S_diag for reliable convergence.")
        if args.Q_delta_plus is not None or args.Q_delta_minus is not None:
            dp_q_ml = args.Q_delta_plus  * 6e7 if args.Q_delta_plus  is not None else float("inf")
            dm_q_ml = args.Q_delta_minus * 6e7 if args.Q_delta_minus is not None else float("inf")
            print(f"  Q bounds: Q_com[k] +{dp_q_ml:.3f}/-{dm_q_ml:.3f} mL/min  "
                  f"(clipped to [{Q_min_abs:.2e}, {Q_max_abs:.2e}] m³/s abs)")
        else:
            print(f"  Q bounds: [{Q_min_abs:.2e}, {Q_max_abs:.2e}] m³/s  (constant)")
        if _use_rel_bounds and (args.w_delta_plus is not None or args.w_delta_minus is not None):
            dp_mm = args.w_delta_plus  * 1e3 if args.w_delta_plus  is not None else float("inf")
            dm_mm = args.w_delta_minus * 1e3 if args.w_delta_minus is not None else float("inf")
            print(f"  w bounds: W_com[k] ± [{dp_mm:.2f}, {dm_mm:.2f}] mm  "
                  f"(clipped to [{w_min_abs*1e3:.1f}, {w_max_abs*1e3:.1f}] mm abs)")
        else:
            print(f"  w bounds: [{w_min_abs * 1e3:.1f}, {w_max_abs * 1e3:.1f}] mm  (constant)")
        print(f"  max_iter = {args.max_iter},  tol = {args.tol}")
        print(f"  Starting iLQR solve  ({N_ilqr} steps, ~{10 * N_ilqr} MATLAB calls/iter)...")

        from ilqr import ILQRSolver
        S_mat = np.diag(args.S_diag) if args.S_diag is not None else None
        solver = ILQRSolver(
            dynamics=dyn,
            G=args.G,
            R=R_mat,
            G_f=args.G_f,
            S=S_mat,
            max_iter=args.max_iter,
            tol=args.tol,
            eps_ode=args.eps_ode,
            eps_ctrl=args.eps_ctrl,
            eps_ctrl_rel=args.eps_ctrl_rel,
            eps_ctrl_floor_Q=args.eps_ctrl_floor_Q,
            eps_ctrl_floor_w=args.eps_ctrl_floor_w,
            verbose=True,
        )
        if args.use_windowed_cost:
            print("  Mode: windowed cost iLQR (residual-adjusted reference)")
        if args.analytical_only:
            print("  Mode: analytical-only iLQR (ODE only — no LSTM in inner loop)")

        # ---- Segment loop (default: one segment = existing single-solve behaviour) ----
        from dynamics import HybridState   # alongside existing HybridDynamics import
        seg_len = args.segment_len or N_ilqr
        n_segs  = math.ceil(N_ilqr / seg_len)
        if n_segs > 1:
            print(f"  Segmented iLQR: {n_segs} × {seg_len}-step segments  "
                  f"({seg_len * dt:.2f} s each, LSTM h,c reset per segment)")

        U_opt      = np.copy(U_naive)      # [N_ilqr, 2]  — filled in below
        Q_out_iLQR = np.zeros(N_ilqr)     # [N_ilqr]
        cost_hist  = []
        state_curr = state0

        for s_idx in range(n_segs):
            s0 = s_idx * seg_len
            s1 = min(s0 + seg_len, N_ilqr)

            # With the prefix rollout, dyn.rollout(state0_global, U_full) starts
            # from h,c=0 and threads the LSTM through all prefix steps, so h,c
            # at the segment boundary naturally reflects processing from zero.
            # All segments can safely use hybrid mode.
            if n_segs > 1:
                dyn.use_lstm = not args.analytical_only

            if n_segs > 1:
                mode_str = "hybrid" if dyn.use_lstm else "analytical-only"
                print(f"\n  --- Segment {s_idx + 1}/{n_segs}: "
                      f"steps [{s0}:{s1}]  ({s0*dt:.2f}–{s1*dt:.2f} s) "
                      f"[{mode_str}] ---")

            # Slice bounds: support both constant [2] and time-varying [N, 2]
            um  = (u_min[s0:s1] if isinstance(u_min, np.ndarray) and u_min.ndim > 1
                   else u_min)
            umx = (u_max[s0:s1] if isinstance(u_max, np.ndarray) and u_max.ndim > 1
                   else u_max)

            # Build prefix from optimised controls of all previous segments.
            # For segment 1 (s0=0): no prefix, solve() uses the standard path.
            # For segments 2+: prefix = U_opt[:s0], state0_global = zero state.
            # solve() rolls from zero through the prefix then the segment,
            # ensuring rollout_ode() always starts from IC=[0,0].
            U_prefix = U_opt[:s0].copy() if s0 > 0 else None
            state0_global = state0 if s0 > 0 else None

            U_seg, Q_seg, ch = solver.solve(
                state0=state_curr,
                q_ref=q_ref[s0:s1],
                U_init=U_naive[s0:s1].copy(),
                u_min=um, u_max=umx,
                use_windowed_cost=args.use_windowed_cost,
                dt=dt if args.use_windowed_cost else None,
                w_rate_max=(args.w_rate_max * 1e-3 * dt if args.w_rate_max is not None
                            else None),
                q_rate_max=(args.q_rate_max * (1e-6 / 60) * dt if args.q_rate_max is not None
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
            print(f"  iLQR RMSE vs Q_com: {rmse_ilqr:.4f} mL/min  "
                  f"(cost reduction {reduction:.1f}%,  {len(cost_hist)} iterations)")
        else:
            print(f"  iLQR RMSE vs Q_com: {rmse_ilqr:.4f} mL/min")

        if rmse_ilqr < rmse_naive:
            print("  ✓ iLQR improved tracking.")
        else:
            print("  ⚠ iLQR did NOT improve tracking — try more iterations or tune G/R.")

        # ---- Metrics for summary table (windowed RMSE vs Q_com, iLQR horizon) --
        results.append(dict(
            parent=parent_id,
            rmse_win=rmse_win_ilqr,
            rmse_naive=rmse_naive,
            rmse_ilqr=rmse_ilqr,
        ))

        # ---- Save controls to disk (for validate_ilqr_windowed.py / deploy) ----
        if out_dir is not None:
            tag = f"G{args.G:.0e}_R{args.R_diag[0]:.0e}"
            ctrl_path = out_dir / f"{parent_id}_controls_{tag}.npz"
            np.savez(
                ctrl_path,
                t                    = t_ilqr,            # [N_ilqr]  seconds
                Q_cmd_opt            = U_opt[:, 0],        # [N_ilqr]  m³/s
                w_cmd_opt            = U_opt[:, 1],        # [N_ilqr]  m
                Q_cmd_naive          = U_naive[:, 0],      # [N_ilqr]  m³/s (= Q_com[:N_ilqr])
                w_cmd_naive          = U_naive[:, 1],      # [N_ilqr]  m    (= W_com[:N_ilqr])
                Q_com                = Q_com[:N_ilqr],     # [N_ilqr]  m³/s  iLQR tracking target
                Q_out_stepmode_opt   = Q_out_iLQR,         # [N_ilqr]  m³/s
                Q_out_stepmode_naive = Q_out_naive,        # [N_ilqr]  m³/s
            )
            print(f"  Saved controls: {ctrl_path.name}")

        # ---- Figures --------------------------------------------------------
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
            rmse_windowed=rmse_win_ilqr,
            rmse_naive=rmse_naive,
            rmse_iLQR=rmse_ilqr,
        )
        fig_a.canvas.manager.set_window_title(f"Fig A — {parent_id}") \
            if hasattr(fig_a.canvas, "manager") and fig_a.canvas.manager else None

        fig_b = _plot_ilqr_diagnostics(
            parent_id=parent_id,
            t_ilqr=t_ilqr,
            Q_res_npz=Q_res[:N_ilqr],
            Q_res_stepmode=Q_res_stepmode,
            Q_res_windowed=Q_res_windowed_ilqr,
            Q_cmd_naive=U_naive[:, 0],
            Q_cmd_opt=U_opt[:, 0],
            W_cmd_naive=U_naive[:, 1],
            W_cmd_opt=U_opt[:, 1],
            cost_hist=cost_hist,
        )
        fig_b.canvas.manager.set_window_title(f"Fig B — {parent_id}") \
            if hasattr(fig_b.canvas, "manager") and fig_b.canvas.manager else None

        # Save figures to disk if --save_dir was given
        if out_dir is not None:
            tag = f"G{args.G:.0e}_R{args.R_diag[0]:.0e}"
            path_a = out_dir / f"{parent_id}_A_{tag}.png"
            path_b = out_dir / f"{parent_id}_B_{tag}.png"
            fig_a.savefig(path_a, dpi=150, bbox_inches="tight")
            fig_b.savefig(path_b, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path_a.name}")
            print(f"  Saved: {path_b.name}")
            plt.close(fig_a)
            plt.close(fig_b)

    # ------------------------------------------------------------------
    # Cleanup and display
    # ------------------------------------------------------------------
    bridge.quit()

    _print_summary_table(results)

    if not args.no_show:
        plt.show()

    print("\n  Done.")


if __name__ == "__main__":
    main()
