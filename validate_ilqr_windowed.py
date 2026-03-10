"""
validate_ilqr_windowed.py — Validate iLQR-optimized controls via the original
windowed prediction pipeline.

Loads a controls .npz saved by test_ilqr_test_set.py (--save_dir), then runs
both the naive controls (Q_cmd = Q_com, w_cmd = W_com) and the iLQR-optimized
controls (Q_cmd_opt, w_cmd_opt) through the original windowed inference stack:

    flow_predictor_analytical()     (MATLAB ODE — called internally by lstm func)
    flow_predictor_lstm_windowed()  (windowed LSTM with per-window h,c resets)

This is the definitive validation: if Q_pred_windowed_opt is closer to Q_com
than Q_pred_windowed_naive, the iLQR improvement is real and not an artefact
of the step-mode model simplification.

Usage::

    python validate_ilqr_windowed.py <path/to/controls.npz> [options]

    # Example:
    python validate_ilqr_windowed.py \\
        sweep_results/corner_60100_1000_controls_G1e+15_R1e-03.npz \\
        --save_dir sweep_results/

    # With explicit checkpoint (faster — no WandB API call):
    python validate_ilqr_windowed.py controls.npz \\
        --ckpt_path VBN-modeling/<run_id>/checkpoints/epoch=X.ckpt \\
        --config_path config.json

Outputs
-------
- Console RMSE table (naive windowed, iLQR windowed, iLQR step-mode)
- Figure: Q_com / Q_pred_naive / Q_pred_opt / Q_out_stepmode_opt
  Saved to --save_dir if provided.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Scale factor for display ─────────────────────────────────────────────────
_SCALE = 1e6 * 60.0 / 1e3          # m³/s → mL/min   (×1e9 gives nL/s; /1e3×60 = mL/min)
# Correct conversion: 1 m³/s = 1e6 cm³/s = 1e6 mL/s = 6e7 mL/min
# But project convention: _SCALE = 1e9 / 1e3 * 60? Let's match test_ilqr_test_set.py
# which uses np.sqrt(mean(diff²)) * _SCALE and reports in mL/min.
# From context: Q_com ≈ 1e-8 m³/s gives ~0.6 mL/min  →  _SCALE ≈ 6e7
_SCALE = 6e7   # m³/s → mL/min  (= 1e9 nL/s × 60 s/min / 1e3 nL/mL)


# ── Argument parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate iLQR controls via original windowed LSTM pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "controls_npz",
        help="Path to *_controls_*.npz saved by test_ilqr_test_set.py --save_dir",
    )

    grp = p.add_argument_group("Checkpoint")
    grp.add_argument("--sweep_id",   default="mnywg829",
                     help="WandB sweep ID to resolve best checkpoint.")
    grp.add_argument("--ckpt_path",  default=None,
                     help="Explicit .ckpt path (overrides --sweep_id).")
    grp.add_argument("--config_path", default=None,
                     help="JSON config path (required when using --ckpt_path).")

    p.add_argument("--data_folder", default="dataset/recursive_samples",
                   help="DataModule data folder (for norm_stats computation).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--no_show", action="store_true",
                   help="Skip plt.show().")
    p.add_argument("--save_dir", default=None,
                   help="Directory to save the validation PNG (created if absent).")
    return p.parse_args()


# ── Checkpoint resolution ─────────────────────────────────────────────────────

def _resolve_checkpoint(args: argparse.Namespace):
    """Return (ckpt_path, run_config). Mirrors test_ilqr_test_set._resolve_checkpoint."""
    if args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
        if args.config_path is not None:
            cfg_path = Path(args.config_path)
        else:
            candidates = [Path("config.json"), ckpt_path.parent.parent / "config.json"]
            cfg_path = next((c for c in candidates if c.exists()), None)
            if cfg_path is None:
                sys.exit("--ckpt_path given without --config_path; no config.json found.")
        import json
        from types import SimpleNamespace
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        run_config = SimpleNamespace(**cfg_dict)
        return ckpt_path, run_config

    print("  Querying WandB for best run in sweep …")
    from models.traj_WALR import get_best_run
    run_id, run_config = get_best_run(sweep_id=args.sweep_id)
    ckpt_dir = Path("VBN-modeling") / run_id / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        sys.exit(f"No .ckpt files found in {ckpt_dir}")
    ckpt_path = ckpt_files[-1].absolute()
    print(f"  Checkpoint: {ckpt_path}")
    return ckpt_path, run_config


# ── Figure ────────────────────────────────────────────────────────────────────

def _plot_validation(
    t: np.ndarray,
    Q_com: np.ndarray,
    Q_pred_naive: np.ndarray,
    Q_pred_opt: np.ndarray,
    Q_vbn_naive: np.ndarray,
    Q_vbn_opt: np.ndarray,
    Q_out_stepmode_opt: np.ndarray,
    rmse_naive_win: float,
    rmse_opt_win: float,
    rmse_opt_step: float,
    parent_id: str,
) -> plt.Figure:
    """
    Single-panel comparison figure.

    Lines:
      Q_com                  [black dashed]  — tracking target
      Q_pred_naive (windowed)[green]         — windowed pred, naive Q_cmd = Q_com
      Q_pred_opt   (windowed)[blue]          — windowed pred, iLQR Q_cmd_opt  ← key
      Q_out_stepmode_opt     [orange dashed] — iLQR step-mode prediction (reference)
      Q_vbn_opt    (ODE)     [purple dotted] — analytical ODE under opt controls
    """
    s = _SCALE
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        f"Windowed Pipeline Validation — {parent_id}\n"
        f"Naive windowed RMSE: {rmse_naive_win:.4f} mL/min  |  "
        f"iLQR windowed RMSE: {rmse_opt_win:.4f} mL/min  |  "
        f"iLQR step-mode RMSE: {rmse_opt_step:.4f} mL/min",
        fontsize=9,
    )

    ax.plot(t, Q_com * s,              "k--",  lw=1.4,  label="Q_com (target)")
    ax.plot(t, Q_pred_naive * s,       "g-",   lw=1.2,  label=f"Windowed naive  (RMSE {rmse_naive_win:.4f} mL/min)")
    ax.plot(t, Q_pred_opt * s,         "b-",   lw=1.5,  label=f"Windowed iLQR   (RMSE {rmse_opt_win:.4f} mL/min)")
    ax.plot(t, Q_out_stepmode_opt * s, "C1--", lw=1.0,  label=f"Step-mode iLQR  (RMSE {rmse_opt_step:.4f} mL/min)", alpha=0.7)
    ax.plot(t, Q_vbn_opt * s,          "m:",   lw=0.8,  label="ODE (opt ctrl)",  alpha=0.6)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Flowrate [mL/min]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    ctrl_path = Path(args.controls_npz)
    if not ctrl_path.exists():
        sys.exit(f"Controls file not found: {ctrl_path}")

    # ── 1. Load controls .npz ────────────────────────────────────────────────
    print(f"\n  Loading controls: {ctrl_path.name}")
    data = np.load(ctrl_path)

    t                    = data["t"]                     # [N]  seconds
    Q_cmd_opt            = data["Q_cmd_opt"]             # [N]  m³/s
    w_cmd_opt            = data["w_cmd_opt"]             # [N]  m
    Q_cmd_naive          = data["Q_cmd_naive"]           # [N]  m³/s
    w_cmd_naive          = data["w_cmd_naive"]           # [N]  m
    Q_com                = data["Q_com"]                 # [N]  m³/s
    Q_out_stepmode_opt   = data["Q_out_stepmode_opt"]    # [N]  m³/s
    Q_out_stepmode_naive = data["Q_out_stepmode_naive"]  # [N]  m³/s

    # Infer parent_id from filename: <parent_id>_controls_G*_R*.npz
    parent_id = ctrl_path.stem.split("_controls_")[0]
    print(f"  Parent: {parent_id}   N={len(t)} steps   "
          f"duration={t[-1] - t[0]:.2f} s   dt={t[1]-t[0]:.4f} s")

    # ── 2. Resolve checkpoint + norm_stats ───────────────────────────────────
    print("\n  Resolving checkpoint …")
    ckpt_path, run_config = _resolve_checkpoint(args)
    print(f"  num_layers  = {run_config.num_layers}")
    print(f"  hidden_size = {run_config.hidden_size}")

    print("\n  Loading DataModule for norm_stats …")
    from models.traj_WALR import DataModule
    dm = DataModule(run_config, data_folderpath=Path(args.data_folder))
    dm.setup("fit")
    norm_stats = dm.norm_stats

    # ── 3. Windowed pipeline — naive controls ────────────────────────────────
    print("\n  Running windowed pipeline on NAIVE controls (Q_cmd = Q_com) …")
    print("  (This will start a MATLAB engine — expected delay ~30–60 s)")
    from flow_predictor_lstm import flow_predictor_lstm_windowed
    Q_pred_naive, Q_vbn_naive, _ = flow_predictor_lstm_windowed(
        time_np     = t,
        command_np  = Q_cmd_naive,   # m³/s
        bead_np     = w_cmd_naive,   # m  (bead_units="m" → ×1000 internally for LSTM)
        model_type  = "WALR",
        ckpt_path   = ckpt_path,
        run_config  = run_config,
        norm_stats  = norm_stats,
        bead_units  = "m",
        device_type = args.device,
    )
    print("  Naive windowed pipeline: done.")

    # ── 4. Windowed pipeline — optimized controls ────────────────────────────
    print("\n  Running windowed pipeline on iLQR-OPTIMIZED controls …")
    print("  (Starting second MATLAB engine …)")
    Q_pred_opt, Q_vbn_opt, _ = flow_predictor_lstm_windowed(
        time_np     = t,
        command_np  = Q_cmd_opt,     # m³/s
        bead_np     = w_cmd_opt,     # m
        model_type  = "WALR",
        ckpt_path   = ckpt_path,
        run_config  = run_config,
        norm_stats  = norm_stats,
        bead_units  = "m",
        device_type = args.device,
    )
    print("  Optimized windowed pipeline: done.")

    # ── 5. RMSE table ────────────────────────────────────────────────────────
    rmse_naive_win  = np.sqrt(np.mean((Q_pred_naive - Q_com) ** 2)) * _SCALE
    rmse_opt_win    = np.sqrt(np.mean((Q_pred_opt   - Q_com) ** 2)) * _SCALE
    rmse_opt_step   = np.sqrt(np.mean((Q_out_stepmode_opt - Q_com) ** 2)) * _SCALE
    rmse_naive_step = np.sqrt(np.mean((Q_out_stepmode_naive - Q_com) ** 2)) * _SCALE

    print(f"\n{'=' * 62}")
    print(f"  Windowed pipeline validation — {parent_id}")
    print(f"{'=' * 62}")
    print(f"  {'Method':40}  {'RMSE (mL/min)':>14}")
    print(f"  {'-' * 40}  {'-' * 14}")
    print(f"  {'Naive step-mode  (hybrid model, Q_cmd=Q_com)':40}  {rmse_naive_step:>14.4f}")
    print(f"  {'Naive windowed   (original pipeline, Q_cmd=Q_com)':40}  {rmse_naive_win:>14.4f}")
    print(f"  {'iLQR  step-mode  (hybrid model, Q_cmd_opt)':40}  {rmse_opt_step:>14.4f}")
    print(f"  {'iLQR  windowed   (original pipeline, Q_cmd_opt)':40}  {rmse_opt_win:>14.4f}")
    print(f"{'=' * 62}")

    if rmse_opt_win < rmse_naive_win:
        impr = (rmse_naive_win - rmse_opt_win) / rmse_naive_win * 100
        print(f"  ✓ Windowed pipeline confirms iLQR improvement: "
              f"{impr:.1f}% RMSE reduction")
    else:
        print(f"  ⚠ Windowed pipeline does NOT confirm iLQR improvement — "
              f"step-mode model may be overfitting its own simplification.")

    if rmse_opt_step < rmse_opt_win:
        print(f"  Note: step-mode RMSE ({rmse_opt_step:.4f}) < windowed RMSE ({rmse_opt_win:.4f}) "
              f"— iLQR over-optimized for its internal model.")

    # ── 6. Figure ─────────────────────────────────────────────────────────────
    fig = _plot_validation(
        t=t,
        Q_com=Q_com,
        Q_pred_naive=Q_pred_naive,
        Q_pred_opt=Q_pred_opt,
        Q_vbn_naive=Q_vbn_naive,
        Q_vbn_opt=Q_vbn_opt,
        Q_out_stepmode_opt=Q_out_stepmode_opt,
        rmse_naive_win=rmse_naive_win,
        rmse_opt_win=rmse_opt_win,
        rmse_opt_step=rmse_opt_step,
        parent_id=parent_id,
    )

    if args.save_dir is not None:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = ctrl_path.stem.replace("_controls_", "_windowed_val_")
        png_path = out_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure: {png_path}")

    if not args.no_show:
        plt.show()

        plt.figure()
        plt.plot(t, Q_com)
        plt.plot(t, Q_cmd_naive)
        plt.plot(t, Q_cmd_opt)


        plt.figure()
        plt.plot(t, w_cmd_naive)
        plt.plot(t, w_cmd_opt)


    print("\n  Done.")


if __name__ == "__main__":
    main()

