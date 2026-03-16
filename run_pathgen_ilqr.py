"""run_pathgen_ilqr.py — Run iLQR FFEC on any vbn_pathgen trajectory.

Standalone CLI entry point that generates a trajectory from vbn_pathgen,
runs the hybrid ODE+LSTM flow prediction and iLQR optimization, saves
optimized controls, and optionally validates via the windowed pipeline.

Usage::

    # Basic run with m_VBN trajectory
    python run_pathgen_ilqr.py \\
        --trajectory m_VBN \\
        --speed 1.0 --accel 50 --dt_real 0.02 --max_bead 2.3 \\
        --G 1e15 --R_diag 1e10 1e2 --S_diag 1e12 1e6 \\
        --segment_len 450 --max_iter 20 \\
        --out_dir pathgen_ilqr_results/

    # With windowed validation
    python run_pathgen_ilqr.py \\
        --trajectory m_VBN \\
        --G 1e15 --R_diag 1e10 1e2 --S_diag 1e12 1e6 \\
        --segment_len 450 --max_iter 20 \\
        --validate --out_dir pathgen_ilqr_results/

Outputs (in --out_dir)::

    <traj_name>_controls_G<G>_R<R>.npz    — optimized Q_cmd, w_cmd
    <traj_name>_A_G<G>_R<R>.png           — flowrate comparison figure
    <traj_name>_B_G<G>_R<R>.png           — iLQR diagnostics figure
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from ilqr_config import ILQRConfig, add_ilqr_args, load_config


# ── Shared display scale ─────────────────────────────────────────────────────
_SCALE = 6e7  # m³/s → mL/min


def _parse_args() -> tuple[ILQRConfig, argparse.Namespace]:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Run iLQR FFEC on a vbn_pathgen trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Script-specific args --
    p.add_argument("--trajectory", required=True,
                   help="Name of trajectory function from vbn_pathgen "
                        "(e.g., m_VBN, corner_flowmatch, bead_flow_constant)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="PathGenerator speed [mm/s] (default: 1.0)")
    p.add_argument("--accel", type=float, default=50.0,
                   help="PathGenerator acceleration [mm/s^2] (default: 50)")
    p.add_argument("--dt_real", type=float, default=0.02,
                   help="PathGenerator sampling period [s] (default: 0.02)")
    p.add_argument("--max_bead", type=float, default=2.3,
                   help="PathGenerator max bead width [mm] (default: 2.3)")
    p.add_argument("--bead_width_override", action="store_true",
                   help="PathGenerator bead_width_override flag")
    p.add_argument("--use_poisson_compensation", action="store_true",
                   help="PathGenerator Poisson compensation flag")
    p.add_argument("--dt_target", type=float, default=0.01,
                   help="Target timestep for iLQR pipeline [s] (default: 0.01)")
    p.add_argument("--out_dir", default="pathgen_ilqr_results",
                   help="Output directory for controls .npz + figures")
    p.add_argument("--save_npz", action="store_true",
                   help="Also save the converted trajectory as standalone .npz")
    p.add_argument("--validate", action="store_true",
                   help="Run validate_ilqr_windowed.py after saving controls")
    p.add_argument("--ilqr_config", default=None,
                   help="Path to iLQR config JSON file")

    # -- Shared iLQR args --
    add_ilqr_args(p)

    raw = p.parse_args()
    cfg = load_config(raw, ilqr_config_path=getattr(raw, "ilqr_config", None))
    return cfg, raw


def _plot_results(
    traj_name: str,
    t_ilqr: np.ndarray,
    Q_com: np.ndarray,
    Q_out_naive: np.ndarray,
    Q_out_opt: np.ndarray,
    Q_cmd_naive: np.ndarray,
    Q_cmd_opt: np.ndarray,
    W_cmd_naive: np.ndarray,
    W_cmd_opt: np.ndarray,
    cost_hist: list,
    segment_len: int,
):
    """Generate flowrate comparison and diagnostics figures."""
    import matplotlib.pyplot as plt

    N = len(t_ilqr)
    n_segs = max(1, int(np.ceil(N / segment_len))) if segment_len else 1

    rmse_naive = np.sqrt(np.mean((Q_out_naive - Q_com) ** 2)) * _SCALE
    rmse_opt   = np.sqrt(np.mean((Q_out_opt   - Q_com) ** 2)) * _SCALE

    # ── Figure A: flowrate comparison ────────────────────────────────────
    fig_a, axes_a = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes_a

    # Segment shading
    for s in range(n_segs):
        s0 = s * segment_len
        s1 = min(s0 + segment_len, N)
        if s % 2 == 1:
            ax1.axvspan(t_ilqr[s0], t_ilqr[min(s1 - 1, N - 1)],
                        alpha=0.06, color="gray")

    ax1.plot(t_ilqr, Q_com * _SCALE, "k--", lw=1, label="Q_com (reference)")
    ax1.plot(t_ilqr, Q_out_naive * _SCALE, "b-", lw=1.5, alpha=0.7,
             label=f"Naive (RMSE={rmse_naive:.3f})")
    ax1.plot(t_ilqr, Q_out_opt * _SCALE, color="orange", lw=2,
             label=f"iLQR optimized (RMSE={rmse_opt:.3f})")
    ax1.set_ylabel("Flowrate [mL/min]")
    ax1.set_title(f"{traj_name} — iLQR FFEC")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_ilqr, W_cmd_naive * 1e3, "b-", lw=1, alpha=0.7, label="w_naive")
    ax2.plot(t_ilqr, W_cmd_opt * 1e3, color="orange", lw=1.5, label="w_opt")
    ax2.set_ylabel("Bead width [mm]")
    ax2.set_xlabel("Time [s]")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig_a.tight_layout()

    # ── Figure B: diagnostics ────────────────────────────────────────────
    fig_b, axes_b = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    ax_q, ax_w, ax_c = axes_b

    ax_q.plot(t_ilqr, Q_cmd_naive * _SCALE, "b-", lw=1, alpha=0.7, label="Q_cmd naive")
    ax_q.plot(t_ilqr, Q_cmd_opt * _SCALE, color="orange", lw=1.5, label="Q_cmd opt")
    ax_q.set_ylabel("Q_cmd [mL/min]")
    ax_q.set_xlabel("Time [s]")
    ax_q.legend()
    ax_q.grid(True, alpha=0.3)
    ax_q.set_title(f"{traj_name} — iLQR diagnostics")

    ax_w.plot(t_ilqr, W_cmd_naive * 1e3, "b-", lw=1, alpha=0.7, label="w_cmd naive")
    ax_w.plot(t_ilqr, W_cmd_opt * 1e3, color="orange", lw=1.5, label="w_cmd opt")
    ax_w.set_ylabel("Bead width [mm]")
    ax_w.set_xlabel("Time [s]")
    ax_w.legend()
    ax_w.grid(True, alpha=0.3)

    if cost_hist:
        ax_c.semilogy(range(len(cost_hist)), cost_hist, "k.-")
        ax_c.set_ylabel("Cost")
        ax_c.set_xlabel("Iteration")
        ax_c.grid(True, alpha=0.3)

    fig_b.tight_layout()

    return fig_a, fig_b


def main() -> None:
    import matplotlib.pyplot as plt

    cfg, args = _parse_args()
    if cfg.cost.G_f is None:
        cfg.cost.G_f = cfg.cost.G

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_name = args.trajectory
    tag = f"G{cfg.cost.G:.0e}_R{cfg.cost.R_diag[0]:.0e}"

    # ── Generate trajectory ──────────────────────────────────────────────
    print(f"\n== Generating trajectory: {traj_name} ==")
    from pathgen_bridge import get_trajectory_func, pathgen_to_npz

    # Ensure inputs/ is importable for PathGenerator
    import sys as _sys
    _inputs_dir = str(Path(__file__).resolve().parent / "inputs")
    if _inputs_dir not in _sys.path:
        _sys.path.insert(0, _inputs_dir)
    from vbn_pathgen import PathGenerator

    pathgen = PathGenerator(
        speed=args.speed,
        accel=args.accel,
        dt_real=args.dt_real,
        max_bead=args.max_bead,
        bead_width_override=args.bead_width_override,
        use_poisson_compensation=args.use_poisson_compensation,
    )

    traj_func = get_trajectory_func(traj_name)
    path_df = traj_func(pathgen)
    print(f"  Generated {len(path_df)} points  ({path_df['t'].iloc[-1]:.2f} s)")

    # Optionally save as standalone .npz
    if args.save_npz:
        npz_path = out_dir / f"{traj_name}_trajectory.npz"
        pathgen_to_npz(path_df, npz_path, dt_target=args.dt_target)

    # ── Build ilqr_kwargs from config ────────────────────────────────────
    ilqr_kwargs = dict(
        G=cfg.cost.G,
        G_f=cfg.cost.G_f,
        R_diag=cfg.cost.R_diag,
        S_diag=cfg.cost.S_diag,
        max_iter=cfg.solver.max_iter,
        tol=cfg.solver.tol,
        eps_ode=cfg.solver.eps_ode,
        eps_ctrl_rel=cfg.solver.eps_ctrl_rel,
        eps_ctrl_floor_Q=cfg.solver.eps_ctrl_floor_Q,
        eps_ctrl_floor_w=cfg.solver.eps_ctrl_floor_w,
        segment_len=cfg.bounds.segment_len,
        n_samples=cfg.run.n_samples,
        Q_min=cfg.bounds.Q_min,
        Q_max=cfg.bounds.Q_max,
        w_min=cfg.bounds.w_min,
        w_max=cfg.bounds.w_max,
        w_delta_plus=cfg.bounds.w_delta_plus,
        w_delta_minus=cfg.bounds.w_delta_minus,
        Q_delta_plus=cfg.bounds.Q_delta_plus,
        Q_delta_minus=cfg.bounds.Q_delta_minus,
        analytical_only=cfg.run.analytical_only,
        device=cfg.display.device,
        sweep_id=cfg.checkpoint.sweep_id,
    )
    if cfg.checkpoint.ckpt_path is not None:
        ilqr_kwargs["ckpt_path"] = cfg.checkpoint.ckpt_path
    if cfg.checkpoint.config_path is not None:
        ilqr_kwargs["config_path"] = cfg.checkpoint.config_path

    # ── Run iLQR ─────────────────────────────────────────────────────────
    ctrl_path = out_dir / f"{traj_name}_controls_{tag}.npz"
    from pathgen_bridge import run_ilqr_on_pathgen
    results = run_ilqr_on_pathgen(
        path_df,
        ilqr_kwargs=ilqr_kwargs,
        dt_target=args.dt_target,
        save_path=str(ctrl_path),
    )

    # ── Figures ──────────────────────────────────────────────────────────
    fig_a, fig_b = _plot_results(
        traj_name=traj_name,
        t_ilqr=results["t"],
        Q_com=results["Q_com"],
        Q_out_naive=results["Q_out_naive"],
        Q_out_opt=results["Q_out_opt"],
        Q_cmd_naive=results["Q_cmd_naive"],
        Q_cmd_opt=results["Q_cmd_opt"],
        W_cmd_naive=results["w_cmd_naive"],
        W_cmd_opt=results["w_cmd_opt"],
        cost_hist=results["cost_hist"],
        segment_len=cfg.bounds.segment_len,
    )

    path_a = out_dir / f"{traj_name}_A_{tag}.png"
    path_b = out_dir / f"{traj_name}_B_{tag}.png"
    fig_a.savefig(path_a, dpi=150, bbox_inches="tight")
    fig_b.savefig(path_b, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path_a.name}")
    print(f"  Saved: {path_b.name}")

    if not cfg.display.no_show:
        plt.show()
    else:
        plt.close("all")

    # ── Optional windowed validation ─────────────────────────────────────
    if args.validate:
        print("\n== Running windowed pipeline validation ==")
        val_cmd = [
            sys.executable, "validate_ilqr_windowed.py",
            str(ctrl_path),
            "--save_dir", str(out_dir),
            "--no_show",
            "--device", cfg.display.device,
        ]
        if cfg.checkpoint.ckpt_path:
            val_cmd += ["--ckpt_path", str(cfg.checkpoint.ckpt_path)]
            if cfg.checkpoint.config_path:
                val_cmd += ["--config_path", cfg.checkpoint.config_path]
        else:
            val_cmd += ["--sweep_id", cfg.checkpoint.sweep_id]
        val_cmd += ["--data_folder", cfg.data.data_folder]
        print("  CMD:", " ".join(val_cmd))
        subprocess.run(val_cmd, check=False)

    # ── Summary ──────────────────────────────────────────────────────────
    N = len(results["t"])
    dt = results["t"][1] - results["t"][0] if N > 1 else 0
    rmse_naive = np.sqrt(np.mean((results["Q_out_naive"] - results["Q_com"]) ** 2)) * _SCALE
    rmse_opt   = np.sqrt(np.mean((results["Q_out_opt"]   - results["Q_com"]) ** 2)) * _SCALE

    print(f"\n== Summary ==")
    print(f"  Trajectory  : {traj_name}")
    print(f"  Horizon     : {N} steps  ({N * dt:.2f} s)")
    print(f"  G={cfg.cost.G:.1e}  R={cfg.cost.R_diag[0]:.1e}  "
          f"iters={len(results['cost_hist'])}")
    print(f"  Naive RMSE  : {rmse_naive:.4f} mL/min")
    print(f"  iLQR  RMSE  : {rmse_opt:.4f} mL/min")
    print(f"\n  Controls saved to: {ctrl_path.resolve()}")
    print(f"  Figures  saved to: {out_dir.resolve()}/")
    print("\n  Done.")


if __name__ == "__main__":
    main()
