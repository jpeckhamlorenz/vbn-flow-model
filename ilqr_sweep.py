"""
ilqr_sweep.py — G × R grid-search driver for test_ilqr_test_set.py.

Runs one subprocess per (G, R) combination, streams live output to the
terminal, tees each run's output to an individual log file, saves figures
to a shared output directory, and prints a ranked summary table at the end.

Usage (overnight)::

    python ilqr_sweep.py                          # default grid, 450 samples
    python ilqr_sweep.py --n_samples 50 --max_iter 10   # quick smoke test

All extra arguments (--n_samples, --max_iter, --tol, --Q_min/max, --w_min/max,
--sweep_id, --ckpt_path, --config_path, --device) are forwarded to
test_ilqr_test_set.py unchanged.  G and R are always set by the sweep grid.

Output layout::

    sweep_results/
        G1e+12_R1e-04.log          <- per-run transcript
        G1e+13_R1e-04.log
        ...
        corner_60100_1000_A_G1e+12_R1e-04.png   <- Figure A per run
        corner_60100_1000_B_G1e+12_R1e-04.png   <- Figure B per run
        ...
        sweep_summary.txt          <- final ranked table

Monitor progress::

    tail -f sweep_results/G1e+14_R1e-03.log
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Grid definition ─────────────────────────────────────────────────────────
_G_VALUES: list[float] = [1e12, 1e13, 1e14, 1e15]
_R_VALUES: list[float] = [1e-4, 1e-3, 1e-2]

# ── RMSE line pattern from _print_summary_table() in test_ilqr_test_set.py ──
# Matches lines like:
#   corner_60100_1000     0.4523      0.1872
_RMSE_RE = re.compile(
    r"^\s*(\S+)\s+([\d.]+(?:e[+-]?\d+)?)\s+([\d.]+(?:e[+-]?\d+)?)\s*$"
)


# ── Argument parsing ─────────────────────────────────────────────────────────

def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    """
    Parse sweep-level args; collect everything else as passthrough args for
    test_ilqr_test_set.py.  Returns (sweep_args, passthrough_list).
    """
    p = argparse.ArgumentParser(
        description="G × R grid-search driver for test_ilqr_test_set.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out_dir",
        default="sweep_results",
        help="Directory for logs, figures, and summary (created if absent).",
    )
    p.add_argument(
        "--G_values",
        nargs="+",
        type=float,
        default=_G_VALUES,
        metavar="G",
        help="Tracking cost weights to sweep.",
    )
    p.add_argument(
        "--R_values",
        nargs="+",
        type=float,
        default=_R_VALUES,
        metavar="R",
        help="Control cost weights to sweep (applied to both Q_cmd and w_cmd).",
    )
    # Pass-through defaults — override on command line as needed
    p.add_argument("--n_samples", type=int, default=450)
    p.add_argument("--max_iter",  type=int, default=50)
    p.add_argument("--tol",       type=float, default=1e-5)

    sweep_args, extra = p.parse_known_args()
    return sweep_args, extra


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tag(G: float, R: float) -> str:
    return f"G{G:.0e}_R{R:.0e}"


def _build_cmd(
    G: float,
    R: float,
    out_dir: Path,
    sweep_args: argparse.Namespace,
    extra: list[str],
) -> list[str]:
    return [
        sys.executable, "test_ilqr_test_set.py",
        "--G",       str(G),
        "--R_diag",  str(R), str(R),
        "--n_samples", str(sweep_args.n_samples),
        "--max_iter",  str(sweep_args.max_iter),
        "--tol",       str(sweep_args.tol),
        "--save_dir",  str(out_dir),
        "--no_show",
        *extra,  # any user-supplied passthrough flags
    ]


def _run_one(
    G: float,
    R: float,
    out_dir: Path,
    sweep_args: argparse.Namespace,
    extra: list[str],
) -> dict[str, float]:
    """
    Run one (G, R) combination.  Streams output live to terminal and tees to
    a per-run log file.  Returns dict mapping parent_id → (rmse_naive, rmse_ilqr).
    """
    tag = _tag(G, R)
    log_path = out_dir / f"{tag}.log"
    cmd = _build_cmd(G, R, out_dir, sweep_args, extra)

    header = (
        f"\n{'=' * 66}\n"
        f"  G = {G:.2e}   R = {R:.2e}   [{datetime.now():%H:%M:%S}]\n"
        f"{'=' * 66}"
    )
    print(header, flush=True)

    results: dict[str, tuple[float, float]] = {}

    with open(log_path, "w") as log_f:
        log_f.write(header + "\n")
        log_f.write("CMD: " + " ".join(cmd) + "\n\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)

            # Parse RMSE summary lines on the fly
            m = _RMSE_RE.match(line)
            if m:
                parent_id, rmse_naive, rmse_ilqr = m.group(1), float(m.group(2)), float(m.group(3))
                results[parent_id] = (rmse_naive, rmse_ilqr)

        proc.wait()
        rc = proc.returncode
        footer = f"\n[exit code {rc}]  [{datetime.now():%H:%M:%S}]\n"
        print(footer, flush=True)
        log_f.write(footer)

    return results


def _print_ranked_table(
    all_results: list[tuple[float, float, str, float, float]],
    summary_path: Path,
) -> None:
    """
    Print and save a table: G | R | parent_id | RMSE_naive | RMSE_iLQR | Δ%
    Sorted by mean iLQR RMSE ascending.
    """
    if not all_results:
        print("\n  (No RMSE data collected — check per-run logs for errors.)")
        return

    # Sort by iLQR RMSE ascending
    all_results.sort(key=lambda r: r[4])

    header = (
        f"\n{'=' * 78}\n"
        f"  SWEEP SUMMARY — sorted by iLQR RMSE (best first)\n"
        f"{'=' * 78}\n"
        f"  {'G':>8}  {'R':>8}  {'Parent':40}  {'Naive':>8}  {'iLQR':>8}  {'Δ%':>8}\n"
        f"  {'-' * 8}  {'-' * 8}  {'-' * 40}  {'-' * 8}  {'-' * 8}  {'-' * 8}"
    )
    print(header)

    lines = [header]
    for G, R, parent_id, rmse_naive, rmse_ilqr in all_results:
        delta = 100.0 * (rmse_naive - rmse_ilqr) / rmse_naive if rmse_naive > 0 else 0.0
        sign = "+" if delta >= 0 else ""
        row = (
            f"  {G:>8.1e}  {R:>8.1e}  {parent_id:40}  "
            f"{rmse_naive:>8.4f}  {rmse_ilqr:>8.4f}  {sign}{delta:>7.1f}%"
        )
        print(row)
        lines.append(row)

    footer = f"{'=' * 78}\n  RMSE units: mL/min\n{'=' * 78}"
    print(footer)
    lines.append(footer)

    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\n  Summary saved to: {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sweep_args, extra = _parse_args()

    out_dir = Path(sweep_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G_values = sweep_args.G_values
    R_values = sweep_args.R_values
    n_combos = len(G_values) * len(R_values)

    print(
        f"\n{'=' * 66}\n"
        f"  iLQR G × R sweep\n"
        f"  G values : {[f'{g:.0e}' for g in G_values]}\n"
        f"  R values : {[f'{r:.0e}' for r in R_values]}\n"
        f"  Combos   : {n_combos}\n"
        f"  n_samples: {sweep_args.n_samples}  "
        f"max_iter: {sweep_args.max_iter}  tol: {sweep_args.tol:.0e}\n"
        f"  Output   : {out_dir.resolve()}\n"
        f"  Started  : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"{'=' * 66}"
    )

    # Flat list of (G, R, parent_id, rmse_naive, rmse_ilqr) for final table
    all_results: list[tuple[float, float, str, float, float]] = []

    combo_idx = 0
    for G in G_values:
        for R in R_values:
            combo_idx += 1
            print(f"\n  [{combo_idx}/{n_combos}]  G={G:.1e}  R={R:.1e}", flush=True)
            run_results = _run_one(G, R, out_dir, sweep_args, extra)
            for parent_id, (rmse_naive, rmse_ilqr) in run_results.items():
                all_results.append((G, R, parent_id, rmse_naive, rmse_ilqr))

    print(f"\n  All {n_combos} combinations finished.  [{datetime.now():%H:%M:%S}]")
    _print_ranked_table(all_results, out_dir / "sweep_summary.txt")


if __name__ == "__main__":
    main()
