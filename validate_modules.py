"""
validate_modules.py — Three-test validation script for the iLQR FFEC modules.

Tests:
  1. MatlabBridge.rollout_ode()  vs  flow_predictor_analytical()
     Compares new persistent-engine ODE wrapper to the original per-call function.

  2. LSTMStepWrapper.step()  vs  WalrLSTM batch forward pass
     Verifies step-mode hidden-state threading gives identical output to batch mode.

  3. ILQRSolver._backward_pass()  vs  scipy DARE (LQR reference)
     Verifies Riccati recursion converges to the analytically known steady-state gains.

Usage::

    # All tests (MATLAB + checkpoint required for 1 & 2):
    python validate_modules.py --tests all \\
        --ckpt_path VBN-modeling/<run_id>/checkpoints/epoch=X.ckpt \\
        --config_path config.json \\
        --train_list splits/train.txt \\
        --data_folder dataset/LSTM_sim_samples

    # Test 3 only (no MATLAB / checkpoint needed):
    python validate_modules.py --tests 3

    # Tests 1 and 2 with a short trajectory (faster):
    python validate_modules.py --tests 1,2 --n_samples 300 \\
        --ckpt_path ... --config_path ...
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

# ======================================================================
# ANSI colour helpers
# ======================================================================

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _pass(msg: str = "") -> None:
    print(f"  {_GREEN}{_BOLD}PASS{_RESET}" + (f"  {msg}" if msg else ""))


def _fail(msg: str = "") -> None:
    print(f"  {_RED}{_BOLD}FAIL{_RESET}" + (f"  {msg}" if msg else ""))


def _skip(msg: str = "") -> None:
    print(f"  {_YELLOW}{_BOLD}SKIP{_RESET}" + (f"  {msg}" if msg else ""))


def _header(title: str) -> None:
    print(f"\n{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}{title}{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}")


# ======================================================================
# Test 1 — MatlabBridge.rollout_ode() vs flow_predictor_analytical()
# ======================================================================

def test1_matlab_bridge(
    data_path: Path,
    w_nom: float,
    n_samples: int | None,
) -> bool:
    """
    Compare Q_analytical output of MatlabBridge.rollout_ode() to the
    original flow_predictor_analytical() function.

    Both call the same underlying MATLAB function (VBN_flow_model_solver.m)
    with identical inputs, so outputs should match to machine epsilon.

    Args:
        data_path:  path to .npz with 'time' and 'Q_com' arrays
        w_nom:      nominal bead width [m] (used as constant w_cmd since .npz has no W_com)
        n_samples:  if given, use only first n_samples timesteps (for speed)

    Returns:
        True if PASS, False if FAIL.
    """
    _header("Test 1: MatlabBridge.rollout_ode() vs flow_predictor_analytical()")

    try:
        import matlab.engine  # noqa: F401 — just check it's importable
    except ImportError:
        _skip("matlab.engine not importable — skipping Test 1")
        return True  # don't count as failure

    # ---- load data
    print(f"  Loading: {data_path}")
    d = np.load(data_path)
    ts = d["time"].ravel().astype(np.float64)
    Q_com = d["Q_com"].ravel().astype(np.float64)

    if n_samples is not None and n_samples < len(ts):
        ts = ts[:n_samples]
        Q_com = Q_com[:n_samples]
        print(f"  Using first {n_samples} samples (dt = {np.median(np.diff(ts)):.4f} s)")
    else:
        print(f"  Using all {len(ts)} samples")

    w_cmd = np.full_like(ts, w_nom)  # constant bead width

    # ---- Reference: flow_predictor_analytical (starts+quits its own engine)
    print("  Running flow_predictor_analytical() [will start+quit MATLAB]...")
    from flow_predictor_analytical import flow_predictor as _fp_analytical
    try:
        _, _, _, Q_vbn_ref = _fp_analytical(ts, Q_com, w_cmd, match_time_steps=True)
        Q_vbn_ref = np.asarray(Q_vbn_ref, dtype=np.float64).ravel()
    except Exception as exc:
        _fail(f"flow_predictor_analytical raised: {exc}")
        return False

    # ---- New: MatlabBridge.rollout_ode (persistent engine)
    print("  Running MatlabBridge.rollout_ode() [persistent engine]...")
    from matlab_bridge import MatlabBridge
    bridge = None
    try:
        bridge = MatlabBridge()
        _, _, Q_analytical_new = bridge.rollout_ode(ts, Q_com, w_cmd)
        Q_analytical_new = np.asarray(Q_analytical_new, dtype=np.float64).ravel()
    except Exception as exc:
        _fail(f"MatlabBridge.rollout_ode() raised: {exc}")
        return False
    finally:
        if bridge is not None:
            bridge.quit()

    # ---- Compare
    abs_err = np.abs(Q_vbn_ref - Q_analytical_new)
    max_err = float(np.max(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))
    threshold = 1e-12  # m³/s — machine epsilon for identical MATLAB calls

    print(f"  Q_vbn_ref range:  [{Q_vbn_ref.min():.4e}, {Q_vbn_ref.max():.4e}] m³/s")
    print(f"  Max abs error:    {max_err:.3e} m³/s   (threshold: {threshold:.0e})")
    print(f"  RMSE:             {rmse:.3e} m³/s")

    passed = max_err < threshold
    if passed:
        _pass("rollout_ode() output matches flow_predictor_analytical()")
    else:
        _fail(
            f"max error {max_err:.3e} exceeds threshold {threshold:.0e} m³/s\n"
            "  (Check that both functions use the same constants and MATLAB function.)"
        )
    return passed


# ======================================================================
# Test 2 — LSTMStepWrapper.step() vs WalrLSTM batch forward pass
# ======================================================================

def test2_lstm_step(
    data_path: Path,
    ckpt_path: Path,
    run_config,
    train_list: Path,
    data_folder: Path,
    w_nom: float,
    n_samples: int | None,
    device: str,
) -> bool:
    """
    Verify that LSTMStepWrapper.step() in sequential mode gives identical
    output to WalrLSTM.forward() in batch mode.

    In eval mode, PyTorch disables dropout, so the two code paths must give
    numerically identical outputs (differences only from float32 rounding order).

    Both use the same normalization stats (computed from train split at init).

    Args:
        data_path:    .npz with 'Q_com' and 'Q_vbn' arrays
        ckpt_path:    path to LightningModule checkpoint
        run_config:   config object with hidden_size, num_layers, lr, huber_delta
        train_list:   path to splits/train.txt
        data_folder:  path to folder containing training .npz files
        w_nom:        nominal bead width [m]
        n_samples:    if given, use only first n_samples timesteps
        device:       'cpu', 'cuda', or 'mps'

    Returns:
        True if PASS, False if FAIL.
    """
    _header("Test 2: LSTMStepWrapper.step() vs WalrLSTM batch forward pass")

    import torch

    # ---- load data
    print(f"  Loading: {data_path}")
    d = np.load(data_path)
    Q_com = d["Q_com"].ravel().astype(np.float64)
    Q_vbn = d["Q_vbn"].ravel().astype(np.float64)

    if n_samples is not None and n_samples < len(Q_com):
        Q_com = Q_com[:n_samples]
        Q_vbn = Q_vbn[:n_samples]
        print(f"  Using first {n_samples} samples")
    else:
        print(f"  Using all {len(Q_com)} samples")

    T = len(Q_com)
    w_nom_mm = w_nom * 1000.0  # bead width in mm (matching training)

    # ---- Initialise LSTMStepWrapper (loads checkpoint + norm stats)
    print(f"  Loading LSTMStepWrapper from {ckpt_path}")
    from lstm_step import LSTMStepWrapper, _FLOW_SCALE, _BEAD_SCALE, _NORM_EPS
    try:
        wrapper = LSTMStepWrapper(
            ckpt_path=ckpt_path,
            run_config=run_config,
            train_list_path=train_list,
            data_folder=data_folder,
            device=device,
        )
    except Exception as exc:
        _fail(f"LSTMStepWrapper init raised: {exc}")
        traceback.print_exc()
        return False

    # ---- Build normalised input tensor (reuse wrapper's norm stats for both paths)
    in_mu = wrapper._in_mu.cpu().numpy()   # [3]
    in_sd = wrapper._in_sd.cpu().numpy()   # [3]
    tgt_mu = wrapper._tgt_mu
    tgt_sd = wrapper._tgt_sd
    dev = wrapper.device

    x_raw = np.stack([
        Q_com * _FLOW_SCALE,
        np.full(T, w_nom_mm),
        Q_vbn * _FLOW_SCALE,
    ], axis=1).astype(np.float32)  # [T, 3]

    x_norm_np = (x_raw - in_mu[None, :]) / (in_sd[None, :] + _NORM_EPS)  # [T, 3]
    x_t = torch.from_numpy(x_norm_np).unsqueeze(0).to(dev)  # [1, T, 3]
    len_t = torch.tensor([T], dtype=torch.long, device=dev)

    # ---- Reference: WalrLSTM batch forward pass (uses pack_padded_sequence)
    print("  Running batch forward pass [WalrLSTM.forward()]...")
    net = wrapper._net
    net.eval()
    try:
        with torch.no_grad():
            y_hat_norm_batch = net(x_t, len_t)[0, :, 0].cpu().numpy()  # [T]
    except Exception as exc:
        _fail(f"Batch forward pass raised: {exc}")
        return False

    Q_res_batch = (y_hat_norm_batch * (tgt_sd + _NORM_EPS) + tgt_mu) / _FLOW_SCALE  # [T]

    # ---- Step mode: sequential net.lstm calls with explicit (h, c) threading
    print(f"  Running step-by-step mode [LSTMStepWrapper.step()] T={T}...")
    Q_res_step = np.empty(T, dtype=np.float64)
    h, c = wrapper.init_state()
    try:
        for k in range(T):
            Q_res_k, h, c = wrapper.step(
                Q_cmd=float(Q_com[k]),
                w_cmd_m=w_nom,
                Q_analytical=float(Q_vbn[k]),
                h=h,
                c=c,
            )
            Q_res_step[k] = Q_res_k
    except Exception as exc:
        _fail(f"Step mode raised at k={k}: {exc}")
        return False

    # ---- Compare
    abs_err = np.abs(Q_res_batch - Q_res_step)
    max_err = float(np.max(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))
    # Typical Q_res magnitude
    q_scale = float(np.mean(np.abs(Q_res_batch))) + 1e-20
    rel_err = float(np.max(abs_err / q_scale))
    threshold_abs = 1e-5  # m³/s

    print(f"  Q_res range (batch): [{Q_res_batch.min():.4e}, {Q_res_batch.max():.4e}] m³/s")
    print(f"  Max abs error: {max_err:.3e} m³/s   (threshold: {threshold_abs:.0e})")
    print(f"  Max rel error: {rel_err:.3e}")
    print(f"  RMSE:          {rmse:.3e} m³/s")

    passed = max_err < threshold_abs
    if passed:
        _pass("Step mode matches batch mode within float32 tolerance")
    else:
        _fail(
            f"max error {max_err:.3e} m³/s exceeds threshold {threshold_abs:.0e}\n"
            "  (Check normalization stats and that dropout is disabled in eval mode.)"
        )
    return passed


# ======================================================================
# Test 3 — ILQRSolver._backward_pass() Riccati recursion vs scipy DARE
# ======================================================================

def test3_ilqr_riccati() -> bool:
    """
    Test iLQR Riccati backward pass on a toy discrete-time linear system with
    known LQR gains (computed via scipy's DARE solver) and verify convergence.

    System: x_{k+1} = A @ x_k + B @ u_k,  q_out_k = C @ x_k  (scalar output)
    Cost:   Σ G*(q_out_k - q_ref_k)^2 + u_k @ R @ u_k

    At steady state (far from terminal boundary), the iLQR feedback gains K_k
    must satisfy K_k = -K_lqr where K_lqr is the solution to the DARE with
    Q_state = 2G * C.T @ C and R_eff = 2R  (factor-of-2 from Gauss-Newton approx).

    This test uses T=200 timesteps and checks the gain at k=50.

    Returns:
        True if PASS, False if FAIL.
    """
    _header("Test 3: ILQRSolver._backward_pass() Riccati recursion vs scipy DARE")

    try:
        import scipy.linalg
    except ImportError:
        _fail("scipy not available — install scipy to run Test 3")
        return False

    # ---- Toy system
    n, m = 5, 2
    rng = np.random.default_rng(42)

    A_raw = rng.standard_normal((n, n))
    # Scale eigenvalues to be strictly inside unit circle (stable system)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(A_raw)))
    A = A_raw / (spectral_radius + 0.5)

    B = rng.standard_normal((n, m))
    C = np.zeros(n)
    C[0] = 1.0   # scalar output q_out_k = C @ x_k  (C is [n,] row vector)

    G = 1.0
    R_mat = np.eye(m)

    print(f"  System: n={n}, m={m}")
    print(f"  A spectral radius: {np.max(np.abs(np.linalg.eigvals(A))):.4f}")

    # ---- LQR reference via DARE
    # Our Gauss-Newton cost Hessians: l_xx = 2G * C.T@C,  l_uu = 2R  (when dq_du=0)
    Q_dare = 2.0 * G * np.outer(C, C)      # [n, n]
    R_dare = 2.0 * R_mat                    # [m, m]
    try:
        P_dare = scipy.linalg.solve_discrete_are(A, B, Q_dare, R_dare)
    except Exception as exc:
        _fail(f"scipy DARE failed: {exc}")
        return False

    K_lqr = np.linalg.solve(R_dare + B.T @ P_dare @ B, B.T @ P_dare @ A)  # [m, n]
    # Convention: u* = -K_lqr @ x  (standard LQR)
    print(f"  DARE solved. ||K_lqr|| = {np.linalg.norm(K_lqr):.4f}")

    # ---- Set up backward pass inputs
    # Nominal: u=0, x=0 → q_out_k = 0; track q_ref = 1.0 (constant error)
    T = 200
    check_k = 50   # timestep at which to read gains (far from terminal)

    As = [A] * T
    Bs = [B] * T
    dq_dxs = [C.copy() for _ in range(T)]     # ∂q_out/∂x = C [n,]
    dq_dus = [np.zeros(m) for _ in range(T)]   # ∂q_out/∂u = 0 (output doesn't depend on u)
    Q_out_nom = np.zeros(T)
    q_ref = np.ones(T)
    U_nom = np.zeros((T, m))

    # ---- Instantiate ILQRSolver (bypass __init__ to avoid needing real dynamics)
    #      _backward_pass only uses: self.G, self.R, self.G_f, self._n, self._m
    from ilqr import ILQRSolver
    solver = object.__new__(ILQRSolver)
    solver.G = G
    solver.R = R_mat
    solver.G_f = G    # same gain for terminal
    solver._n = n
    solver._m = m

    print(f"  Running backward pass (T={T})...")
    try:
        Ks, ks, success = solver._backward_pass(
            states=[None] * (T + 1),   # not accessed inside loop
            U=U_nom,
            Q_out=Q_out_nom,
            q_ref=q_ref,
            As=As,
            Bs=Bs,
            dq_dxs=dq_dxs,
            dq_dus=dq_dus,
            lam=0.0,
        )
    except Exception as exc:
        _fail(f"_backward_pass raised: {exc}")
        traceback.print_exc()
        return False

    if not success:
        _fail("_backward_pass returned success=False (Q_uu not PD?)")
        return False

    # ---- Compare: K_ilqr at check_k should equal -K_lqr (sign convention)
    K_ilqr = Ks[check_k]  # [m, n]

    # The iLQR gain: delta_u = K_k @ delta_x, where u_nom=0 and x_nom=0 →
    # optimal u = K_k @ x. For LQR regulator: u* = -K_lqr @ x.
    # Therefore at convergence: K_k = -K_lqr.
    K_ilqr_signed = -K_ilqr   # should equal K_lqr

    abs_diff = np.abs(K_ilqr_signed - K_lqr)
    max_K = float(np.max(np.abs(K_lqr))) + 1e-30
    max_rel_err = float(np.max(abs_diff)) / max_K
    rms_rel_err = float(np.sqrt(np.mean((abs_diff / max_K) ** 2)))
    threshold_rel = 0.01   # 1% relative error at k=check_k

    print(f"  ||K_lqr|| = {np.linalg.norm(K_lqr):.4f}")
    print(f"  ||-K_ilqr[k={check_k}]|| = {np.linalg.norm(K_ilqr_signed):.4f}")
    print(f"  Max relative error: {max_rel_err:.4e}   (threshold: {threshold_rel:.0e})")
    print(f"  RMS relative error: {rms_rel_err:.4e}")
    print()
    print(f"  K_lqr (first row):        {K_lqr[0]}")
    print(f"  -K_ilqr[{check_k}] (first row): {K_ilqr_signed[0]}")

    passed = max_rel_err < threshold_rel
    if passed:
        _pass(f"Riccati gains at k={check_k} converged to DARE solution (< {threshold_rel:.0%} error)")
    else:
        _fail(
            f"Max relative error {max_rel_err:.3e} exceeds {threshold_rel:.0e}.\n"
            "  (Consider increasing T or check_k, or verify cost Jacobian signs.)"
        )
    return passed


# ======================================================================
# Argument parsing + main
# ======================================================================

class _DictObj:
    """Convert a JSON dict to attribute-access object."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validation tests for matlab_bridge / lstm_step / ilqr modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tests", default="all",
        help="Comma-separated list of tests to run: '1', '2', '3', or 'all' (default: all)"
    )
    p.add_argument(
        "--data_path",
        default="dataset/LSTM_sim_samples/corner_4065_1000.npz",
        help="Path to .npz file for tests 1 and 2"
    )
    p.add_argument(
        "--ckpt_path", default=None,
        help="Checkpoint .ckpt path (required for test 2)"
    )
    p.add_argument(
        "--config_path", default=None,
        help="JSON config path (required for test 2). "
             "Fields: hidden_size, num_layers, lr [, huber_delta]"
    )
    p.add_argument(
        "--train_list", default="splits/train.txt",
        help="Train split .txt path (for norm stats in test 2)"
    )
    p.add_argument(
        "--data_folder", default="dataset/recursive_samples",
        help="Folder with training .npz files (for norm stats in test 2). "
             "Must contain all files listed in splits/train.txt "
             "(dataset/recursive_samples/, NOT dataset/LSTM_sim_samples/)."
    )
    p.add_argument(
        "--w_nom", type=float, default=0.0029,
        help="Nominal bead width [m] (default 0.0029 m = 2.9 mm)"
    )
    p.add_argument(
        "--n_samples", type=int, default=None,
        help="If given, use only first N timesteps in tests 1 and 2 (for speed)"
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="PyTorch device for test 2"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Parse test selection
    if args.tests.strip().lower() == "all":
        run_tests = {1, 2, 3}
    else:
        try:
            run_tests = {int(t.strip()) for t in args.tests.split(",")}
        except ValueError:
            sys.exit(f"Invalid --tests value: {args.tests!r}. Use '1', '2', '3', or 'all'.")

    data_path = Path(args.data_path)
    results: dict[int, bool | None] = {1: None, 2: None, 3: None}

    # ---- Test 1
    if 1 in run_tests:
        if not data_path.exists():
            print(f"[Test 1] data_path not found: {data_path}")
            _skip("data_path not found")
        else:
            try:
                results[1] = test1_matlab_bridge(data_path, args.w_nom, args.n_samples)
            except Exception as exc:
                print(f"[Test 1] Unexpected error: {exc}")
                traceback.print_exc()
                results[1] = False

    # ---- Test 2
    if 2 in run_tests:
        if args.ckpt_path is None or args.config_path is None:
            _header("Test 2: LSTMStepWrapper.step() vs WalrLSTM batch forward pass")
            _skip("--ckpt_path and --config_path required for Test 2")
        elif not data_path.exists():
            _header("Test 2: LSTMStepWrapper.step() vs WalrLSTM batch forward pass")
            _skip(f"data_path not found: {data_path}")
        else:
            with open(args.config_path) as f:
                cfg = json.load(f)
            if "huber_delta" not in cfg:
                cfg["huber_delta"] = 1.0
            run_config = _DictObj(cfg)
            try:
                results[2] = test2_lstm_step(
                    data_path=data_path,
                    ckpt_path=Path(args.ckpt_path),
                    run_config=run_config,
                    train_list=Path(args.train_list),
                    data_folder=Path(args.data_folder),
                    w_nom=args.w_nom,
                    n_samples=args.n_samples,
                    device=args.device,
                )
            except Exception as exc:
                print(f"[Test 2] Unexpected error: {exc}")
                traceback.print_exc()
                results[2] = False

    # ---- Test 3
    if 3 in run_tests:
        try:
            results[3] = test3_ilqr_riccati()
        except Exception as exc:
            print(f"[Test 3] Unexpected error: {exc}")
            traceback.print_exc()
            results[3] = False

    # ---- Summary
    print(f"\n{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}Summary{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    labels = {
        1: "MatlabBridge.rollout_ode()",
        2: "LSTMStepWrapper.step()",
        3: "ILQRSolver._backward_pass() Riccati",
    }
    n_pass = n_fail = n_skip = 0
    for test_id in sorted(run_tests):
        result = results[test_id]
        label = labels[test_id]
        if result is None:
            print(f"  Test {test_id}: {_YELLOW}SKIP{_RESET}  {label}")
            n_skip += 1
        elif result:
            print(f"  Test {test_id}: {_GREEN}PASS{_RESET}  {label}")
            n_pass += 1
        else:
            print(f"  Test {test_id}: {_RED}FAIL{_RESET}  {label}")
            n_fail += 1

    print()
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
