"""
ilqr.py — iLQR solver with hybrid FD/autograd linearization.

State vector: x = [theta, theta_dot, motor_pos, h_flat, c_flat]
  Dimension:  n = 3 + 2 * num_layers * hidden_size  (e.g. 771 for 128×3 LSTM)

Control vector: u = [Q_cmd (m³/s), w_cmd (m)]
  Dimension:  m = 2

Linearization strategy (HybridLinearizer):
  A_k [n, n]:
    Rows 0-2 (ODE/motor): central FD using bridge.step_ode() — 6 MATLAB calls
    Rows 3-n (LSTM):      PyTorch autograd via lstm.step_tensor()
  B_k [n, m]:
    Rows 0-2 (ODE/motor): central FD — 4 MATLAB calls
    Rows 3-n (LSTM):      PyTorch autograd
  dq_out/dx, dq_out/du:  PyTorch autograd (q_out = Q_analytical + Q_res)

Total MATLAB calls per linearize() call: 10 (vs ~1540 for full FD)

iLQR algorithm (ILQRSolver.solve()):
  1. Forward rollout → nominal {states, Q_out, cost}
  2. Linearize around nominal trajectory
  3. Backward Riccati pass → gains {K_k, k_k}
  4. Forward pass with line search (Armijo backtracking) → new U
  5. Repeat until convergence or max_iter

Regularization: V_xx is regularized with λI (Levenberg-Marquardt style) if
the backward pass produces a non-PD Q_uu. λ is increased on failure and
decreased on success.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.func

from dynamics import HybridDynamics, HybridState
from cost import stage_derivatives, terminal_derivatives, stage_cost, terminal_cost


# ======================================================================
# Hybrid Linearizer
# ======================================================================

class HybridLinearizer:
    """
    Computes A_k [n, n] and B_k [n, m] at a nominal (state_k, u_k) point.

    Also returns dq_dx [n] and dq_du [m] — the output Jacobians needed for
    cost gradient computation.

    FD perturbation sizes are chosen relative to typical state/control scales.
    """

    def __init__(
        self,
        dyn: HybridDynamics,
        eps_ode: float = 1e-5,
        eps_ctrl: float = 1e-7,
    ) -> None:
        """
        Args:
            dyn:      HybridDynamics instance
            eps_ode:  FD perturbation for ODE state dims (theta, theta_dot, motor_pos)
            eps_ctrl: FD perturbation for control dims (Q_cmd, w_cmd)
        """
        self.dyn = dyn
        self.eps_ode = eps_ode
        self.eps_ctrl = eps_ctrl
        self._er = dyn.bridge.extrusion_ratio
        self._lstm = dyn.lstm
        self._n = dyn.state_dim
        self._m = dyn.ctrl_dim

    # ------------------------------------------------------------------

    def linearize(
        self,
        state_k: HybridState,
        u_k: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize dynamics and output at (state_k, u_k).

        Args:
            state_k: nominal state at step k
            u_k:     nominal control [Q_cmd, w_cmd], shape (2,)

        Returns:
            A_k:   [n, n]  state Jacobian  ∂x_{k+1}/∂x_k
            B_k:   [n, m]  control Jacobian ∂x_{k+1}/∂u_k
            dq_dx: [n,]    output Jacobian ∂q_out/∂x_k
            dq_du: [m,]    output Jacobian ∂q_out/∂u_k
        """
        n, m = self._n, self._m
        A_k = np.zeros((n, n), dtype=np.float64)
        B_k = np.zeros((n, m), dtype=np.float64)
        dq_dx = np.zeros(n, dtype=np.float64)
        dq_du = np.zeros(m, dtype=np.float64)

        Q_cmd_k = float(u_k[0])
        w_cmd_k = float(u_k[1])
        theta_dot_k = state_k.theta_dot

        # Nominal analytical output at step k
        Q_analytical_nom = theta_dot_k * self._er

        # ---- 1. ODE rows (0-2): central FD via bridge.step_ode() --------

        # Perturb theta
        A_k[0:2, 0], A_k[0:2, 1], A_k[0:2, 2] = self._ode_cols_wrt_state(state_k, u_k)

        # motor_pos row: motor_pos_next = motor_pos_k + Q_cmd_k * dt / ER  (analytical)
        A_k[2, 2] = 1.0  # ∂motor_pos_next/∂motor_pos = 1

        # ---- 2. LSTM rows (3-n): autograd --------------------------------

        lstm_jac = self._lstm_jacobians(
            Q_cmd_k, w_cmd_k, Q_analytical_nom, state_k.h, state_k.c
        )
        # Layout: h_flat is indices [3, 3+lstm_sz), c_flat is [3+lstm_sz, n)
        lstm_sz = self._lstm._num_layers * self._lstm._hidden_size
        h_idx = slice(3, 3 + lstm_sz)
        c_idx = slice(3 + lstm_sz, n)

        # ∂(h_next, c_next)/∂theta: 0 (theta doesn't enter LSTM)
        # ∂(h_next, c_next)/∂theta_dot: via Q_analytical = theta_dot * ER
        A_k[h_idx, 1] = lstm_jac["dh_dQanal"] * self._er  # chain rule
        A_k[c_idx, 1] = lstm_jac["dc_dQanal"] * self._er
        # ∂(h_next, c_next)/∂motor_pos: 0
        # ∂(h_next, c_next)/∂h, ∂c
        A_k[h_idx, h_idx] = lstm_jac["dh_dh"]
        A_k[h_idx, c_idx] = lstm_jac["dh_dc"]
        A_k[c_idx, h_idx] = lstm_jac["dc_dh"]
        A_k[c_idx, c_idx] = lstm_jac["dc_dc"]

        # ---- 3. Control Jacobian B_k -------------------------------------

        # Rows 0-1 (theta, theta_dot): FD on Q_cmd and w_cmd
        B_k[0:2, :] = self._ode_cols_wrt_ctrl(state_k, u_k)
        # Row 2 (motor_pos): ∂motor_pos_next/∂Q_cmd = dt/ER, ∂.../∂w_cmd = 0
        B_k[2, 0] = self.dyn.dt / self._er
        B_k[2, 1] = 0.0
        # LSTM rows: autograd
        B_k[h_idx, 0] = lstm_jac["dh_dQcmd"]
        B_k[h_idx, 1] = lstm_jac["dh_dwcmd"]
        B_k[c_idx, 0] = lstm_jac["dc_dQcmd"]
        B_k[c_idx, 1] = lstm_jac["dc_dwcmd"]

        # ---- 4. Output Jacobians dq_dx, dq_du ----------------------------

        # q_out_k = theta_dot_k * ER + Q_res_k
        # ∂q_out/∂theta = 0
        dq_dx[0] = 0.0
        # ∂q_out/∂theta_dot = ER + ∂Q_res/∂theta_dot (via Q_analytical)
        dq_dx[1] = self._er + lstm_jac["dQres_dQanal"] * self._er
        # ∂q_out/∂motor_pos = 0
        dq_dx[2] = 0.0
        # ∂q_out/∂(h, c)
        dq_dx[h_idx] = lstm_jac["dQres_dh"]
        dq_dx[c_idx] = lstm_jac["dQres_dc"]

        # ∂q_out/∂Q_cmd, ∂q_out/∂w_cmd
        dq_du[0] = lstm_jac["dQres_dQcmd"]
        dq_du[1] = lstm_jac["dQres_dwcmd"]

        return A_k, B_k, dq_dx, dq_du

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ode_cols_wrt_state(
        self,
        state_k: HybridState,
        u_k: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Central FD for ODE state columns of A_k.

        Returns three columns of size 2 (rows: theta_next, theta_dot_next):
          col_theta, col_theta_dot, col_motor_pos
        """
        eps = self.eps_ode
        Q_cmd_k = float(u_k[0])
        w_cmd_k = float(u_k[1])

        def _step(theta, theta_dot, motor_pos):
            th, thd, _ = self.dyn.bridge.step_ode(
                t_k=state_k.t,
                dt=self.dyn.dt,
                Q_cmd_k=Q_cmd_k,
                w_cmd_k=w_cmd_k,
                theta_k=theta,
                theta_dot_k=theta_dot,
                motor_pos_k=motor_pos,
            )
            return np.array([th, thd])

        th0 = state_k.theta
        thd0 = state_k.theta_dot
        mp0 = state_k.motor_pos

        col_theta = (_step(th0 + eps, thd0, mp0) - _step(th0 - eps, thd0, mp0)) / (2 * eps)
        col_theta_dot = (_step(th0, thd0 + eps, mp0) - _step(th0, thd0 - eps, mp0)) / (2 * eps)
        col_motor = (_step(th0, thd0, mp0 + eps) - _step(th0, thd0, mp0 - eps)) / (2 * eps)

        return col_theta, col_theta_dot, col_motor

    def _ode_cols_wrt_ctrl(
        self,
        state_k: HybridState,
        u_k: np.ndarray,
    ) -> np.ndarray:
        """
        Central FD for ODE rows (theta, theta_dot) of B_k.

        Returns shape (2, 2): rows = [theta_next, theta_dot_next], cols = [Q_cmd, w_cmd]
        """
        eps = self.eps_ctrl
        Q_cmd_k = float(u_k[0])
        w_cmd_k = float(u_k[1])

        def _step(Q_cmd, w_cmd):
            th, thd, _ = self.dyn.bridge.step_ode(
                t_k=state_k.t,
                dt=self.dyn.dt,
                Q_cmd_k=Q_cmd,
                w_cmd_k=w_cmd,
                theta_k=state_k.theta,
                theta_dot_k=state_k.theta_dot,
                motor_pos_k=state_k.motor_pos,
            )
            return np.array([th, thd])

        col_Qcmd = (_step(Q_cmd_k + eps, w_cmd_k) - _step(Q_cmd_k - eps, w_cmd_k)) / (2 * eps)
        col_wcmd = (_step(Q_cmd_k, w_cmd_k + eps) - _step(Q_cmd_k, w_cmd_k - eps)) / (2 * eps)

        return np.stack([col_Qcmd, col_wcmd], axis=1)  # [2, 2]

    def _lstm_jacobians(
        self,
        Q_cmd: float,
        w_cmd_m: float,
        Q_analytical: float,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """
        Compute all LSTM Jacobians via PyTorch autograd (torch.func.jacrev).

        Returns dict with numpy arrays for all partial derivatives needed
        to assemble A_k, B_k, dq_dx, dq_du.
        """
        lstm = self._lstm
        device = lstm.device
        lstm_sz = lstm._num_layers * lstm._hidden_size

        # Detach h, c to avoid accumulating graphs across timesteps
        h_det = h.detach().clone()
        c_det = c.detach().clone()

        # ---- Define the function to differentiate
        # Inputs: (Q_cmd_s, w_cmd_s, Q_anal_s, h_flat, c_flat) — all float tensors
        # Outputs: (h_next_flat, c_next_flat, Q_res_s)
        def lstm_fn(
            Q_cmd_s: torch.Tensor,
            w_cmd_s: torch.Tensor,
            Q_anal_s: torch.Tensor,
            h_flat: torch.Tensor,
            c_flat: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h_in = h_flat.reshape(lstm._num_layers, 1, lstm._hidden_size)
            c_in = c_flat.reshape(lstm._num_layers, 1, lstm._hidden_size)
            Q_res_t, h_next, c_next = lstm.step_tensor(
                Q_cmd_s, w_cmd_s, Q_anal_s, h_in, c_in
            )
            return h_next.ravel(), c_next.ravel(), Q_res_t.reshape(1)

        # Nominal input tensors
        Q_cmd_t = torch.tensor(Q_cmd, dtype=torch.float32, device=device)
        w_cmd_t = torch.tensor(w_cmd_m, dtype=torch.float32, device=device)
        Q_anal_t = torch.tensor(Q_analytical, dtype=torch.float32, device=device)
        h_flat_t = h_det.ravel()
        c_flat_t = c_det.ravel()

        # ---- jacrev: one call per output group -------------------------
        # argnums=(0,1,2,3,4) gives Jacobians w.r.t. each input
        try:
            jac = torch.func.jacrev(lstm_fn, argnums=(0, 1, 2, 3, 4))(
                Q_cmd_t, w_cmd_t, Q_anal_t, h_flat_t, c_flat_t
            )
        except Exception:
            # Fallback: torch.autograd.functional.jacobian (older API)
            jac = torch.autograd.functional.jacobian(
                lstm_fn,
                (Q_cmd_t, w_cmd_t, Q_anal_t, h_flat_t, c_flat_t),
            )

        # jac is a 3-tuple of 5-tuples: jac[output_idx][input_idx]
        # output 0 = h_next_flat [lstm_sz,], output 1 = c_next_flat, output 2 = Q_res [1,]

        def _np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        # Jacobians of h_next w.r.t. inputs
        dh_dQcmd   = _np(jac[0][0]).ravel()           # [lstm_sz,]
        dh_dwcmd   = _np(jac[0][1]).ravel()           # [lstm_sz,]
        dh_dQanal  = _np(jac[0][2]).ravel()           # [lstm_sz,]
        dh_dh      = _np(jac[0][3]).reshape(lstm_sz, lstm_sz)  # [lstm_sz, lstm_sz]
        dh_dc      = _np(jac[0][4]).reshape(lstm_sz, lstm_sz)

        # Jacobians of c_next w.r.t. inputs
        dc_dQcmd   = _np(jac[1][0]).ravel()
        dc_dwcmd   = _np(jac[1][1]).ravel()
        dc_dQanal  = _np(jac[1][2]).ravel()
        dc_dh      = _np(jac[1][3]).reshape(lstm_sz, lstm_sz)
        dc_dc      = _np(jac[1][4]).reshape(lstm_sz, lstm_sz)

        # Jacobians of Q_res w.r.t. inputs
        dQres_dQcmd  = float(_np(jac[2][0]))
        dQres_dwcmd  = float(_np(jac[2][1]))
        dQres_dQanal = float(_np(jac[2][2]))
        dQres_dh     = _np(jac[2][3]).ravel()         # [lstm_sz,]
        dQres_dc     = _np(jac[2][4]).ravel()

        return {
            "dh_dQcmd":   dh_dQcmd,
            "dh_dwcmd":   dh_dwcmd,
            "dh_dQanal":  dh_dQanal,
            "dh_dh":      dh_dh,
            "dh_dc":      dh_dc,
            "dc_dQcmd":   dc_dQcmd,
            "dc_dwcmd":   dc_dwcmd,
            "dc_dQanal":  dc_dQanal,
            "dc_dh":      dc_dh,
            "dc_dc":      dc_dc,
            "dQres_dQcmd":  dQres_dQcmd,
            "dQres_dwcmd":  dQres_dwcmd,
            "dQres_dQanal": dQres_dQanal,
            "dQres_dh":     dQres_dh,
            "dQres_dc":     dQres_dc,
        }


# ======================================================================
# iLQR Solver
# ======================================================================

class ILQRSolver:
    """
    Iterative Linear Quadratic Regulator (iLQR) with hybrid FD/autograd
    linearization through the hybrid ODE+LSTM dynamics.

    Algorithm:
      while not converged:
        1. Forward rollout with current U → states, Q_out, cost
        2. Linearize at each (state_k, u_k) → A_k, B_k, dq_dx_k, dq_du_k
        3. Backward Riccati pass → gains (K_k, k_k)
        4. Forward pass with Armijo line search → U_new, cost_new
        5. Check convergence: |cost_new - cost_old| / |cost_old| < tol
    """

    def __init__(
        self,
        dynamics: HybridDynamics,
        G: float,
        R: np.ndarray,
        G_f: float,
        max_iter: int = 50,
        tol: float = 1e-4,
        lam_init: float = 1e-4,
        lam_max: float = 1e10,
        lam_factor: float = 10.0,
        alpha_init: float = 1.0,
        n_alpha: int = 10,
        alpha_decay: float = 0.5,
        eps_ode: float = 1e-5,
        eps_ctrl: float = 1e-7,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            dynamics:    HybridDynamics instance
            G:           tracking cost weight (scalar)
            R:           control cost matrix [2, 2]
            G_f:         terminal tracking weight
            max_iter:    maximum iLQR iterations
            tol:         relative cost decrease for convergence
            lam_init:    initial Levenberg-Marquardt regularisation on V_xx
            lam_max:     maximum allowed lambda (signals failure if exceeded)
            lam_factor:  multiply/divide lambda by this on backward failure/success
            alpha_init:  initial line search step (1.0 = full Newton step)
            n_alpha:     number of line search halvings before giving up
            alpha_decay: multiply alpha by this factor each halving (0.5 = halving)
            eps_ode:     FD perturbation for ODE state dims
            eps_ctrl:    FD perturbation for control dims
            verbose:     print iteration progress
        """
        self.dyn = dynamics
        self.G = G
        self.R = np.asarray(R, dtype=np.float64)
        self.G_f = G_f
        self.max_iter = max_iter
        self.tol = tol
        self.lam_init = lam_init
        self.lam_max = lam_max
        self.lam_factor = lam_factor
        self.alpha_init = alpha_init
        self.n_alpha = n_alpha
        self.alpha_decay = alpha_decay
        self.verbose = verbose

        self._linearizer = HybridLinearizer(
            dynamics, eps_ode=eps_ode, eps_ctrl=eps_ctrl
        )
        self._n = dynamics.state_dim
        self._m = dynamics.ctrl_dim

    # ------------------------------------------------------------------

    def solve(
        self,
        state0: HybridState,
        q_ref: np.ndarray,
        U_init: np.ndarray,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        use_windowed_cost: bool = False,
        dt: float | None = None,
        w_rate_max: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """
        Run iLQR to find optimal controls U* that minimise tracking error.

        Args:
            state0:             initial HybridState (zero state or warm start)
            q_ref:              reference flowrate trajectory [m³/s], shape (T,)
            U_init:             initial control guess [T, 2]
            u_min:              lower control bounds [2,] or [T, 2]  (None = no bound)
            u_max:              upper control bounds [2,] or [T, 2]  (None = no bound)
            use_windowed_cost:  if True, adjust the effective reference at each iteration
                                by subtracting the windowed LSTM residual so that the
                                Riccati backward pass optimises the windowed prediction
                                (Q_vbn + Q_res_windowed) toward q_ref rather than the
                                step-mode output.  Requires dt to be set.
            dt:                 timestep [s]  (required when use_windowed_cost=True)
            w_rate_max:         max bead width change per step [m/step]  (None = no limit).
                                Enforced in _forward_pass() as |w[k]-w[k-1]| <= w_rate_max.

        Returns:
            U_opt:      optimal controls [T, 2]
            Q_out_opt:  resulting output flowrate [m³/s], shape (T,).
                        In windowed cost mode this is the windowed prediction
                        (Q_vbn + Q_res_windowed); in standard mode it is the
                        step-mode hybrid output.
            cost_hist:  list of total costs per iLQR iteration
        """
        if use_windowed_cost and dt is None:
            raise ValueError("dt must be provided when use_windowed_cost=True")

        T = len(q_ref)
        U = np.copy(U_init)
        lam = self.lam_init
        cost_hist: list[float] = []

        # Initial forward rollout
        states, Q_out = self.dyn.rollout(state0, U)

        if use_windowed_cost:
            # Q_vbn[k] = ODE angular velocity at t_k × extrusion ratio
            # states[k].theta_dot is the theta_dot used for Q_analytical during step k
            Q_vbn = np.array([states[k].theta_dot * self.dyn._er for k in range(T)])
            # Use U[:, 0] (current Q_cmd) so the LSTM sees the actual command input,
            # not q_ref — this makes the windowed cost sensitive to Q_cmd chattering.
            Q_res_win = self.dyn.lstm.run_windowed(U[:, 0], Q_vbn, U[:, 1], dt)
            q_ref_eff = q_ref - Q_res_win          # adjusted ODE target
            cost = self._total_cost(Q_vbn, q_ref_eff, U)   # = windowed cost
            Q_out_bp = Q_vbn                        # backward pass drives ODE output
            q_ref_bp = q_ref_eff
        else:
            cost = self._total_cost(Q_out, q_ref, U)
            Q_out_bp = Q_out
            q_ref_bp = q_ref

        cost_hist.append(cost)

        if self.verbose:
            mode_str = " [windowed cost]" if use_windowed_cost else ""
            print(f"iLQR init{mode_str}: cost = {cost:.6e}")

        for it in range(self.max_iter):
            # ---- Linearize
            if self.verbose:
                print(f"iLQR iter {it + 1}/{self.max_iter}: linearizing...")

            As, Bs, dq_dxs, dq_dus = self._linearize_trajectory(states, U, T)

            # ---- Backward pass (Riccati) with LM regularization
            # In windowed mode: Q_out_bp = Q_vbn, q_ref_bp = q_ref_eff.
            # This drives gains that minimise G*||Q_vbn - q_ref_eff||^2
            # = G*||Q_pred_windowed - q_ref||^2.
            Ks, ks, success = self._backward_pass(
                states, U, Q_out_bp, q_ref_bp, As, Bs, dq_dxs, dq_dus, lam
            )

            if not success:
                lam *= self.lam_factor
                if lam > self.lam_max:
                    print("iLQR: lambda exceeded max — diverged.")
                    break
                if self.verbose:
                    print(f"  Backward pass failed; increasing lam to {lam:.2e}")
                continue

            # ---- Forward pass with Armijo line search
            # In windowed mode the Armijo check uses the true windowed cost
            # G*||Q_vbn_new + Q_res_win_new - q_ref||^2 + R*||U_new||^2,
            # making it consistent with the outer cost and preventing oscillation.
            U_new, states_new, Q_out_new, cost_proxy, alpha = self._forward_pass(
                state0, states, U, Q_out_bp, Ks, ks, q_ref_bp, u_min, u_max,
                use_windowed_cost=use_windowed_cost,
                q_ref_true=q_ref if use_windowed_cost else None,
                dt_win=dt if use_windowed_cost else None,
                w_rate_max=w_rate_max,
            )

            if alpha is None:
                # Line search failed — increase regularization and retry
                lam *= self.lam_factor
                if self.verbose:
                    print(f"  Line search failed; increasing lam to {lam:.2e}")
                continue

            # ---- Accept step
            U = U_new
            states = states_new
            Q_out = Q_out_new   # windowed prediction in windowed mode, step-mode otherwise
            lam = max(lam / self.lam_factor, 1e-12)

            # ---- Update for next backward pass + record convergence cost
            if use_windowed_cost:
                # cost_proxy is already the true windowed cost (from _forward_pass).
                # Update q_ref_eff for the NEXT backward pass using Q_vbn from accepted states.
                # states[k].theta_dot * ER = Q_analytical at step k = Q_vbn[k].
                Q_vbn = np.array([states[k].theta_dot * self.dyn._er for k in range(T)])
                # Use accepted U[:, 0] so the LSTM sees the real Q_cmd (not q_ref).
                Q_res_win = self.dyn.lstm.run_windowed(U[:, 0], Q_vbn, U[:, 1], dt)
                q_ref_eff = q_ref - Q_res_win
                cost_new = cost_proxy   # true windowed cost, guaranteed < previous by Armijo
                Q_out_bp = Q_vbn
                q_ref_bp = q_ref_eff
            else:
                cost_new = cost_proxy
                Q_out_bp = Q_out  # refresh Armijo baseline to current accepted trajectory

            rel_change = abs(cost_new - cost) / (abs(cost) + 1e-30)
            cost = cost_new
            cost_hist.append(cost)

            if self.verbose:
                print(
                    f"  cost = {cost:.6e}  alpha = {alpha:.3f}  "
                    f"rel_change = {rel_change:.2e}  lam = {lam:.2e}"
                )

            if rel_change < self.tol:
                if self.verbose:
                    print(f"iLQR converged in {it + 1} iterations.")
                break
        else:
            if self.verbose:
                print("iLQR: reached max_iter without convergence.")

        # Final output: windowed prediction if use_windowed_cost, else step-mode
        if use_windowed_cost:
            Q_out_final = Q_vbn + Q_res_win   # windowed prediction at final U
        else:
            Q_out_final = Q_out

        return U, Q_out_final, cost_hist

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _total_cost(
        self,
        Q_out: np.ndarray,
        q_ref: np.ndarray,
        U: np.ndarray,
    ) -> float:
        """Sum stage costs + terminal cost."""
        T = len(q_ref)
        total = sum(
            stage_cost(Q_out[k], q_ref[k], U[k], self.G, self.R)
            for k in range(T)
        )
        total += terminal_cost(Q_out[-1], q_ref[-1], self.G_f)
        return float(total)

    def _linearize_trajectory(
        self,
        states: list[HybridState],
        U: np.ndarray,
        T: int,
    ) -> tuple[list, list, list, list]:
        """Linearize at every timestep along the nominal trajectory."""
        As, Bs, dq_dxs, dq_dus = [], [], [], []
        for k in range(T):
            A_k, B_k, dq_dx_k, dq_du_k = self._linearizer.linearize(states[k], U[k])
            As.append(A_k)
            Bs.append(B_k)
            dq_dxs.append(dq_dx_k)
            dq_dus.append(dq_du_k)
        return As, Bs, dq_dxs, dq_dus

    def _backward_pass(
        self,
        states: list[HybridState],
        U: np.ndarray,
        Q_out: np.ndarray,
        q_ref: np.ndarray,
        As: list[np.ndarray],
        Bs: list[np.ndarray],
        dq_dxs: list[np.ndarray],
        dq_dus: list[np.ndarray],
        lam: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        """
        Standard iLQR backward Riccati pass.

        Returns (Ks, ks, success) where:
          Ks[k]: feedback gain [m, n]
          ks[k]: feedforward gain [m,]
          success: False if Q_uu is not positive definite
        """
        n, m = self._n, self._m
        T = len(U)

        Ks = [None] * T
        ks = [None] * T

        # Terminal value function
        terminal_d = terminal_derivatives(
            Q_out[-1], q_ref[-1], dq_dxs[-1], self.G_f
        )
        V_x = terminal_d["l_x"].copy()    # [n]
        V_xx = terminal_d["l_xx"].copy()  # [n, n]

        for k in reversed(range(T)):
            A_k = As[k]
            B_k = Bs[k]
            stage_d = stage_derivatives(
                Q_out[k], q_ref[k], U[k], dq_dxs[k], dq_dus[k], self.G, self.R
            )
            l_x = stage_d["l_x"]
            l_u = stage_d["l_u"]
            l_xx = stage_d["l_xx"]
            l_uu = stage_d["l_uu"]
            l_xu = stage_d["l_xu"]

            # Q-function approximation
            V_xx_reg = V_xx + lam * np.eye(n)

            Q_x = l_x + A_k.T @ V_x                      # [n]
            Q_u = l_u + B_k.T @ V_x                      # [m]
            Q_xx = l_xx + A_k.T @ V_xx_reg @ A_k         # [n, n]
            Q_uu = l_uu + B_k.T @ V_xx_reg @ B_k         # [m, m]
            Q_xu = l_xu + A_k.T @ V_xx_reg @ B_k         # [n, m]

            # Check Q_uu positive definite
            try:
                Q_uu_chol = np.linalg.cholesky(Q_uu)
            except np.linalg.LinAlgError:
                return Ks, ks, False

            # Gains via Cholesky solve
            Q_uu_inv = np.linalg.inv(Q_uu)
            K_k = -Q_uu_inv @ Q_xu.T     # [m, n]  feedback gain
            k_k = -Q_uu_inv @ Q_u        # [m,]    feedforward gain

            Ks[k] = K_k
            ks[k] = k_k

            # Update value function
            V_x = Q_x + K_k.T @ Q_uu @ k_k + K_k.T @ Q_u + Q_xu @ k_k
            V_xx = Q_xx + K_k.T @ Q_uu @ K_k + K_k.T @ Q_xu.T + Q_xu @ K_k
            # Symmetrize to avoid drift
            V_xx = 0.5 * (V_xx + V_xx.T)

        return Ks, ks, True

    def _forward_pass(
        self,
        state0: HybridState,
        states_nom: list[HybridState],
        U_nom: np.ndarray,
        Q_out_nom: np.ndarray,
        Ks: list[np.ndarray],
        ks: list[np.ndarray],
        q_ref: np.ndarray,
        u_min: np.ndarray | None,
        u_max: np.ndarray | None,
        use_windowed_cost: bool = False,
        q_ref_true: np.ndarray | None = None,
        dt_win: float | None = None,
        w_rate_max: float | None = None,
    ) -> tuple[np.ndarray | None, list | None, np.ndarray | None, float | None, float | None]:
        """
        Forward pass with Armijo line search.

        Control update: u_k = u_nom_k + alpha * k_k + K_k @ (x_k - x_nom_k)

        When use_windowed_cost=True the Armijo condition uses the true windowed
        cost G*||Q_vbn_new + Q_res_windowed_new - q_ref_true||^2 + R*||U_new||^2,
        which is consistent with the cost minimised in solve().  The nominal cost
        Q_out_nom must be Q_vbn (ODE output) and q_ref must be q_ref_eff so that
        cost_nom = G*||Q_vbn - q_ref_eff||^2 = G*||Q_pred_win - q_ref_true||^2.

        Returns (U_new, states_new, Q_out_new, cost_new, alpha) or
                (None, None, None, None, None) if line search fails.
        In windowed cost mode Q_out_new is the windowed prediction Q_vbn+Q_res_win.
        """
        T = len(U_nom)
        alpha = self.alpha_init
        nom_vecs = [s.to_vec() for s in states_nom]
        # cost_nom = windowed cost at nominal (when use_windowed_cost=True,
        # Q_out_nom=Q_vbn and q_ref=q_ref_eff, so this equals G*||Q_pred_win-q_ref||^2)
        cost_nom = self._total_cost(Q_out_nom, q_ref, U_nom)

        for _ in range(self.n_alpha):
            U_new = np.empty_like(U_nom)
            state = state0
            states_new = [state0]
            Q_out_new = np.empty(T)

            h, c = state0.h, state0.c

            # We need to rollout with the new controls — use dynamics.rollout()
            # first build U_new from gains
            x = state0.to_vec()
            for k in range(T):
                delta_x = x - nom_vecs[k]
                u_new_k = U_nom[k] + alpha * ks[k] + Ks[k] @ delta_x
                # Clip to bounds (supports both constant [2] and time-varying [T, 2])
                if u_min is not None:
                    u_new_k = np.maximum(u_new_k, u_min[k] if u_min.ndim > 1 else u_min)
                if u_max is not None:
                    u_new_k = np.minimum(u_new_k, u_max[k] if u_max.ndim > 1 else u_max)
                # Rate-limit bead width: |w[k] - w[k-1]| <= w_rate_max (m/step)
                if w_rate_max is not None:
                    w_prev = U_new[k - 1, 1] if k > 0 else U_nom[0, 1]
                    u_new_k[1] = np.clip(u_new_k[1], w_prev - w_rate_max, w_prev + w_rate_max)
                U_new[k] = u_new_k

                # Step dynamics to get next state for gain computation
                next_state, q_out_k = self.dyn.step(state, u_new_k)
                Q_out_new[k] = q_out_k
                states_new.append(next_state)
                x = next_state.to_vec()
                state = next_state

            if use_windowed_cost:
                # Consistent windowed Armijo: evaluate G*||Q_pred_win_new - q_ref_true||^2
                # Extract Q_vbn from states (states_new[k].theta_dot * ER = Q_analytical at step k)
                Q_vbn_ls = np.array(
                    [states_new[k].theta_dot * self.dyn._er for k in range(T)]
                )
                # Use U_new[:, 0] (proposed Q_cmd) so Armijo sees the true windowed cost
                # at the candidate controls, including any chattering in Q_cmd.
                Q_res_ls = self.dyn.lstm.run_windowed(
                    U_new[:, 0], Q_vbn_ls, U_new[:, 1], dt_win
                )
                Q_pred_win_ls = Q_vbn_ls + Q_res_ls
                cost_new = self._total_cost(Q_pred_win_ls, q_ref_true, U_new)
                if cost_new < cost_nom:
                    return U_new, states_new, Q_pred_win_ls, cost_new, alpha
            else:
                cost_new = self._total_cost(Q_out_new, q_ref, U_new)
                if cost_new < cost_nom:  # Armijo: any decrease accepted
                    return U_new, states_new, Q_out_new, cost_new, alpha

            alpha *= self.alpha_decay

        return None, None, None, None, None
