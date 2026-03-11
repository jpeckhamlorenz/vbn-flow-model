"""
matlab_bridge.py — Persistent MATLAB engine singleton for ODE evaluation.

Wraps VBN_flow_model_solver.m (ode15s) with:
  - Persistent engine lifecycle (start once, reuse across all calls)
  - rollout_ode(): full trajectory in a single MATLAB call (for iLQR forward pass)
  - step_ode(): single-step call for finite-difference linearization

Units:
  time        [s]
  flowrate    [m³/s]
  bead width  [m]
  motor pos   [rad]  (cumulative integral of Q_cmd / EXTRUSION_RATIO)
  angle       [rad]
  ang. vel.   [rad/s]

Do NOT instantiate matlab.engine anywhere else in the project.
"""

from __future__ import annotations

import numpy as np
import scipy.integrate as integrate

import matlab.engine


class MatlabBridge:
    """
    Persistent MATLAB engine wrapper around VBN_flow_model_solver.m.

    Lifecycle::

        bridge = MatlabBridge()
        theta, theta_dot, Q_analytical = bridge.rollout_ode(ts, Q_cmd, w_cmd)
        bridge.quit()

    Constants are pre-computed once at init from the constants/ modules,
    mirroring the logic in flow_predictor_analytical.py:52-86.
    """

    def __init__(
        self,
        fluid: str = "fluid_DOW121",
        mixer: str = "mixer_ISSM50nozzle",
        pump: str = "pump_viscotec_outdated",
    ) -> None:
        """
        Start MATLAB engine and pre-compute ODE constants.

        Args:
            fluid: constants module name (currently only 'fluid_DOW121')
            mixer: constants module name ('mixer_ISSM50nozzle', 'mixer_ISSM160',
                   'mixer_pipe160')
            pump:  constants module name ('pump_viscotec', 'pump_viscotec_outdated')
        """
        # ---- load constants modules (mirrors flow_predictor_analytical.py)
        if fluid == "fluid_DOW121":
            from constants import fluid_DOW121 as fld
        else:
            raise ValueError(f"Unrecognized fluid: {fluid!r}")

        if mixer == "mixer_ISSM50nozzle":
            from constants import mixer_ISSM50nozzle as mix
        elif mixer == "mixer_ISSM160":
            from constants import mixer_ISSM160 as mix
        elif mixer == "mixer_pipe160":
            from constants import mixer_pipe160 as mix
        else:
            raise ValueError(f"Unrecognized mixer: {mixer!r}")

        if pump == "pump_viscotec":
            from constants import pump_viscotec as pmp
        elif pump == "pump_viscotec_outdated":
            from constants import pump_viscotec_outdated as pmp
        else:
            raise ValueError(f"Unrecognized pump: {pump!r}")

        self.extrusion_ratio: float = float(pmp.EXTRUSION_RATIO)

        # ---- build constants array (matches flow_predictor_analytical.py:52-86)
        A_const = (np.pi * pmp.D_C**4 * pmp.R * pmp.RHO_C * pmp.T_C) / (
            16 * pmp.MC_C * pmp.L_C * pmp.M_ROTOR * pmp.R_R**2
        )
        B_const = (-8 * np.pi * pmp.R_SO**2 * pmp.L_S) / (
            pmp.M_ROTOR * (pmp.R_SO**2 - pmp.R_R**2)
        )
        C_const = (-2 * pmp.N_CAV * pmp.A_CAV * np.sin(np.deg2rad(pmp.PHI))) / (
            pmp.M_ROTOR * pmp.R_R
        )
        D_const = (
            (-6 * pmp.RHO_S * pmp.R * pmp.T_S)
            * (pmp.S * pmp.A_RS * pmp.MU_FRIC)
            / (pmp.MC_S * pmp.M_ROTOR * pmp.R_R)
        )
        N_const = (
            128 * fld.K_INDEX * mix.L_NOZ * pmp.EXTRUSION_RATIO**fld.N_INDEX
        ) / (
            3 * fld.N_INDEX * (3 * np.pi)**fld.N_INDEX * mix.KG**(1 - fld.N_INDEX)
        )
        M_const = (
            128
            * fld.K_INDEX
            * mix.KL_SM
            * pmp.EXTRUSION_RATIO**fld.N_INDEX
            * np.pi**(1 - fld.N_INDEX)
        ) / (
            np.pi
            * mix.D_IN**(3 * fld.N_INDEX + 1)
            * (4 * mix.KG)**(1 - fld.N_INDEX)
        )
        U_const = fld.K_INDEX * (
            np.pi * mix.D_MIX**3 / (4 * mix.KG * pmp.EXTRUSION_RATIO)
        ) ** (1 - fld.N_INDEX)

        a = 1.0 / (pmp.EXTRUSION_RATIO * 6e7)
        b = 1000.0
        c = 0.1
        d = 0.01

        self._constants = np.array(
            [A_const, B_const, C_const, D_const,
             N_const, M_const, U_const,
             fld.N_INDEX, mix.D_IN,
             a, b, c, d],
            dtype=np.float64,
        )

        # ---- start persistent MATLAB engine
        print("MatlabBridge: starting MATLAB engine...")
        self._eng = matlab.engine.start_matlab()
        print("MatlabBridge: engine ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rollout_ode(
        self,
        ts: np.ndarray,
        Q_cmd: np.ndarray,
        w_cmd: np.ndarray,
        IC: np.ndarray = np.array([0.0, 0.0]),
        motor_pos_initial: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full trajectory ODE rollout in a single MATLAB call.

        Args:
            ts:    time vector [s], shape (T,)
            Q_cmd: commanded flowrate [m³/s], shape (T,)
            w_cmd: commanded bead width [m], shape (T,)
            IC:    initial conditions [theta_0 (rad), theta_dot_0 (rad/s)]
            motor_pos_initial: initial motor position [rad] at ts[0].
                   Must be consistent with IC[0] (theta_0) to avoid a
                   forcing mismatch in the ODE (forcing ∝ motor_pos − theta).
                   Default 0.0 matches the zero-start case.

        Returns:
            theta_traj:     rotor angle [rad], shape (T,)
            theta_dot_traj: angular velocity [rad/s], shape (T,)
            Q_analytical:   analytical output flowrate [m³/s], shape (T,)
                            Q_analytical[k] = theta_dot_traj[k] * EXTRUSION_RATIO
        """
        ts = np.asarray(ts, dtype=np.float64).ravel()
        Q_cmd = np.asarray(Q_cmd, dtype=np.float64).ravel()
        w_cmd = np.asarray(w_cmd, dtype=np.float64).ravel()
        IC = np.asarray(IC, dtype=np.float64).ravel()  # coerce — engine needs ndarray, not list

        # Cumulative motor angle [rad] — matches flow_predictor_analytical.py:48
        # When starting from a non-zero state (segment 2+), motor_pos_initial
        # must match the theta IC to avoid a forcing mismatch in the ODE
        # (the ODE forcing is proportional to motor_pos − theta).
        input_motor = integrate.cumulative_trapezoid(
            Q_cmd / self.extrusion_ratio, ts, initial=motor_pos_initial
        )

        # Pass numpy arrays directly — matlab.engine converts ndarray → MATLAB double vector.
        # Do NOT use .tolist(): Python lists become MATLAB cell arrays, breaking <= comparisons.
        [t_out, x_out] = self._eng.VBN_flow_model_solver(
            ts,
            input_motor,
            w_cmd,
            IC,
            self._constants,
            nargout=2,
        )

        t_out = np.array(t_out, dtype=np.float64).ravel()
        x_out = np.array(x_out, dtype=np.float64)

        # Interpolate back onto original ts grid
        theta_traj = np.interp(ts, t_out, x_out[:, 0])
        theta_dot_traj = np.interp(ts, t_out, x_out[:, 1])
        Q_analytical = theta_dot_traj * self.extrusion_ratio

        return theta_traj, theta_dot_traj, Q_analytical

    # ------------------------------------------------------------------

    def step_ode(
        self,
        t_k: float,
        dt: float,
        Q_cmd_k: float,
        w_cmd_k: float,
        theta_k: float,
        theta_dot_k: float,
        motor_pos_k: float,
    ) -> tuple[float, float, float]:
        """
        Single-step ODE evaluation from t_k to t_k + dt.

        Used for finite-difference linearization of the ODE state rows.
        The motor position at k+1 is computed analytically (no MATLAB needed):
            motor_pos_{k+1} = motor_pos_k + Q_cmd_k * dt / EXTRUSION_RATIO

        Args:
            t_k:          current time [s]
            dt:           timestep [s]
            Q_cmd_k:      commanded flowrate at step k [m³/s]
            w_cmd_k:      commanded bead width at step k [m]
            theta_k:      rotor angle at step k [rad]
            theta_dot_k:  angular velocity at step k [rad/s]
            motor_pos_k:  cumulative motor position at step k [rad]

        Returns:
            theta_next:      rotor angle at k+1 [rad]
            theta_dot_next:  angular velocity at k+1 [rad/s]
            motor_pos_next:  cumulative motor position at k+1 [rad]
        """
        delta_motor = Q_cmd_k * dt / self.extrusion_ratio

        # Build numpy arrays — matlab.engine converts ndarray → MATLAB double vector.
        # Do NOT pass bare Python lists: they become MATLAB cell arrays.
        t_pair    = np.array([t_k,          t_k + dt],                  dtype=np.float64)
        motor_pair = np.array([motor_pos_k, motor_pos_k + delta_motor], dtype=np.float64)
        bead_pair  = np.array([w_cmd_k,     w_cmd_k],                   dtype=np.float64)
        IC         = np.array([theta_k,     theta_dot_k],               dtype=np.float64)

        [_, x_out] = self._eng.VBN_flow_model_solver(
            t_pair,
            motor_pair,
            bead_pair,
            IC,
            self._constants,
            nargout=2,
        )

        x_out = np.array(x_out, dtype=np.float64)
        theta_next = float(x_out[-1, 0])
        theta_dot_next = float(x_out[-1, 1])
        motor_pos_next = motor_pos_k + delta_motor

        return theta_next, theta_dot_next, motor_pos_next

    # ------------------------------------------------------------------

    def quit(self) -> None:
        """Release the MATLAB process."""
        self._eng.quit()
        print("MatlabBridge: engine quit.")
