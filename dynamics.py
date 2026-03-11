"""
dynamics.py — HybridState and HybridDynamics for iLQR FFEC.

State representation (HybridState):
  theta:      rotor angle [rad]
  theta_dot:  rotor angular velocity [rad/s]
  motor_pos:  cumulative motor position [rad]  (needed for elastic coupling in ODE)
  h:          LSTM hidden state [num_layers, 1, hidden_size]
  c:          LSTM cell state   [num_layers, 1, hidden_size]
  t:          current time [s]

State vector (for iLQR) shape: [3 + 2 * num_layers * hidden_size]
  [theta, theta_dot, motor_pos, h_flat, c_flat]

Control vector: [Q_cmd (m³/s), w_cmd (m)]

Step sequence at timestep k:
  1. Q_analytical_k = theta_dot_k * EXTRUSION_RATIO
  2. (Q_res_k, h_{k+1}, c_{k+1}) = lstm.step(Q_cmd_k, w_cmd_k, Q_analytical_k, h_k, c_k)
  3. q_out_k = Q_analytical_k + Q_res_k
  4. (theta_{k+1}, theta_dot_{k+1}, motor_pos_{k+1}) = bridge.step_ode(...)

For the iLQR forward pass, rollout() uses a single MATLAB call via
bridge.rollout_ode() for efficiency, then threads the LSTM state sequentially.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from matlab_bridge import MatlabBridge
from lstm_step import LSTMStepWrapper


@dataclass
class HybridState:
    """
    Full hybrid model state at a single timestep.

    Fields:
        theta:     rotor angle [rad]
        theta_dot: angular velocity [rad/s]
        motor_pos: cumulative motor angle [rad]
        h:         LSTM hidden state [num_layers, 1, hidden_size]
        c:         LSTM cell state   [num_layers, 1, hidden_size]
        t:         current time [s]
    """

    theta: float
    theta_dot: float
    motor_pos: float
    h: torch.Tensor
    c: torch.Tensor
    t: float

    # ------------------------------------------------------------------

    def to_vec(self) -> np.ndarray:
        """
        Flatten to numpy vector for iLQR arithmetic.

        Layout: [theta, theta_dot, motor_pos, h_flat, c_flat]
        Length: 3 + 2 * num_layers * hidden_size
        """
        h_flat = self.h.detach().cpu().numpy().ravel()
        c_flat = self.c.detach().cpu().numpy().ravel()
        return np.concatenate(
            [[self.theta, self.theta_dot, self.motor_pos], h_flat, c_flat]
        )

    @staticmethod
    def from_vec(
        v: np.ndarray,
        num_layers: int,
        hidden_size: int,
        t: float,
        device: torch.device,
    ) -> "HybridState":
        """
        Reconstruct HybridState from a flat numpy vector.

        Args:
            v:           flat array [3 + 2*num_layers*hidden_size]
            num_layers:  LSTM num_layers
            hidden_size: LSTM hidden_size
            t:           current time [s]
            device:      torch device for h, c tensors
        """
        theta = float(v[0])
        theta_dot = float(v[1])
        motor_pos = float(v[2])
        lstm_sz = num_layers * hidden_size
        h_flat = v[3: 3 + lstm_sz]
        c_flat = v[3 + lstm_sz: 3 + 2 * lstm_sz]
        h = torch.tensor(
            h_flat.reshape(num_layers, 1, hidden_size),
            dtype=torch.float32,
            device=device,
        )
        c = torch.tensor(
            c_flat.reshape(num_layers, 1, hidden_size),
            dtype=torch.float32,
            device=device,
        )
        return HybridState(
            theta=theta, theta_dot=theta_dot, motor_pos=motor_pos, h=h, c=c, t=t
        )

    @property
    def state_dim(self) -> int:
        """Total state vector dimension."""
        lstm_sz = self.h.shape[0] * self.h.shape[2]
        return 3 + 2 * lstm_sz


class HybridDynamics:
    """
    Unified dynamics interface wrapping MatlabBridge + LSTMStepWrapper.

    Provides:
      - step(): single step (MATLAB + LSTM), used for FD linearization
      - rollout(): full trajectory (one MATLAB call + sequential LSTM), used for
                   iLQR forward pass
    """

    def __init__(
        self,
        bridge: MatlabBridge,
        lstm: LSTMStepWrapper,
        dt: float,
        use_lstm: bool = True,
    ) -> None:
        """
        Args:
            bridge:   initialized MatlabBridge (MATLAB engine already started)
            lstm:     initialized LSTMStepWrapper
            dt:       fixed timestep [s]
            use_lstm: if False, skip the LSTM residual correction — outputs only the
                      analytical ODE flow (Q_vbn = theta_dot * ER).  The LSTM is still
                      required for initialisation but is not called during step/rollout.
                      Useful for analytical-only iLQR where the full windowed LSTM
                      prediction is applied separately after optimisation.
        """
        self.bridge = bridge
        self.lstm = lstm
        self.dt = dt
        self.use_lstm = use_lstm
        self._er = bridge.extrusion_ratio

    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Dimension of the flat state vector (3 + 2*num_layers*hidden_size)."""
        return 3 + 2 * self.lstm._num_layers * self.lstm._hidden_size

    @property
    def ctrl_dim(self) -> int:
        """Dimension of the control vector (always 2: [Q_cmd, w_cmd])."""
        return 2

    # ------------------------------------------------------------------

    def make_initial_state(self, t0: float = 0.0) -> HybridState:
        """
        Construct a zero initial state at time t0.

        Returns HybridState with theta=0, theta_dot=0, motor_pos=0,
        zero LSTM hidden/cell state.
        """
        h0, c0 = self.lstm.init_state()
        return HybridState(
            theta=0.0, theta_dot=0.0, motor_pos=0.0, h=h0, c=c0, t=t0
        )

    # ------------------------------------------------------------------

    def step(
        self,
        state: HybridState,
        u: np.ndarray,
    ) -> tuple[HybridState, float]:
        """
        Single dynamics step (scalar, no gradient tracking).

        Args:
            state: current HybridState at time t_k
            u:     control vector [Q_cmd (m³/s), w_cmd (m)], shape (2,)

        Returns:
            next_state: HybridState at time t_{k+1}
            q_out_k:    output flowrate [m³/s] at time k
        """
        Q_cmd_k = float(u[0])
        w_cmd_k = float(u[1])

        # 1. Analytical output from current ODE state
        Q_analytical_k = state.theta_dot * self._er

        # 2. LSTM step (updates h, c; produces residual correction)
        if self.use_lstm:
            Q_res_k, h_next, c_next = self.lstm.step(
                Q_cmd_k, w_cmd_k, Q_analytical_k, state.h, state.c
            )
            q_out_k = Q_analytical_k + Q_res_k
        else:
            # Analytical-only mode: skip LSTM, pass h/c through unchanged
            h_next, c_next = state.h, state.c
            q_out_k = Q_analytical_k

        # 3. ODE step: advance (theta, theta_dot, motor_pos) by one dt
        theta_next, theta_dot_next, motor_pos_next = self.bridge.step_ode(
            t_k=state.t,
            dt=self.dt,
            Q_cmd_k=Q_cmd_k,
            w_cmd_k=w_cmd_k,
            theta_k=state.theta,
            theta_dot_k=state.theta_dot,
            motor_pos_k=state.motor_pos,
        )

        next_state = HybridState(
            theta=theta_next,
            theta_dot=theta_dot_next,
            motor_pos=motor_pos_next,
            h=h_next,
            c=c_next,
            t=state.t + self.dt,
        )
        return next_state, q_out_k

    # ------------------------------------------------------------------

    def rollout(
        self,
        state0: HybridState,
        U: np.ndarray,
    ) -> tuple[list[HybridState], np.ndarray]:
        """
        Full trajectory rollout.

        For efficiency, the ODE is solved in a single MATLAB call via
        bridge.rollout_ode(), then LSTM state is threaded sequentially.

        Args:
            state0: initial HybridState (t0, zero or warm-started)
            U:      controls array [T, 2] where U[k] = [Q_cmd_k (m³/s), w_cmd_k (m)]

        Returns:
            states: list of HybridState, length T+1 (states[0] == state0)
            Q_out:  output flowrates [m³/s], shape (T,)
        """
        T = U.shape[0]
        dt = self.dt

        # Build time vector covering t0 to t0 + T*dt (T+1 points)
        ts = np.array([state0.t + k * dt for k in range(T + 1)], dtype=np.float64)

        # Control arrays extended by one (MATLAB needs ts-length vectors)
        Q_cmd_traj = U[:, 0]
        w_cmd_traj = U[:, 1]
        Q_cmd_full = np.append(Q_cmd_traj, Q_cmd_traj[-1])
        w_cmd_full = np.append(w_cmd_traj, w_cmd_traj[-1])

        # ODE: full trajectory in ONE MATLAB call
        IC = np.array([state0.theta, state0.theta_dot])
        theta_traj, theta_dot_traj, _ = self.bridge.rollout_ode(
            ts, Q_cmd_full, w_cmd_full, IC,
            motor_pos_initial=state0.motor_pos,
        )

        # Cumulative motor position — computed analytically, no MATLAB needed
        motor_traj = np.empty(T + 1, dtype=np.float64)
        motor_traj[0] = state0.motor_pos
        for k in range(T):
            motor_traj[k + 1] = motor_traj[k] + Q_cmd_traj[k] * dt / self._er

        # LSTM: sequential stepping (h/c state is causal)
        states: list[HybridState] = [state0]
        Q_out = np.empty(T, dtype=np.float64)
        h, c = state0.h, state0.c

        for k in range(T):
            Q_analytical_k = float(theta_dot_traj[k]) * self._er

            if self.use_lstm:
                Q_res_k, h_next, c_next = self.lstm.step(
                    float(Q_cmd_traj[k]),
                    float(w_cmd_traj[k]),
                    Q_analytical_k,
                    h,
                    c,
                )
                Q_out[k] = Q_analytical_k + Q_res_k
            else:
                # Analytical-only mode: output is ODE prediction only
                h_next, c_next = h, c
                Q_out[k] = Q_analytical_k

            next_state = HybridState(
                theta=float(theta_traj[k + 1]),
                theta_dot=float(theta_dot_traj[k + 1]),
                motor_pos=float(motor_traj[k + 1]),
                h=h_next,
                c=c_next,
                t=float(ts[k + 1]),
            )
            states.append(next_state)
            h, c = h_next, c_next

        return states, Q_out
