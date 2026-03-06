"""
lstm_step.py — Step-mode LSTM wrapper for WalrLSTM.

Wraps models/traj_WALR.py::WalrLSTM for single-step inference with explicit
(h, c) hidden state threading. Bypasses pack_padded_sequence by calling
WalrLSTM.lstm (torch.nn.LSTM) directly with shape [1, 1, input_size].

Units (physical, at the interface):
  Q_cmd, Q_analytical, Q_res  [m³/s]
  w_cmd_m                     [m]    (converted to mm internally)

Internally scales by flow_scale=1e9 and bead_scale=1000 to match training.
Normalization stats are recomputed from the train split at init time,
matching DataModule._compute_norm_stats() exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Must match DataModule constants
_FLOW_SCALE: float = 1e9
_BEAD_SCALE: float = 1000.0   # m → mm
_NORM_EPS: float = 1e-8        # DataModule.norm_epsilon
_DEFAULT_BEAD_M: float = 0.0029  # fallback if W_com missing from .npz


def _compute_norm_stats(
    train_list_path: Path,
    data_folder: Path,
) -> dict[str, dict[str, float]]:
    """
    Replicate DataModule._compute_norm_stats() to recover per-channel mean/std.

    Reads the train split file and scans all listed .npz files.
    Returns dict with 'mean' and 'std', each mapping:
      'command', 'bead', 'analytical', 'target'  →  float
    """
    sums = {k: 0.0 for k in ["command", "bead", "analytical", "target"]}
    sqs = {k: 0.0 for k in ["command", "bead", "analytical", "target"]}
    counts = {k: 0 for k in ["command", "bead", "analytical", "target"]}

    with open(train_list_path) as f:
        filenames = [
            line.strip()
            for line in f
            if line.strip()
            and not line.strip().startswith("#")
            and line.strip().endswith(".npz")
        ]

    if not filenames:
        raise RuntimeError(f"No .npz files listed in {train_list_path}")

    for name in filenames:
        d = np.load(data_folder / name)
        command = (_FLOW_SCALE * d["Q_com"]).astype(np.float64)
        analytical = (_FLOW_SCALE * d["Q_vbn"]).astype(np.float64)
        target = (_FLOW_SCALE * d["Q_res"]).astype(np.float64)
        bead = (
            (_BEAD_SCALE * d["W_com"]).astype(np.float64)
            if "W_com" in d
            else np.full_like(command, _BEAD_SCALE * _DEFAULT_BEAD_M)
        )

        for key, arr in [
            ("command", command),
            ("bead", bead),
            ("analytical", analytical),
            ("target", target),
        ]:
            arr = arr.ravel()
            sums[key] += arr.sum()
            sqs[key] += (arr * arr).sum()
            counts[key] += arr.size

    mean = {k: sums[k] / max(counts[k], 1) for k in sums}
    std = {
        k: float(np.sqrt(max(sqs[k] / max(counts[k], 1) - mean[k] ** 2, 0.0)) + _NORM_EPS)
        for k in sums
    }

    return {"mean": mean, "std": std}


class LSTMStepWrapper:
    """
    Single-step inference wrapper around WalrLSTM.

    Usage::

        wrapper = LSTMStepWrapper(ckpt_path, run_config, train_list_path, data_folder)
        h, c = wrapper.init_state()
        for k in range(T):
            Q_res_k, h, c = wrapper.step(Q_cmd[k], w_cmd[k], Q_analytical[k], h, c)

    For autograd-based Jacobians use step_tensor(), which keeps the computation
    graph alive and accepts/returns torch.Tensor inputs.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        run_config,
        train_list_path: str | Path,
        data_folder: str | Path,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            ckpt_path:       path to .ckpt checkpoint file
            run_config:      config object with attributes: hidden_size, num_layers,
                             lr, huber_delta  (matches WandB run_config)
            train_list_path: path to splits/train.txt
            data_folder:     path to folder containing training .npz files
                             listed in splits/train.txt — use dataset/recursive_samples/
                             (NOT dataset/LSTM_sim_samples/, which holds only the 36 base files)
            device:          'cpu', 'cuda', or 'mps'
        """
        from models.traj_WALR import LightningModule

        self.device = torch.device(device)
        self._num_layers: int = run_config.num_layers
        self._hidden_size: int = run_config.hidden_size

        module = LightningModule.load_from_checkpoint(
            Path(ckpt_path), config=run_config
        )
        module.eval()
        module.to(self.device)
        self._net = module.net  # WalrLSTM

        # Norm stats — recomputed from train split, matching training
        print("LSTMStepWrapper: computing norm stats from train split...")
        stats = _compute_norm_stats(Path(train_list_path), Path(data_folder))
        m, s = stats["mean"], stats["std"]

        # Input normalisation [command, bead, analytical]
        self._in_mu = torch.tensor(
            [m["command"], m["bead"], m["analytical"]],
            dtype=torch.float32,
            device=self.device,
        )
        self._in_sd = torch.tensor(
            [s["command"], s["bead"], s["analytical"]],
            dtype=torch.float32,
            device=self.device,
        )
        self._tgt_mu = float(m["target"])
        self._tgt_sd = float(s["target"])
        print("LSTMStepWrapper: ready.")

    # ------------------------------------------------------------------

    def init_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return zero initial hidden and cell states.

        Returns:
            h0, c0: both shape [num_layers, 1, hidden_size], dtype float32
        """
        shape = (self._num_layers, 1, self._hidden_size)
        h0 = torch.zeros(shape, dtype=torch.float32, device=self.device)
        c0 = torch.zeros(shape, dtype=torch.float32, device=self.device)
        return h0, c0

    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        Q_cmd: float,
        w_cmd_m: float,
        Q_analytical: float,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        Single LSTM inference step (no gradient tracking).

        Args:
            Q_cmd:        commanded flowrate [m³/s]
            w_cmd_m:      commanded bead width [m]  (converted to mm internally)
            Q_analytical: analytical output flowrate [m³/s]
            h:            hidden state [num_layers, 1, hidden_size]
            c:            cell state   [num_layers, 1, hidden_size]

        Returns:
            Q_res:   residual flowrate correction [m³/s]
            h_next:  updated hidden state [num_layers, 1, hidden_size]
            c_next:  updated cell state   [num_layers, 1, hidden_size]
        """
        x_raw = torch.tensor(
            [[Q_cmd * _FLOW_SCALE, w_cmd_m * _BEAD_SCALE, Q_analytical * _FLOW_SCALE]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, 1, 3]

        x_norm = (x_raw - self._in_mu[None, None, :]) / (
            self._in_sd[None, None, :] + _NORM_EPS
        )

        out, (h_next, c_next) = self._net.lstm(x_norm, (h, c))
        y_hat_norm = self._net.fc(out).squeeze()  # scalar

        Q_res = float(
            (y_hat_norm * (self._tgt_sd + _NORM_EPS) + self._tgt_mu) / _FLOW_SCALE
        )
        return Q_res, h_next, c_next

    # ------------------------------------------------------------------

    def step_tensor(
        self,
        Q_cmd_t: torch.Tensor,
        w_cmd_m_t: torch.Tensor,
        Q_analytical_t: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tensor version of step() that retains the computation graph for autograd.

        All inputs must be scalar tensors (or broadcastable to [1]).
        The caller is responsible for enabling grad on the relevant inputs.

        Args:
            Q_cmd_t:        commanded flowrate [m³/s], scalar tensor
            w_cmd_m_t:      commanded bead width [m], scalar tensor
            Q_analytical_t: analytical flowrate [m³/s], scalar tensor
            h:              hidden state [num_layers, 1, hidden_size]
            c:              cell state   [num_layers, 1, hidden_size]

        Returns:
            Q_res_t: residual flowrate [m³/s], scalar tensor
            h_next:  [num_layers, 1, hidden_size]
            c_next:  [num_layers, 1, hidden_size]
        """
        x_scaled = torch.stack(
            [Q_cmd_t * _FLOW_SCALE, w_cmd_m_t * _BEAD_SCALE, Q_analytical_t * _FLOW_SCALE]
        ).reshape(1, 1, 3)  # [1, 1, 3]

        x_norm = (x_scaled - self._in_mu[None, None, :]) / (
            self._in_sd[None, None, :] + _NORM_EPS
        )

        out, (h_next, c_next) = self._net.lstm(x_norm, (h, c))
        y_hat_norm = self._net.fc(out).squeeze()  # scalar tensor

        Q_res_t = (y_hat_norm * (self._tgt_sd + _NORM_EPS) + self._tgt_mu) / _FLOW_SCALE
        return Q_res_t, h_next, c_next
