# flow_predictor_lstm.py
# Deployment: analytical VBN + normalized windowed residual LSTM + stitching

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch

from flow_predictor_analytical import flow_predictor as flow_predictor_analytical


@dataclass
class WindowParams:
    # Choose either *_s (seconds) OR *_n (samples).
    window_len_s: float = 4.5
    window_step_s: float = 0.1
    window_len_n: Optional[int] = None
    window_step_n: Optional[int] = None
    include_tail: bool = True


def _as_float(x) -> float:
    """Tensor/np scalar -> python float."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return float(x)


def _extract_norm(norm_stats: Dict) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Robustly extract:
      - input_mean: shape (3,)
      - input_std:  shape (3,)
      - target_mean: scalar
      - target_std:  scalar

    Supports common layouts:
      A) norm_stats["mean"]["input"] (len 3), norm_stats["std"]["input"]
      B) norm_stats["mean"]["x"] etc.
      C) per-feature keys: command/bead/analytical + target
    """
    mean = norm_stats["mean"]
    std = norm_stats["std"]

    # Case A / B: vector input stats
    for key in ["input", "inputs", "x", "X", "features"]:
        if key in mean and key in std:
            mu = mean[key]
            sd = std[key]
            mu = np.asarray(mu.detach().cpu().numpy() if hasattr(mu, "detach") else mu, dtype=np.float32).reshape(-1)
            sd = np.asarray(sd.detach().cpu().numpy() if hasattr(sd, "detach") else sd, dtype=np.float32).reshape(-1)
            if mu.shape[0] != 3 or sd.shape[0] != 3:
                raise ValueError(f"Expected 3 input features, got mean {mu.shape}, std {sd.shape}")
            t_mu = _as_float(mean.get("target", mean.get("y", mean.get("Y"))))
            t_sd = _as_float(std.get("target", std.get("y", std.get("Y"))))
            return mu, sd, t_mu, t_sd

    # Case C: per-feature keys
    # NOTE: naming depends on your DataModule implementation
    feat_keys = None
    for candidate in [
        ("command", "bead", "analytical"),
        ("Q_com", "W_com", "Q_vbn"),
        ("cmd", "bead", "vbn"),
    ]:
        if all(k in mean for k in candidate) and all(k in std for k in candidate):
            feat_keys = candidate
            break

    if feat_keys is None:
        raise KeyError(
            "Could not infer input normalization keys from norm_stats. "
            "Expected mean/std to contain either an input vector (key 'input'/'x') "
            "or per-feature keys like ('command','bead','analytical')."
        )

    mu = np.array([_as_float(mean[k]) for k in feat_keys], dtype=np.float32)
    sd = np.array([_as_float(std[k]) for k in feat_keys], dtype=np.float32)

    t_mu = _as_float(mean.get("target", mean.get("y", mean.get("Y"))))
    t_sd = _as_float(std.get("target", std.get("y", std.get("Y"))))
    return mu, sd, t_mu, t_sd


def _make_windows(time: np.ndarray, N: int, win: WindowParams) -> Tuple[list[int], list[int]]:
    if win.window_len_n is not None and win.window_step_n is not None:
        L = int(win.window_len_n)
        S = int(win.window_step_n)
    else:
        dt = float(np.median(np.diff(time)))
        if dt <= 0:
            raise ValueError("Non-positive dt inferred from time array")
        L = int(round(win.window_len_s / dt))
        S = int(round(win.window_step_s / dt))

    if L <= 1:
        raise ValueError(f"window_len too small (samples): {L}")
    if S <= 0:
        raise ValueError(f"window_step must be >=1 sample, got {S}")

    starts = list(range(0, max(N - L + 1, 0), S))

    if win.include_tail:
        last_start = max(N - L, 0)
        if len(starts) == 0 or starts[-1] != last_start:
            starts.append(last_start)

    lens = [min(L, N - s) for s in starts]
    return starts, lens


@torch.no_grad()
def flow_predictor_lstm_windowed(
    time_np: np.ndarray,
    command_np: np.ndarray,
    bead_np: np.ndarray,
    model_type: str,
    ckpt_path: Union[str, Path],
    run_config,
    norm_stats: Dict,
    win: WindowParams = WindowParams(),
    device_type: str = "mps",
    flow_scale: float = 1e9,
    bead_units: str = "m",  # "m" or "mm"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Q_pred (m^3/s), Q_vbn (m^3/s), Q_res_pred (m^3/s)
    """

    # ---- input checks
    time = np.asarray(time_np).reshape(-1)
    Q_com = np.asarray(command_np).reshape(-1)
    W_com = np.asarray(bead_np).reshape(-1)

    if not (len(time) == len(Q_com) == len(W_com)):
        raise ValueError("time_np, command_np, bead_np must have equal length")

    N = len(time)
    if N == 0:
        raise ValueError("Empty input time series")

    # ---- analytical series
    # keep your convention: analytical uses bead/10.0
    _, _, _, Q_vbn = flow_predictor_analytical(time, Q_com, W_com / 10.0)
    Q_vbn = np.asarray(Q_vbn).reshape(-1)

    # ---- load model
    assert device_type in ["cpu", "cuda", "mps"], "device_type must be 'cpu', 'cuda', or 'mps'"
    device = torch.device(device_type)

    if model_type == "WALR":
        from models.traj_WALR import LightningModule
    else:
        raise ValueError(f"Model type '{model_type}' not supported here (expected 'WALR').")

    module = LightningModule.load_from_checkpoint(Path(ckpt_path), config=run_config)
    module.to(device)
    module.eval()

    # ---- normalization stats (must match training)
    in_mu, in_sd, tgt_mu, tgt_sd = _extract_norm(norm_stats)

    # print("input mean/std used:", in_mu, in_sd)
    # print("target mean/std used:", tgt_mu, tgt_sd)

    # ---- window plan
    starts, lens = _make_windows(time, N, win)

    # ---- stitch accumulators (residual in physical units m^3/s)
    acc = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.float64)

    # helper: bead feature in mm like training
    if bead_units == "m":
        bead_feat = W_com * 1000.0
    elif bead_units == "mm":
        bead_feat = W_com
    else:
        raise ValueError("bead_units must be 'm' or 'mm'")

    for s, L in zip(starts, lens):
        sl = slice(s, s + L)

        # Build features in the SAME space used for computing norm_stats during training.
        # Your training used:
        #   command = flow_scale * Q_com
        #   analytical = flow_scale * Q_vbn
        #   bead = 1000 * W_com  (mm)
        x0 = (Q_com[sl] * flow_scale).astype(np.float32)
        x1 = (bead_feat[sl]).astype(np.float32)
        x2 = (Q_vbn[sl] * flow_scale).astype(np.float32)
        x = np.stack([x0, x1, x2], axis=1)  # [L, 3]

        # normalize inputs
        x = (x - in_mu[None, :]) / (in_sd[None, :] + 1e-12)

        x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # [1, L, 3]
        len_t = torch.tensor([L], dtype=torch.long, device=device)

        # model output = normalized target residual (by your statement)
        y_hat_norm = module.net(x_t, len_t)[:, :, 0].squeeze(0)  # [L]
        y_hat_norm = y_hat_norm.detach().cpu().numpy()

        # de-normalize target back to scaled residual
        y_hat_scaled = y_hat_norm * (tgt_sd + 1e-12) + tgt_mu  # still in "scaled" units

        # convert to physical residual (m^3/s)
        y_hat_phys = y_hat_scaled / flow_scale

        acc[sl] += y_hat_phys
        cnt[sl] += 1.0

    res_pred = (acc / np.clip(cnt, 1.0, None)).astype(np.float32)
    Q_pred = (Q_vbn + res_pred).astype(np.float32)

    # print how many windows there are in total
    # print(f"Total windows processed: {len(starts)}")

    # plot all the windows for debugging
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 5))
    # for s, L in zip(starts, lens):
    #     sl = slice(s, s + L)
    #     plt.plot(time[sl], Q_vbn[sl] + res_pred[sl], alpha=0.3)




    return Q_pred, Q_vbn.astype(np.float32), res_pred