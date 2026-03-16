"""ilqr_config.py — Unified configuration for iLQR FFEC scripts.

Provides a nested dataclass hierarchy with:
  - Sensible defaults matching test_ilqr_test_set.py
  - JSON serialization / deserialization  (to_json / from_json)
  - CLI override support  (add_ilqr_args + load_config)
  - Programmatic construction for notebooks / scripts

Three-tier precedence (lowest → highest):
  1. Dataclass defaults
  2. JSON config file  (--ilqr_config)
  3. CLI arguments      (only those explicitly passed)

Example — programmatic::

    from ilqr_config import ILQRConfig, CostConfig, SolverConfig

    cfg = ILQRConfig(
        cost=CostConfig(G=1e15, R_diag=[1e10, 1e2]),
        solver=SolverConfig(max_iter=50),
    )
    cfg.to_json("my_experiment.json")

Example — CLI::

    python test_ilqr_test_set.py --ilqr_config my_experiment.json --G 1e16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Union


# ======================================================================
#  Sub-config dataclasses
# ======================================================================

@dataclass
class CheckpointConfig:
    """WandB / explicit checkpoint resolution."""
    sweep_id: str = "mnywg829"
    ckpt_path: Optional[str] = None
    config_path: Optional[str] = None      # LSTM model JSON (hidden_size, …)


@dataclass
class DataConfig:
    """Training data paths for norm-stat computation."""
    train_list: str = "splits/train.txt"
    data_folder: str = "dataset/recursive_samples"


@dataclass
class PhysicsConfig:
    """Physical constants module names and nominal values."""
    fluid: str = "fluid_DOW121"
    mixer: str = "mixer_ISSM50nozzle"
    pump: str = "pump_viscotec_outdated"
    w_nom: float = 0.0029                  # nominal bead width [m]
    dt: float = 0.01                       # timestep [s]


@dataclass
class CostConfig:
    """iLQR quadratic cost weights."""
    G: float = 1e15                        # tracking cost weight
    G_f: Optional[float] = None            # terminal cost (defaults to G at runtime)
    R_diag: List[float] = field(default_factory=lambda: [1e10, 1e2])    # [R_Q, R_w]
    S_diag: Optional[List[float]] = field(default_factory=lambda: [1e12, 1e5])  # rate penalty [S_Q, S_w]


@dataclass
class SolverConfig:
    """iLQR solver hyper-parameters."""
    max_iter: int = 200
    tol: float = 1e-4
    eps_ode: float = 1e-5                  # FD perturbation for ODE dims
    eps_ctrl: float = 1e-7                 # (legacy) absolute FD for controls
    eps_ctrl_rel: float = 1e-3             # relative FD: eps = max(rel*|u|, floor)
    eps_ctrl_floor_Q: float = 1e-12        # floor for Q_cmd [m³/s]
    eps_ctrl_floor_w: float = 1e-7         # floor for w_cmd [m]


@dataclass
class BoundsConfig:
    """Control bounds and rate limits."""
    Q_min: float = -20e-9                   # [m³/s]
    Q_max: float = 20e-9                    # [m³/s]
    w_min: float = 0.0007                  # [m]
    w_max: float = 0.0029                  # [m]
    w_delta_plus: Optional[float] = 0.0003    # max width above W_com [m]
    w_delta_minus: Optional[float] = 0.0015   # max width below W_com [m]
    Q_delta_plus: Optional[float] = None      # max flow above Q_com [m³/s]
    Q_delta_minus: Optional[float] = None     # max flow below Q_com [m³/s]
    w_rate_max: Optional[float] = None        # max bead width rate [mm/s]
    q_rate_max: Optional[float] = None        # max flow rate [mL/min/s]
    segment_len: int = 450                     # iLQR segment length [steps]


@dataclass
class RunConfig:
    """Run-mode parameters controlling the experiment."""
    n_samples: int = 14000                   # iLQR horizon length [timesteps]
    window_len_s: float = 4.5             # windowed LSTM window length [s]
    window_step_s: float = 0.1            # windowed LSTM window step [s]
    use_windowed_cost: bool = False        # target deployed windowed pipeline
    analytical_only: bool = False          # ODE-only (no LSTM in iLQR)


@dataclass
class DisplayConfig:
    """Output / display settings."""
    device: str = "cpu"                    # PyTorch device
    no_show: bool = False                  # skip plt.show()
    save_dir: Optional[str] = None         # directory for PNG output


# ======================================================================
#  Top-level config
# ======================================================================

@dataclass
class ILQRConfig:
    """Complete iLQR FFEC experiment configuration."""
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    run: RunConfig = field(default_factory=RunConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    # -- JSON I/O -------------------------------------------------------

    def to_json(self, path: Union[str, Path]) -> None:
        """Serialize config to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> ILQRConfig:
        """Load config from a JSON file, ignoring unknown keys."""
        with open(path) as f:
            d = json.load(f)
        return cls._from_nested_dict(d)

    @classmethod
    def _from_nested_dict(cls, d: dict) -> ILQRConfig:
        """Reconstruct from a nested dict, ignoring unknown keys."""
        def _build(dc_cls, sub: dict):
            valid = {f.name for f in fields(dc_cls)}
            return dc_cls(**{k: v for k, v in sub.items() if k in valid})

        return cls(
            checkpoint=_build(CheckpointConfig, d.get("checkpoint", {})),
            data=_build(DataConfig, d.get("data", {})),
            physics=_build(PhysicsConfig, d.get("physics", {})),
            cost=_build(CostConfig, d.get("cost", {})),
            solver=_build(SolverConfig, d.get("solver", {})),
            bounds=_build(BoundsConfig, d.get("bounds", {})),
            run=_build(RunConfig, d.get("run", {})),
            display=_build(DisplayConfig, d.get("display", {})),
        )


# ======================================================================
#  Flat CLI name → (sub-config group, field name) mapping
# ======================================================================

_FLAT_MAP: dict[str, tuple[str, str]] = {
    # CheckpointConfig
    "sweep_id":          ("checkpoint", "sweep_id"),
    "ckpt_path":         ("checkpoint", "ckpt_path"),
    "config_path":       ("checkpoint", "config_path"),
    # DataConfig
    "train_list":        ("data",       "train_list"),
    "data_folder":       ("data",       "data_folder"),
    # PhysicsConfig
    "fluid":             ("physics",    "fluid"),
    "mixer":             ("physics",    "mixer"),
    "pump":              ("physics",    "pump"),
    "w_nom":             ("physics",    "w_nom"),
    "dt":                ("physics",    "dt"),
    # CostConfig
    "G":                 ("cost",       "G"),
    "G_f":               ("cost",       "G_f"),
    "R_diag":            ("cost",       "R_diag"),
    "S_diag":            ("cost",       "S_diag"),
    # SolverConfig
    "max_iter":          ("solver",     "max_iter"),
    "tol":               ("solver",     "tol"),
    "eps_ode":           ("solver",     "eps_ode"),
    "eps_ctrl":          ("solver",     "eps_ctrl"),
    "eps_ctrl_rel":      ("solver",     "eps_ctrl_rel"),
    "eps_ctrl_floor_Q":  ("solver",     "eps_ctrl_floor_Q"),
    "eps_ctrl_floor_w":  ("solver",     "eps_ctrl_floor_w"),
    # BoundsConfig
    "Q_min":             ("bounds",     "Q_min"),
    "Q_max":             ("bounds",     "Q_max"),
    "w_min":             ("bounds",     "w_min"),
    "w_max":             ("bounds",     "w_max"),
    "w_delta_plus":      ("bounds",     "w_delta_plus"),
    "w_delta_minus":     ("bounds",     "w_delta_minus"),
    "Q_delta_plus":      ("bounds",     "Q_delta_plus"),
    "Q_delta_minus":     ("bounds",     "Q_delta_minus"),
    "w_rate_max":        ("bounds",     "w_rate_max"),
    "q_rate_max":        ("bounds",     "q_rate_max"),
    "segment_len":       ("bounds",     "segment_len"),
    # RunConfig
    "n_samples":         ("run",        "n_samples"),
    "window_len_s":      ("run",        "window_len_s"),
    "window_step_s":     ("run",        "window_step_s"),
    "use_windowed_cost": ("run",        "use_windowed_cost"),
    "analytical_only":   ("run",        "analytical_only"),
    # DisplayConfig
    "device":            ("display",    "device"),
    "no_show":           ("display",    "no_show"),
    "save_dir":          ("display",    "save_dir"),
}


# ======================================================================
#  argparse integration
# ======================================================================

def add_ilqr_args(parser: argparse.ArgumentParser) -> None:
    """Register all ILQRConfig fields as argparse arguments.

    Every argument uses ``default=argparse.SUPPRESS`` so that only
    explicitly-passed CLI values appear in ``vars(parse_args())``.
    This enables clean three-tier merging in :func:`load_config`.
    """
    _S = argparse.SUPPRESS  # shorthand

    # -- Checkpoint --
    grp = parser.add_argument_group("Checkpoint")
    grp.add_argument("--sweep_id", default=_S,
                     help="WandB sweep ID for get_best_run() (default: mnywg829)")
    grp.add_argument("--ckpt_path", default=_S,
                     help="Explicit .ckpt path")
    grp.add_argument("--config_path", default=_S,
                     help="LSTM model JSON config (with --ckpt_path)")

    # -- Data --
    grp = parser.add_argument_group("Data")
    grp.add_argument("--train_list", default=_S,
                     help="Path to training split .txt (default: splits/train.txt)")
    grp.add_argument("--data_folder", default=_S,
                     help="Folder with training .npz files (default: dataset/recursive_samples)")

    # -- Physics --
    grp = parser.add_argument_group("Physics")
    grp.add_argument("--fluid", default=_S, help="Fluid constants module")
    grp.add_argument("--mixer", default=_S, help="Mixer constants module")
    grp.add_argument("--pump", default=_S, help="Pump constants module")
    grp.add_argument("--w_nom", type=float, default=_S,
                     help="Nominal bead width [m] (default: 0.0029)")
    grp.add_argument("--dt", type=float, default=_S,
                     help="Timestep [s] (default: 0.01)")

    # -- Cost --
    grp = parser.add_argument_group("iLQR cost")
    grp.add_argument("--G", type=float, default=_S,
                     help="Tracking cost weight (default: 1e14)")
    grp.add_argument("--G_f", type=float, default=_S,
                     help="Terminal cost weight (default: same as --G)")
    grp.add_argument("--R_diag", nargs=2, type=float, default=_S,
                     metavar=("R_Q", "R_w"),
                     help="Control cost diagonal [R_Q, R_w] (default: 1e-3 1e-3)")
    grp.add_argument("--S_diag", nargs=2, type=float, default=_S,
                     metavar=("S_Q", "S_w"),
                     help="Rate penalty diagonal [S_Q, S_w]. "
                          "Penalises Σ (u[k]-u[k-1])^T diag(S) (u[k]-u[k-1]). "
                          "Default: None (no rate penalty).")

    # -- Solver --
    grp = parser.add_argument_group("iLQR solver")
    grp.add_argument("--max_iter", type=int, default=_S,
                     help="Max iLQR iterations (default: 10)")
    grp.add_argument("--tol", type=float, default=_S,
                     help="Convergence tolerance (default: 1e-4)")
    grp.add_argument("--eps_ode", type=float, default=_S,
                     help="FD perturbation for ODE dims (default: 1e-5)")
    grp.add_argument("--eps_ctrl", type=float, default=_S,
                     help="(legacy) Absolute FD perturbation for controls (default: 1e-7)")
    grp.add_argument("--eps_ctrl_rel", type=float, default=_S,
                     help="Relative FD perturbation: eps = max(rel*|u|, floor) (default: 1e-3)")
    grp.add_argument("--eps_ctrl_floor_Q", type=float, default=_S,
                     help="Floor FD perturbation for Q_cmd [m³/s] (default: 1e-12)")
    grp.add_argument("--eps_ctrl_floor_w", type=float, default=_S,
                     help="Floor FD perturbation for w_cmd [m] (default: 1e-7)")

    # -- Bounds --
    grp = parser.add_argument_group("Control bounds")
    grp.add_argument("--Q_min", type=float, default=_S, help="Min Q_cmd [m³/s]")
    grp.add_argument("--Q_max", type=float, default=_S, help="Max Q_cmd [m³/s]")
    grp.add_argument("--w_min", type=float, default=_S, help="Min w_cmd [m]")
    grp.add_argument("--w_max", type=float, default=_S, help="Max w_cmd [m]")
    grp.add_argument("--w_delta_plus", type=float, default=_S,
                     help="Max bead width ABOVE W_com[k] [m]")
    grp.add_argument("--w_delta_minus", type=float, default=_S,
                     help="Max bead width BELOW W_com[k] [m]")
    grp.add_argument("--Q_delta_plus", type=float, default=_S,
                     help="Max flow command ABOVE Q_com[k] [m³/s]")
    grp.add_argument("--Q_delta_minus", type=float, default=_S,
                     help="Max flow command BELOW Q_com[k] [m³/s]")
    grp.add_argument("--w_rate_max", type=float, default=_S,
                     help="Max bead width change rate [mm/s]")
    grp.add_argument("--q_rate_max", type=float, default=_S,
                     help="Max flow rate change [mL/min/s]")
    grp.add_argument("--segment_len", type=int, default=_S,
                     help="Segment length for multi-segment iLQR [steps]")

    # -- Run --
    grp = parser.add_argument_group("Run mode")
    grp.add_argument("--n_samples", type=int, default=_S,
                     help="iLQR horizon length [timesteps] (default: 200)")
    grp.add_argument("--window_len_s", type=float, default=_S,
                     help="Window length [s] for windowed LSTM (default: 4.5)")
    grp.add_argument("--window_step_s", type=float, default=_S,
                     help="Window step [s] for windowed LSTM (default: 0.1)")
    grp.add_argument("--use_windowed_cost", action="store_true", default=_S,
                     help="Use windowed LSTM cost for iLQR")
    grp.add_argument("--analytical_only", action="store_true", default=_S,
                     help="ODE-only iLQR (no LSTM)")

    # -- Display --
    grp = parser.add_argument_group("Display")
    grp.add_argument("--device", default=_S, choices=["cpu", "cuda", "mps"],
                     help="PyTorch device (default: cpu)")
    grp.add_argument("--no_show", action="store_true", default=_S,
                     help="Skip plt.show()")
    grp.add_argument("--save_dir", default=_S,
                     help="Directory for PNG output")


def load_config(
    args: argparse.Namespace | None = None,
    ilqr_config_path: str | Path | None = None,
) -> ILQRConfig:
    """Build an :class:`ILQRConfig` with three-tier precedence.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Parsed CLI arguments.  Only keys that were explicitly passed on the
        command line should be present (use ``default=argparse.SUPPRESS``
        when registering arguments with :func:`add_ilqr_args`).
    ilqr_config_path : str or Path, optional
        Path to a JSON config file (tier 2 defaults).

    Returns
    -------
    ILQRConfig
        Merged configuration.
    """
    # Tier 1: dataclass defaults
    cfg = ILQRConfig()

    # Tier 2: JSON file overrides
    if ilqr_config_path is not None:
        cfg = ILQRConfig.from_json(ilqr_config_path)

    # Tier 3: CLI overrides (only explicitly-passed keys)
    if args is not None:
        cli_dict = vars(args)
        for flat_key, value in cli_dict.items():
            if flat_key not in _FLAT_MAP:
                continue  # script-specific arg, not part of ILQRConfig
            group_name, field_name = _FLAT_MAP[flat_key]
            sub = getattr(cfg, group_name)
            setattr(sub, field_name, value)

    return cfg
