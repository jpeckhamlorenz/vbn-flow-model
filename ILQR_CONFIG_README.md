# iLQR Configuration System

All iLQR-related scripts in this repository share a unified configuration
system built on Python dataclasses. Parameters can be set in three ways, each
overriding the previous:

```
Dataclass defaults  →  JSON config file  →  CLI arguments
   (Tier 1)               (Tier 2)            (Tier 3)
```

**Tier 1** values are hard-coded in `ilqr_config.py` and represent sensible
starting points.  A **Tier 2** JSON file (passed via `--ilqr_config`) overrides
any Tier 1 value it contains.  **Tier 3** CLI flags override everything — only
flags you *explicitly* type on the command line are applied; omitted flags
leave the Tier 1 / Tier 2 value untouched.

---

## Quick Start

```bash
# Run iLQR on the full test set with all defaults (auto-resolves checkpoint):
python test_ilqr_test_set.py

# Quick smoke test on one trajectory:
python test_ilqr_test_set.py --parents corner_60100_1000 --n_samples 200 --max_iter 5

# Use a saved experiment config, but override one param:
python test_ilqr_test_set.py --ilqr_config my_experiment.json --max_iter 50

# End-to-end pipeline validation:
python test_ilqr_pipeline.py --n_samples 300

# Deploy iLQR on an arbitrary trajectory:
python run_ilqr_deploy.py --traj_npz dataset/LSTM_sim_samples/corner_4065_1000.npz
```

---

## Key Files

| File | Purpose |
|------|---------|
| `ilqr_config.py` | Defines `ILQRConfig` dataclass, CLI registration, JSON I/O, and the merge logic |
| `ilqr_defaults.json` | Reference JSON showing all fields — copy and edit to create experiment configs |

---

## Config Groups and Fields

### Checkpoint

Controls how the LSTM model checkpoint is resolved.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sweep_id` | `str` | `"mnywg829"` | WandB sweep ID for auto-resolving best checkpoint |
| `ckpt_path` | `str?` | `null` | Explicit `.ckpt` path (skips WandB lookup) |
| `config_path` | `str?` | `null` | LSTM model config JSON (`hidden_size`, `num_layers`, etc.) |

If `ckpt_path` is `null`, the system calls `get_best_run()` from `models/traj_WALR.py`
using `sweep_id` to find the best checkpoint automatically.

### Data

Paths used to compute LSTM normalisation statistics at runtime.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_list` | `str` | `"splits/train.txt"` | Training split file listing `.npz` filenames |
| `data_folder` | `str` | `"dataset/recursive_samples"` | Folder containing those `.npz` files |

### Physics

Physical system parameters and MATLAB bridge configuration.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `fluid` | `str` | `"fluid_DOW121"` | — | Fluid constants module |
| `mixer` | `str` | `"mixer_ISSM50nozzle"` | — | Mixer constants module |
| `pump` | `str` | `"pump_viscotec_outdated"` | — | Pump constants module |
| `w_nom` | `float` | `0.0029` | m | Nominal bead width |
| `dt` | `float` | `0.01` | s | Simulation timestep |

### Cost

iLQR quadratic cost weights.  The cost at each step is
`G * (q - q_ref)² + u' R u [+ Δu' S Δu]`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `G` | `float` | `1e15` | Tracking cost weight |
| `G_f` | `float?` | `null` | Terminal cost weight (defaults to `G` at runtime) |
| `R_diag` | `[float, float]` | `[1e10, 1e2]` | Control cost `[R_Q, R_w]` |
| `S_diag` | `[float, float]?` | `[1e12, 1e5]` | Rate-of-change penalty `[S_Q, S_w]` |

### Solver

iLQR algorithm hyper-parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_iter` | `int` | `100` | Maximum iLQR iterations per segment |
| `tol` | `float` | `1e-4` | Convergence tolerance (relative cost change) |
| `eps_ode` | `float` | `1e-5` | FD perturbation for ODE state dimensions |
| `eps_ctrl` | `float` | `1e-7` | (Legacy) absolute FD perturbation for controls |
| `eps_ctrl_rel` | `float` | `1e-3` | Relative FD: `eps = max(rel * |u|, floor)` |
| `eps_ctrl_floor_Q` | `float` | `1e-12` | FD floor for Q_cmd [m³/s] |
| `eps_ctrl_floor_w` | `float` | `1e-7` | FD floor for w_cmd [m] |

### Bounds

Control saturation limits and segment sizing.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `Q_min` | `float` | `-9e-8` | m³/s | Minimum Q_cmd |
| `Q_max` | `float` | `1e-7` | m³/s | Maximum Q_cmd |
| `w_min` | `float` | `0.0007` | m | Minimum w_cmd |
| `w_max` | `float` | `0.0029` | m | Maximum w_cmd |
| `w_delta_plus` | `float?` | `0.0003` | m | Max width above W_com |
| `w_delta_minus` | `float?` | `0.0015` | m | Max width below W_com |
| `Q_delta_plus` | `float?` | `null` | m³/s | Max flow above Q_com |
| `Q_delta_minus` | `float?` | `null` | m³/s | Max flow below Q_com |
| `w_rate_max` | `float?` | `null` | mm/s | Max bead width rate |
| `q_rate_max` | `float?` | `null` | mL/min/s | Max flow rate |
| `segment_len` | `int` | `450` | steps | Multi-segment iLQR segment length |

### Run

Experiment-level settings controlling what the iLQR run does.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_samples` | `int` | `500` | Trajectory horizon length [timesteps] |
| `window_len_s` | `float` | `4.5` | Windowed LSTM window length [s] |
| `window_step_s` | `float` | `0.1` | Windowed LSTM window step [s] |
| `use_windowed_cost` | `bool` | `false` | Use deployed windowed LSTM pipeline for cost |
| `analytical_only` | `bool` | `false` | ODE-only mode (skip LSTM in iLQR loop) |

### Display

Output and device settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | `str` | `"cpu"` | PyTorch device (`"cpu"`, `"cuda"`, `"mps"`) |
| `no_show` | `bool` | `false` | Suppress `plt.show()` |
| `save_dir` | `str?` | `null` | Directory for saving PNG figures |

> **Note on device:** The iLQR linearisation uses `torch.func.jacrev` for LSTM
> Jacobians.  The MPS backend crashes in this codepath due to a PyTorch bug, so
> `"cpu"` is the safe default.  The LSTM forward pass is single-step /
> single-batch, so CPU is not a bottleneck — MATLAB ODE calls dominate runtime.

---

## JSON Config Files

Copy `ilqr_defaults.json` and edit the fields you want to change.
You only need to include the groups/fields you want to override — missing
fields keep their dataclass defaults.

**Example: a minimal experiment config** (`my_experiment.json`)

```json
{
  "cost": {
    "G": 1e18,
    "R_diag": [1e8, 1e0]
  },
  "solver": {
    "max_iter": 50
  },
  "run": {
    "n_samples": 1300
  }
}
```

```bash
python test_ilqr_test_set.py --ilqr_config my_experiment.json --parents corner_60100_1000
```

This loads dataclass defaults → overrides with the JSON → any additional CLI
flags would override further.

**Saving a full experiment snapshot:**

```python
from ilqr_config import ILQRConfig, CostConfig, SolverConfig

cfg = ILQRConfig(
    cost=CostConfig(G=1e18, R_diag=[1e8, 1e0]),
    solver=SolverConfig(max_iter=50),
)
cfg.to_json("experiments/high_G_run.json")
```

---

## Programmatic Usage (No CLI)

You can construct an `ILQRConfig` directly in Python — useful for notebooks,
sweep scripts, or importing into other modules:

```python
from ilqr_config import ILQRConfig, CostConfig, SolverConfig, RunConfig

# All defaults
cfg = ILQRConfig()

# Custom cost + solver
cfg = ILQRConfig(
    cost=CostConfig(G=1e16, R_diag=[1e10, 1e2]),
    solver=SolverConfig(max_iter=30),
    run=RunConfig(n_samples=400),
)

# Access fields
print(cfg.cost.G)            # 1e16
print(cfg.solver.max_iter)   # 30
print(cfg.bounds.Q_min)      # -9e-8 (untouched default)

# Load from JSON
cfg = ILQRConfig.from_json("my_experiment.json")

# Save to JSON
cfg.to_json("snapshot.json")
```

---

## Script-Specific Arguments

Each script adds a few arguments that are **not** part of `ILQRConfig` (they
stay in the `argparse.Namespace` returned alongside the config):

| Script | Extra arguments |
|--------|-----------------|
| `test_ilqr_test_set.py` | `--parents`, `--test_list`, `--dt_train` |
| `run_ilqr_deploy.py` | `--traj_npz`, `--out_dir`, `--dt_target`, `--validate` |
| `test_ilqr_pipeline.py` | `--data_path` |
| `validate_ilqr_windowed.py` | positional `controls_npz` |
| `validate_modules.py` | `--tests`, `--data_path`, `--device` |

> **Note on `run_ilqr_deploy.py` input:** `--traj_npz` accepts any `.npz` that contains
> at least `time` and `Q_com`. `Q_vbn`, `Q_res`, and ground-truth keys
> (`Q_sim`/`Q_exp`/`Q_tru`) are optional — if absent, the windowed LSTM reference
> and ground-truth comparisons are skipped automatically.

All scripts accept `--ilqr_config <path>` plus every shared flag listed in the
tables above.  Run any script with `--help` to see the full list.

---

## Common Recipes

```bash
# Run one trajectory with aggressive tracking and save figures
python test_ilqr_test_set.py \
    --parents corner_60100_1000 \
    --G 1e18 --R_diag 1e8 1e0 \
    --max_iter 50 \
    --save_dir outputs/figs --no_show

# Compare analytical-only vs hybrid model
python test_ilqr_test_set.py --parents corner_60100_1000 --analytical_only
python test_ilqr_test_set.py --parents corner_60100_1000

# Use explicit checkpoint instead of WandB auto-resolve
python test_ilqr_pipeline.py \
    --ckpt_path VBN-modeling/abc123/checkpoints/epoch=10.ckpt \
    --config_path config.json

# Shorter segments for faster iteration during development
python test_ilqr_test_set.py --segment_len 100 --n_samples 200 --max_iter 5
```
