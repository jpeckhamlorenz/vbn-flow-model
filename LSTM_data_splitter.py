from __future__ import annotations

import re
import random
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Iterable, Set


@dataclass
class SplitOverrides:
    """
    Optional manual overrides to force specific parent trajectories into a split.
    Keys are parent IDs (not filenames).
    """
    train: Set[str] = None
    val: Set[str] = None
    test: Set[str] = None

    def __post_init__(self):
        self.train = set() if self.train is None else set(self.train)
        self.val = set() if self.val is None else set(self.val)
        self.test = set() if self.test is None else set(self.test)

    def all_forced(self) -> Set[str]:
        return self.train | self.val | self.test

    def check_disjoint(self):
        overlap = (self.train & self.val) | (self.train & self.test) | (self.val & self.test)
        if overlap:
            raise ValueError(f"Override parent IDs appear in multiple splits: {sorted(overlap)[:20]}")


def infer_type_from_parent(parent_id: str) -> str:
    """
    Determine trajectory type from parent_id prefix.
    """
    if parent_id.startswith("corner"):
        return "corner"
    if parent_id.startswith("flowrate"):
        return "flowrate"
    if parent_id.startswith("twostep"):
        return "twostep"
    return "other"


def extract_parent_id(filename: str) -> str:
    """
    Extract parent trajectory ID from window filename.

    Expected pattern:
        <parent>_window_<index>.npz

    Examples:
        corner_4065_100_window_0.npz
        flowrate_pattern_02_averaged_smoothed_window_550.npz
        twostep_21_[1.0_0.5_0.0]_window_250.npz
    """
    stem = Path(filename).stem  # remove .npz

    token = "_window_"
    if token not in stem:
        raise ValueError(f"Filename does not contain '{token}': {filename}")

    return stem.split(token)[0]


def grouped_stratified_split(
    windows_dir: Path,
    splits_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.20,
    test_frac: float = 0.10,
    seed: int = 0,
    overrides: Optional[SplitOverrides] = None,
    file_glob: str = "*.npz",
    write_filenames_relative: bool = True,
) -> Dict[str, List[str]]:
    """
    Create train/val/test splits by parent trajectory (grouped) and stratified by type prefixes.

    - windows_dir: directory containing ALL window .npz files
    - splits_dir: directory where train.txt / val.txt / test.txt will be written
    - overrides: optional forced parent IDs into specific splits
    - Returns dict with keys 'train', 'val', 'test' mapping to lists of filenames (one per line)

    NOTE: Fractions are applied *within each type group*, on parent IDs (not windows).
    """
    windows_dir = Path(windows_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if overrides is None:
        overrides = SplitOverrides()
    overrides.check_disjoint()

    # sanity: fractions
    s = train_frac + val_frac + test_frac
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"Fractions must sum to 1.0; got {s}")

    # gather window files
    all_files = sorted([p.name for p in windows_dir.glob(file_glob)])
    if not all_files:
        raise RuntimeError(f"No files matching {file_glob} found in {windows_dir}")

    # group window filenames by parent trajectory
    parent_to_files: Dict[str, List[str]] = defaultdict(list)
    for fname in all_files:
        pid = extract_parent_id(fname)
        parent_to_files[pid].append(fname)

    # group parent IDs by type
    type_to_parents: Dict[str, List[str]] = defaultdict(list)
    for pid in parent_to_files.keys():
        t = infer_type_from_parent(pid)
        if t == "other":
            # If you truly have only three types, you might want to error here:
            # raise ValueError(f"Unrecognized parent prefix for parent_id={pid}")
            pass
        type_to_parents[t].append(pid)

    rng = random.Random(seed)

    # output parent assignments
    train_parents: Set[str] = set()
    val_parents: Set[str] = set()
    test_parents: Set[str] = set()

    # apply overrides first
    train_parents |= overrides.train
    val_parents |= overrides.val
    test_parents |= overrides.test

    forced = overrides.all_forced()

    # split within each type
    for t, parents in type_to_parents.items():
        # only split recognized types; you can skip 'other' or include it
        if t not in ("corner", "flowrate", "twostep"):
            continue

        # remove forced parents from this type pool
        pool = [p for p in parents if p not in forced]
        rng.shuffle(pool)

        n = len(pool)
        # round to nearest counts; ensure sum matches
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        n_test = n - n_train - n_val

        # distribute
        train_parents.update(pool[:n_train])
        val_parents.update(pool[n_train:n_train + n_val])
        test_parents.update(pool[n_train + n_val:])

    # final sanity: no overlap
    overlap = (train_parents & val_parents) | (train_parents & test_parents) | (val_parents & test_parents)
    if overlap:
        raise RuntimeError(f"Split overlap detected (should be impossible): {sorted(overlap)[:20]}")

    # build filename lists per split
    def filenames_for(parents_set: Set[str]) -> List[str]:
        out = []
        for pid in sorted(parents_set):
            out.extend(sorted(parent_to_files[pid]))
        return out

    split_files = {
        "train": filenames_for(train_parents),
        "val": filenames_for(val_parents),
        "test": filenames_for(test_parents),
    }

    # write txt files
    def write_list(path: Path, names: List[str]):
        with open(path, "w") as f:
            for name in names:
                # by default write just the filename (relative to windows_dir)
                f.write(name + "\n")

    train_path = splits_dir / "train.txt"
    val_path = splits_dir / "val.txt"
    test_path = splits_dir / "test.txt"

    write_list(train_path, split_files["train"])
    write_list(val_path, split_files["val"])
    write_list(test_path, split_files["test"])

    # helpful summary printout
    def count_parents(parents_set: Set[str]) -> Dict[str, int]:
        c = {"corner": 0, "flowrate": 0, "twostep": 0, "other": 0}
        for pid in parents_set:
            c[infer_type_from_parent(pid)] += 1
        return c

    summary = {
        "n_total_files": len(all_files),
        "n_total_parents": len(parent_to_files),
        "train_files": len(split_files["train"]),
        "val_files": len(split_files["val"]),
        "test_files": len(split_files["test"]),
        "train_parents_by_type": count_parents(train_parents),
        "val_parents_by_type": count_parents(val_parents),
        "test_parents_by_type": count_parents(test_parents),
        "forced_parents": sorted(forced),
        "seed": seed,
        "splits_dir": str(splits_dir),
    }

    # print(summary)

    # print out summary in a more console-friendly readable format
    print("\nSplit Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")



    return split_files

def shuffle_txt_inplace(txt_path: Path, seed: int = 0):
    txt_path = Path(txt_path)
    lines = [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    txt_path.write_text("\n".join(lines) + "\n")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    windows_dir = Path("dataset/recursive_samples")   # folder with 2774 .npz files
    splits_dir = Path("splits")                # tracked folder in git

    overrides = SplitOverrides(
        # Put these parent IDs exactly as parsed by extract_parent_id()
        train = {"flowrate_pattern_00_averaged_smoothed",
                 "flowrate_pattern_02_averaged_smoothed",
                 "flowrate_pattern_07_averaged_smoothed"},
        val = {"flowrate_pattern_03_averaged_smoothed",
               "flowrate_pattern_10_averaged_smoothed"},
        test = set(),
    )

    grouped_stratified_split(
        windows_dir=windows_dir,
        splits_dir=splits_dir,
        train_frac=0.80,
        val_frac=0.15,
        test_frac=0.05,
        seed=0,
        overrides=overrides,
    )

    shuffle_txt_inplace("splits/train.txt", seed=0)
    shuffle_txt_inplace("splits/val.txt", seed=0)  # optional
    shuffle_txt_inplace("splits/test.txt", seed=0)  # optional