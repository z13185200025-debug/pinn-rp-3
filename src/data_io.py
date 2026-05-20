from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


CASE_PATTERN = re.compile(
    r"^(?P<diameter>\d+(?:\.\d+)?)-(?P<tin>\d+(?:\.\d+)?)-(?P<qw>\d+(?:\.\d+)?)$"
)


def parse_case_filename(filename: str | Path) -> dict[str, float | str]:
    """Parse diameter, inlet temperature, and wall heat flux from a dat filename."""
    path = Path(filename)
    stem = path.stem
    match = CASE_PATTERN.match(stem)
    if not match:
        raise ValueError(
            f"Invalid case filename '{path.name}'. Expected diameter-inlet_temperature-wall_heat_flux.dat"
        )
    return {
        "case_id": stem,
        "diameter": float(match.group("diameter")),
        "inlet_temperature": float(match.group("tin")),
        "wall_heat_flux": float(match.group("qw")),
    }


def _first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                return line.strip()
    return ""


def read_single_dat(path: str | Path) -> pd.DataFrame:
    """Read one CFD .dat file and append case metadata.

    The reader accepts whitespace-separated, tab-separated, and comma-like files.
    Bad files raise an exception that includes the source path.
    """
    path = Path(path)
    meta = parse_case_filename(path)
    first = _first_nonempty_line(path)
    if not first:
        raise ValueError(f"Empty file: {path}")
    has_header = any(ch.isalpha() for ch in first)
    sep = r"\s+|,"
    try:
        df = pd.read_csv(
            path,
            sep=sep,
            engine="python",
            header=0 if has_header else None,
            comment="#",
            skip_blank_lines=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    df = df.dropna(axis=1, how="all")
    for key, value in meta.items():
        df[key] = value
    df["source_file"] = path.name
    return df


def load_all_cases(
    data_dir: str | Path,
    max_cases: int | None = None,
    sample_rows: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load and concatenate all .dat files under a directory."""
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*.dat"))
    if not paths:
        raise FileNotFoundError(f"No .dat files found under {data_dir}")
    if max_cases is not None:
        paths = paths[: int(max_cases)]

    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    for path in paths:
        try:
            df = read_single_dat(path)
            if sample_rows is not None and len(df) > sample_rows:
                df = df.sample(int(sample_rows), random_state=random_seed)
            frames.append(df)
        except Exception as exc:
            failures.append((str(path), str(exc)))
            print(f"[WARN] Skipping {path}: {exc}")

    if not frames:
        raise RuntimeError(f"No files could be loaded. Failures: {failures[:5]}")
    if failures:
        print(f"[WARN] Loaded {len(frames)} files, skipped {len(failures)} files.")
    return pd.concat(frames, ignore_index=True)
