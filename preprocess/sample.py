from pathlib import Path
import glob
import re
from typing import Dict, List, Iterable, Tuple

import pandas as pd


DEFAULT_SEED = 42
DEFAULT_SAMPLE_ROWS_PER_MONTH: Dict[str, int] = {
    "train": 100_000,
    "val": 35_000,
    "test": 100_000,
}


def ym_from_name(name: str) -> Tuple[int, int]:
    """
    Parse year and month from a filename containing a YYYY-MM token.

    Example: "yellow_tripdata_2024-10.parquet" -> (2024, 10)
    """
    m = re.search(r"(\d{4})-(\d{2})", name)
    if not m:
        raise ValueError(f"Cannot parse YYYY-MM from {name}")
    return int(m.group(1)), int(m.group(2))


def assign_split(name: str) -> str:
    """
    Assign a split key based on the year/month parsed from the filename.

    - 2024 -> "train"
    - 2025-01..06 -> "val"
    - 2025-07 -> "test"
    - otherwise -> "ignore"
    """
    y, m = ym_from_name(name)
    if y == 2024:
        return "train"
    if y == 2025 and 1 <= m <= 6:
        return "val"
    if y == 2025 and m == 7:
        return "test"
    return "ignore"


def list_month_files(src_dir: Path | str, pattern: str = "yellow_tripdata_20*.parquet") -> List[str]:
    """
    List monthly parquet files under a directory matching the given pattern.
    """
    src_dir = Path(src_dir)
    files = sorted(glob.glob(str(src_dir / pattern)))
    if not files:
        raise FileNotFoundError(f"No files found under {src_dir}")
    return files


def bucket_files_by_split(files: Iterable[str]) -> Dict[str, List[str]]:
    """
    Group file paths into {train,val,test} buckets using `assign_split`.
    Files that map to "ignore" are dropped.
    """
    buckets = {"train": [], "val": [], "test": []}
    for fp in files:
        split = assign_split(Path(fp).name)
        if split in buckets:
            buckets[split].append(fp)
    return buckets


def load_concat_fixed_n(
    file_list: Iterable[str],
    split_key: str,
    sample_rows_per_month: Dict[str, int] = DEFAULT_SAMPLE_ROWS_PER_MONTH,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Load and concatenate a fixed number of rows per monthly file for a split.

    - If a file has fewer rows than the target, take all rows.
    - If more, sample exactly `target_n` rows with a fixed random seed.
    """
    target_n = sample_rows_per_month.get(split_key)
    if not target_n or target_n <= 0:
        raise ValueError(
            f"sample_rows_per_month[{split_key!r}] must be a positive integer."
        )

    parts: List[pd.DataFrame] = []
    for fp in file_list:
        df = pd.read_parquet(fp)
        n_in = len(df)
        if n_in == 0:
            continue
        if n_in > target_n:
            df = df.sample(n=target_n, random_state=seed)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_splits(
    src_dir: Path | str = Path("../data/features_built"),
    pattern: str = "yellow_tripdata_20*.parquet",
    sample_rows_per_month: Dict[str, int] = DEFAULT_SAMPLE_ROWS_PER_MONTH,
    seed: int = DEFAULT_SEED,
) -> Dict[str, pd.DataFrame]:
    """
    Build train/val/test dataframes from monthly feature files.

    Parameters default to reading from `../data/features_built`.
    """
    files = list_month_files(src_dir, pattern)
    buckets = bucket_files_by_split(files)
    return {
        "train": load_concat_fixed_n(buckets["train"], "train", sample_rows_per_month, seed),
        "val": load_concat_fixed_n(buckets["val"], "val", sample_rows_per_month, seed),
        "test": load_concat_fixed_n(buckets["test"], "test", sample_rows_per_month, seed),
    }


def save_splits(dfs: Dict[str, pd.DataFrame], out_dir: Path | str = Path("../data/splits")) -> Path:
    """
    Save split dataframes to parquet files under the output directory.

    Parameters default to writing into `../data/splits`.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if "train" in dfs:
        dfs["train"].to_parquet(out_path / "train.parquet", index=False)
    if "val" in dfs:
        dfs["val"].to_parquet(out_path / "val.parquet", index=False)
    if "test" in dfs:
        dfs["test"].to_parquet(out_path / "test.parquet", index=False)
    return out_path