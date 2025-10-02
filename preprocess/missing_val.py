import glob
import re
from pathlib import Path
import pandas as pd


def drop_impute_missing(
    src: Path | str = Path("../data/month_filtered"),
    dst: Path | str = Path("../data/missing_handled"),
    verbose: bool = False,
) -> Path:
    """
    Handle missing values with year-aware logic and save cleaned files.

    Source and destination directories:
    - Reads monthly-filtered parquet files from `../data/month_filtered`.
    - Saves cleaned outputs to `../data/missing_handled` with zstd compression.

    Logic:
    - For 2024 files: drop rows with NA in any of `passenger_count`, `RatecodeID`,
      or `store_and_fwd_flag` if those columns are present.
    - For 2025 files: impute missing values by mode for the same columns; default
      fallbacks are 1 for numeric and "N" for the flag if the mode is empty.

    Raises
    ------
    FileNotFoundError
        If no parquet files are found in the source directory.
    ValueError
        If a filename does not contain a YYYY-MM token used to infer the year.
    """
    SRC = Path(src)
    DST = Path(dst)
    DST.mkdir(parents=True, exist_ok=True)

    files = sorted(
        glob.glob(str(SRC / "yellow_tripdata_2024-*.parquet"))
        + glob.glob(str(SRC / "yellow_tripdata_2025-*.parquet"))
    )
    if not files:
        raise FileNotFoundError(f"No parquet files in {SRC}")

    DROP_NA_COLS = ["passenger_count", "RatecodeID", "store_and_fwd_flag"]


    def file_year(name: str) -> int:
        m = re.search(r"(\d{4})-(\d{2})", name)
        if not m:
            raise ValueError(f"Cannot parse YYYY-MM from filename: {name}")
        return int(m.group(1))


    for fp_str in files:
        fp = Path(fp_str)
        year = file_year(fp.name)
        df = pd.read_parquet(fp)

        n0 = len(df)

        if year == 2024:
            cols_present = [c for c in DROP_NA_COLS if c in df.columns]
            df = df.dropna(subset=cols_present) if cols_present else df
            action = f"dropped on {cols_present}" if cols_present else "no trio present"
        else:
            if "passenger_count" in df.columns:
                mode_pc = df["passenger_count"].dropna().mode()
                fill_pc = mode_pc.iloc[0] if not mode_pc.empty else 1
                df["passenger_count"] = df["passenger_count"].fillna(fill_pc)

            if "RatecodeID" in df.columns:
                mode_rate = df["RatecodeID"].dropna().mode()
                fill_rate = mode_rate.iloc[0] if not mode_rate.empty else 1
                df["RatecodeID"] = df["RatecodeID"].fillna(fill_rate)

            if "store_and_fwd_flag" in df.columns:
                mode_saf = df["store_and_fwd_flag"].dropna().mode()
                fill_saf = mode_saf.iloc[0] if not mode_saf.empty else "N"
                df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna(fill_saf)

            action = "imputed by mode (pc, rate, saf)"

        n1 = len(df)
        out_fp = DST / fp.name
        df.to_parquet(out_fp, index=False, compression="zstd")

        if verbose:
            print(f"{fp.name}: rows {n0:,} → {n1:,} | {action} | saved {out_fp}")

    return DST