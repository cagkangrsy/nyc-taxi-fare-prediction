import glob
import re
from pathlib import Path
from typing import List
import pandas as pd
import pyarrow.parquet as pq
from pandas.tseries.offsets import MonthEnd


def check_schema(dataframes: List[str], verbose: bool = False) -> None:
    """
    Check column schemas across multiple parquet files and print a summary.

    Parameters
    ----------
    dataframes : List[str]
        List of file paths to parquet files to compare.

    Notes
    -----
    - Uses the first file as the reference schema.
    - Only compares column names and order; dtypes are not validated here.
    - Prints lists of files that match, and files with extra/missing columns.
    """
    schemas = {}
    matching_list = []
    extra_list = []
    missing_list = []

    # Collect schemas
    for df_path in dataframes:
        pf = pq.ParquetFile(df_path)
        cols = list(pf.schema.names)
        schemas[Path(df_path).name] = cols

    # Use the first file as schema reference
    ref_file, ref_cols = next(iter(schemas.items()))
    if verbose:
        print(f"Reference file: {ref_file}\n")

    for fname, cols in schemas.items():
        if fname == ref_file:
            matching_list.append(fname)
            continue

        if cols != ref_cols:
            missing = set(ref_cols) - set(cols)
            extra = set(cols) - set(ref_cols)
            if missing:
                missing_list.append(fname)
            if extra:
                extra_list.append(fname)
        else:
            matching_list.append(fname)

    if verbose:
        print(f"Matching dfs: {matching_list}")
        print(f"Extra col dfs: {extra_list}")
        print(f"Missing col dfs: {missing_list}")


def column_normalize_and_save(dataframes: List[str],
                              save_path: Path = Path("../data/columns_normalized"),
                              verbose: bool = False) -> Path:
    """
    Normalize columns across parquet files and save normalized copies.

    This function:
    - Drops a fixed set of fare-related columns if present
    - Reads each file using a common reference column list derived from the first file
    - Adds back any missing columns as NA so all outputs share the same columns
    - Reorders columns consistently and saves to `save_path`

    Parameters
    ----------
    dataframes : List[str]
        List of file paths to input parquet files.
    save_path : Path, default ../data/columns_normalized
        Output directory for normalized parquet files.

    Returns
    -------
    Path
        Directory path where normalized parquet files were saved.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    # Columns to drop
    DROP_COLS = {
        "cbd_congestion_fee",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "congestion_surcharge",
        "airport_fee",
    }

    # Use first file as reference
    ref_pf = pq.ParquetFile(dataframes[0])
    KEEP_COLS = [c for c in ref_pf.schema.names if c.lower() not in DROP_COLS]

    if verbose:
        print(f"Reference KEEP_COLS ({len(KEEP_COLS)}): {KEEP_COLS}\n")

    for fp in dataframes:
        schema_cols = set(pq.ParquetFile(fp).schema.names)

        # Only keep desired columns that exist in this file
        cols_to_read = [c for c in KEEP_COLS if c in schema_cols]

        df = pd.read_parquet(fp, columns=cols_to_read)

        # Add missing KEEP_COLS back as NA
        missing = [c for c in KEEP_COLS if c not in df.columns]
        for c in missing:
            df[c] = pd.NA

        # Reorder to KEEP_COLS
        df = df[KEEP_COLS]

        # Save processed copy
        out_fp = save_path / Path(fp).name
        df.to_parquet(out_fp, index=False, compression="zstd")
        if verbose:
            print(f"{Path(fp).name}: rows={len(df):,} saved {out_fp}")

    return save_path


def parse_year_month(fp: str) -> tuple[int, int]:
    """
    Extract year and month from a filename like '..._YYYY-MM.parquet'.

    Parameters
    ----------
    fp : str
        File path or name containing a YYYY-MM token.

    Returns
    -------
    tuple[int, int]
        A pair of (year, month).

    Raises
    ------
    ValueError
        If the pattern YYYY-MM is not found in the filename.
    """
    m = re.search(r"(\d{4})-(\d{2})", Path(fp).name)
    if not m:
        raise ValueError(f"Cannot parse YYYY-MM from: {fp}")
    return int(m.group(1)), int(m.group(2))


def year_check_and_save(
    src: Path | str = Path("../data/columns_normalized"),
    dst: Path | str = Path("../data/month_filtered"),
    verbose: bool = False,
) -> Path:
    """
    Filter stage1 parquet files to keep only rows within their respective months.

    Reads each monthly file from `../data/columns_normalized`, coerces pickup datetimes,
    and keeps rows where `tpep_pickup_datetime` falls within that file's month
    boundaries. Saves results to `../data/month_filtered` and prints a summary.
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(src / "yellow_tripdata_20*.parquet")))

    summary = []
    for fp in files:
        yy, mm = parse_year_month(fp)
        month_start = pd.Timestamp(yy, mm, 1)
        next_month_start = (month_start + MonthEnd(1)) + pd.Timedelta(days=1)

        df = pd.read_parquet(fp)
        df["tpep_pickup_datetime"] = pd.to_datetime(
            df["tpep_pickup_datetime"], errors="coerce"
        )

        n_total = len(df)
        n_nat = df["tpep_pickup_datetime"].isna().sum()

        mask_in = (df["tpep_pickup_datetime"] >= month_start) & (
            df["tpep_pickup_datetime"] < next_month_start
        )
        df_clean = df.loc[mask_in].copy()

        n_in = len(df_clean)
        n_out = n_total - n_in
        pct_out = (n_out / n_total * 100) if n_total else 0.0

        out_fp = dst / Path(fp).name
        df_clean.to_parquet(out_fp, index=False)

        if verbose:
            print(
                f"{Path(fp).name}: total={n_total:,}  kept={n_in:,}  dropped={n_out:,} ({pct_out:.4f}%)  NaT_pickup={n_nat:,}"
            )
        summary.append((Path(fp).name, n_total, n_in, n_out, n_nat))

    if verbose:
        print(f"\n[DONE] Saved filtered files to: {dst}")
    return dst