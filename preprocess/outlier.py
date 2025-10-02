import glob
from pathlib import Path
import pandas as pd


def outlier_removal(
    src: Path | str = Path("../data/missing_handled"),
    dst: Path | str = Path("../data/outliers_removed"),
    verbose: bool = False,
) -> Path:
    """
    Remove rule-based outliers from key numeric columns and save cleaned files.

    Source and destination directories:
    - Reads cleaned, missing-handled parquet files from `../data/missing_handled`.
    - Writes outlier-filtered outputs to `../data/outliers_removed` with zstd compression.

    Columns and thresholds:
    - Columns considered: `passenger_count`, `trip_distance`, `total_amount`.
    - Keeps rows within inclusive ranges:
        passenger_count: [1, 9]
        trip_distance: [0.1, 50]
        total_amount: [1, 250]

    Output:
    - Saves per-month cleaned parquet files and prints a summary table of rows kept/removed
      and counts below/above each threshold per file.

    Raises
    ------
    FileNotFoundError
        If no parquet files are found in the source directory.
    """
    SRC = Path(src)
    OUT = Path(dst)
    OUT.mkdir(parents=True, exist_ok=True)

    COLS = ["passenger_count", "trip_distance", "total_amount"]

    # Rule-based thresholds
    THRESH_LEFT = {"passenger_count": 1, "trip_distance": 0.1, "total_amount": 1}
    THRESH_RIGHT = {"passenger_count": 9, "trip_distance": 50, "total_amount": 250}

    files = sorted(glob.glob(str(SRC / "yellow_tripdata_20*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {SRC}")

    summary_rows = []

    for fp in files:
        df_all = pd.read_parquet(fp)
        for c in COLS:
            if c in df_all.columns:
                df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

        n_raw = len(df_all)

        masks = {}
        for c in COLS:
            s = df_all[c]
            masks[c] = s.isna() | s.between(
                THRESH_LEFT[c], THRESH_RIGHT[c], inclusive="both"
            )

        keep_mask = (
            masks["passenger_count"] & masks["trip_distance"] & masks["total_amount"]
        )
        df_clean = df_all.loc[keep_mask].copy()
        n_clean = len(df_clean)
        n_removed = n_raw - n_clean

        breakdown = {}
        for c in COLS:
            s = df_all[c]
            below = (s.notna()) & (s < THRESH_LEFT[c])
            above = (s.notna()) & (s > THRESH_RIGHT[c])
            breakdown[f"{c}_below"] = int(below.sum())
            breakdown[f"{c}_above"] = int(above.sum())

        out_fp = OUT / Path(fp).name
        df_clean.to_parquet(out_fp, index=False, compression="zstd")

        summary_rows.append(
            {
                "file": Path(fp).name,
                "rows_raw": n_raw,
                "rows_clean": n_clean,
                "rows_removed": n_removed,
                **breakdown,
            }
        )

    # Summary table
    clean_summary = pd.DataFrame(summary_rows).sort_values("file", ignore_index=True)
    if verbose:
        print("\n=== Outlier Removal Summary (saved per-month cleaned files) ===\n")
        print(clean_summary.to_string(index=False))
    return OUT