import argparse
import glob
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from preprocess import (
    check_schema,
    column_normalize_and_save,
    year_check_and_save,
    drop_impute_missing,
    outlier_removal,
    build_and_save_features,
    build_splits,
    save_splits,
    build_and_save_preprocess,
)


def main(args):
    RAW = Path(args.raw_root)
    raw_files = sorted(glob.glob(str(RAW / "*.parquet")))

    check_schema(raw_files)
    print(f"[1/8] Schema check: compared {len(raw_files)} raw files")

    stage1_dir = column_normalize_and_save(raw_files)
    print(f"[2/8] Column normalization: saved to {stage1_dir}")

    month_dir = year_check_and_save(stage1_dir)
    print(f"[3/8] Month filtering: saved to {month_dir}")

    missing_dir = drop_impute_missing(month_dir)
    print(f"[4/8] Missing handling: saved to {missing_dir}")

    outlier_dir = outlier_removal(missing_dir)
    print(f"[5/8] Outlier removal: saved to {outlier_dir}")

    features_dir = build_and_save_features(outlier_dir)
    print(f"[6/8] Feature engineering: saved to {features_dir}")

    dfs = build_splits(features_dir)
    splits_dir = save_splits(dfs)
    print(f"[7/8] Dataset splits: saved to {splits_dir}")

    build_and_save_preprocess(train_path=Path(splits_dir) / "train.parquet")
    print("[8/8] Preprocess artifacts: saved to ../artifacts")

    if args.delete_intermediate:

        def safe_rmtree(p: Path) -> None:
            try:
                if p.exists() and p.is_dir():
                    shutil.rmtree(p)
            except Exception:
                pass

        # Intermediate directories to remove
        intermediates = [
            Path("../data/columns_normalized"),
            Path("../data/month_filtered"),
            Path("../data/missing_handled"),
            Path("../data/outliers_removed"),
            Path("../data/features_built"),
        ]
        for d in intermediates:
            safe_rmtree(d)
        print("[CLEANUP] Removed intermediate directories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NYC taxi preprocessing pipeline")
    parser.add_argument(
        "--raw-root",
        type=str,
        default="../data/raw",
        help="Path to the raw parquet files directory",
    )
    parser.add_argument(
        "--delete-intermediate",
        action="store_true",
        help="Delete intermediate directories after processing completes",
    )
    args = parser.parse_args()
    main(args)
