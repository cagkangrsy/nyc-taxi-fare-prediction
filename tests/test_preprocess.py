import importlib
import sys
import unittest
import pandas as pd

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.fe = importlib.import_module("preprocess.feature_eng")
        self.fx = importlib.import_module("preprocess.features")
        self.norm = importlib.import_module("preprocess.normalize")
        self.sample = importlib.import_module("preprocess.sample")
        self.run_pre = importlib.import_module("scripts.run_preprocess")

    # feature_eng
    def test_fix_location_ids_replaces_and_filters(self):
        df = pd.DataFrame({
            "PULocationID": [57, 1, 264],
            "DOLocationID": [2, 57, 2],
        })
        out = self.fe.fix_location_ids(df)
        self.assertIn(56, out["PULocationID"].values)
        self.assertIn(56, out["DOLocationID"].values)
        self.assertFalse((out["PULocationID"] == 264).any())

    def test_haversine_zero_distance(self):
        d = self.fe.haversine_distance(0.0, 0.0, 0.0, 0.0)
        val = float(d if isinstance(d, (int, float)) else d.item() if hasattr(d, "item") else d)
        self.assertAlmostEqual(val, 0.0, places=6)

    # features
    def test_prepare_features_and_split_xy(self):
        df = pd.DataFrame({
            "trip_distance": [1.0, 2.0],
            "haversine_distance": [0.5, 1.0],
            "pickup_borough": ["Manhattan", "Queens"],
            "pickup_date": [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()],
            "total_amount": [10.0, 20.0],
        })
        X_only = self.fx.prepare_features(df)
        self.assertIn("trip_distance", X_only.columns)
        self.assertNotIn("pickup_date", X_only.columns)
        X, y = self.fx.split_xy(df)
        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)

    # normalize & sample utilities
    def test_parse_year_month_and_assign_split(self):
        y, m = self.norm.parse_year_month("yellow_tripdata_2024-10.parquet")
        self.assertEqual((y, m), (2024, 10))
        self.assertEqual(self.sample.assign_split("yellow_tripdata_2024-05.parquet"), "train")
        self.assertEqual(self.sample.assign_split("yellow_tripdata_2025-02.parquet"), "val")
        self.assertEqual(self.sample.assign_split("yellow_tripdata_2025-07.parquet"), "test")


if __name__ == "__main__":
    unittest.main()

