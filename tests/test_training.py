import importlib
import unittest
import numpy as np


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.run_train = importlib.import_module("scripts.run_training")
        self.play = importlib.import_module("training.train_playground")

    def test_run_training_main_callable_and_unknown_mode(self):
        self.assertTrue(callable(self.run_train.main))
        args = type("A", (), {
            "artifact_root": "../artifacts",
            "data_root": "../data/splits",
            "mode": "unknown",
            "models": ["lightgbm"],
            "opt_model": "lightgbm",
            "n_trials": 1,
            "timeout": None,
            "early_stopping_rounds": 10,
            "seed": 0,
        })
        with self.assertRaises(ValueError):
            self.run_train.main(args)

    def test_fit_time_predict_meanbaseline(self):
        ytr = np.array([10.0, 20.0])
        yva = np.array([15.0, 25.0])
        Xtr = np.zeros((2, 1))
        Xva = np.zeros((2, 1))
        res = self.play.fit_time_predict("MeanBaseline", None, Xtr, ytr, Xva, yva)
        self.assertIn("RMSE", res)
        self.assertIn("MAE", res)
        self.assertIn("R2", res)
        self.assertEqual(res["Model"], "MeanBaseline")


if __name__ == "__main__":
    unittest.main()

