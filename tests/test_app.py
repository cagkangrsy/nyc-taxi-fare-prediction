import unittest
import importlib


class TestApp(unittest.TestCase):
    def test_app_has_health_endpoint(self):
        mod = importlib.import_module("scripts.app")
        self.assertTrue(hasattr(mod, "app"))

    def test_healthz_function_direct(self):
        mod = importlib.import_module("scripts.app")
        self.assertTrue(callable(mod.healthz))
        out = mod.healthz()
        self.assertIsInstance(out, dict)
        self.assertEqual(out.get("status"), "ok")

    def test_input_model_construct(self):
        mod = importlib.import_module("scripts.app")
        payload = dict(
            VendorID=1,
            tpep_pickup_datetime="2024-01-01T00:00:00",
            tpep_dropoff_datetime="2024-01-01T00:10:00",
            passenger_count=1,
            trip_distance=1.2,
            RatecodeID=1,
            store_and_fwd_flag="N",
            PULocationID=1,
            DOLocationID=1,
            payment_type=1,
        )
        item = mod.InputData(**payload)
        self.assertEqual(item.VendorID, 1)


if __name__ == "__main__":
    unittest.main()


