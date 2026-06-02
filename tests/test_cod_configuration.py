import unittest

from config import MELBOURNE_TARGET_COLUMN
from server_model.melbourne_utils import build_feature_presets


class CodConfigurationTests(unittest.TestCase):
    def test_default_target_is_cod(self) -> None:
        self.assertEqual(MELBOURNE_TARGET_COLUMN, "Chemical Oxygen Demand")

    def test_process_core_no_bod_excludes_bod(self) -> None:
        full_features = [
            "Average Outflow",
            "Average Inflow",
            "Ammonia",
            "Biological Oxygen Demand",
            "Chemical Oxygen Demand",
            "Total Nitrogen",
            "day_of_week",
            "week_of_year",
            "month_sin",
            "month_cos",
            "weekday_sin",
            "weekday_cos",
            "is_weekend",
        ]

        preset = build_feature_presets(full_features, "Chemical Oxygen Demand")["process_core_no_bod"]

        self.assertIn("Chemical Oxygen Demand", preset)
        self.assertIn("Average Inflow", preset)
        self.assertIn("Average Outflow", preset)
        self.assertIn("Ammonia", preset)
        self.assertIn("Total Nitrogen", preset)
        self.assertNotIn("Biological Oxygen Demand", preset)

    def test_notebook_cod_matches_notebook_feature_set(self) -> None:
        full_features = [
            "Average Outflow",
            "Average Inflow",
            "Ammonia",
            "Biological Oxygen Demand",
            "Chemical Oxygen Demand",
            "Total Nitrogen",
            "Average Temperature",
            "Maximum temperature",
            "Minimum temperature",
            "Total rainfall",
            "day_of_week",
            "week_of_year",
            "month_sin",
            "month_cos",
            "weekday_sin",
            "weekday_cos",
            "is_weekend",
        ]

        preset = build_feature_presets(full_features, "Chemical Oxygen Demand")["notebook_cod"]

        self.assertEqual(
            preset,
            [
                "Ammonia",
                "Total Nitrogen",
                "Average Inflow",
                "Average Outflow",
                "Average Temperature",
                "Maximum temperature",
                "Minimum temperature",
                "Total rainfall",
            ],
        )
        self.assertNotIn("Chemical Oxygen Demand", preset)
        self.assertNotIn("Biological Oxygen Demand", preset)
        self.assertNotIn("day_of_week", preset)


if __name__ == "__main__":
    unittest.main()
