"""The synthetic ICU cohort follows the published monthly protocol."""

from __future__ import annotations

import unittest

from claim.data import rolling_windows
from claim.published import FAMILY_FEATURES
from claim.synthetic import icu_cohort


class SyntheticTests(unittest.TestCase):
    def test_widths_and_first_test_month(self) -> None:
        for family, width in FAMILY_FEATURES.items():
            cohort = icu_cohort(family, per_month=2, seed=1, end=(2016, 3))
            self.assertEqual(cohort.series.shape[1:], (30, width))
            windows = rolling_windows(cohort, decay=1e-5)
            self.assertEqual(windows[0].test_period, "2016-01")
            self.assertEqual(windows[0].x_train.shape[-1], width)
            self.assertGreater(windows[0].weights[0], 0.0)


if __name__ == "__main__":
    unittest.main()
