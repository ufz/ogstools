"""Unit tests for nuclearwasteheat."""

import unittest

import numpy as np

import ogstools.physics.nuclearwasteheat as nuclear


class NuclearWasteHeatTest(unittest.TestCase):
    """Test case for nuclearwasteheat."""

    def test_waste_heat(self):
        """Test heat evaluation for different argument combinations."""
        for model in nuclear.waste_types:
            for i in range(len(model.nuclide_powers)):
                self.assertGreater(model.heat(0.0, baseline=True, ncl_id=i), 0)
            self.assertGreater(model.heat(0.0, baseline=True), 0)
            self.assertGreater(model.heat(0.0), 0)
        self.assertGreater(nuclear.repo_2020.heat(0.0), 0)
        self.assertTrue(
            np.all(
                nuclear.repo_2020_conservative.heat(
                    np.geomspace(1, 1e6, num=10)
                )
            )
        )
