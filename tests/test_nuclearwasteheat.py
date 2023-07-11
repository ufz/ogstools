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
                assert model.heat(0.0, baseline=True, ncl_id=i)
            assert model.heat(0.0, baseline=True)
            assert model.heat(0.0)
        assert nuclear.repo_2020.heat(0.0)
        assert np.all(
            nuclear.repo_2020_conservative.heat(np.geomspace(1, 1e6, num=10))
        )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
