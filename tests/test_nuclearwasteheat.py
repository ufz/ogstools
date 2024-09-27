"""Unit tests for nuclearwasteheat."""


import numpy as np

import ogstools.physics.nuclearwasteheat as nuclear


class TestNuclearWasteHeat:
    """Test case for nuclearwasteheat."""

    def test_waste_heat(self):
        """Test heat evaluation for different argument combinations."""
        for model in nuclear.waste_types:
            for i, _ in enumerate(model.nuclide_powers):
                assert model.heat(0.0, baseline=True, ncl_id=i) > 0
            assert model.heat(0.0, baseline=True) > 0
            assert model.heat(0.0) > 0
        assert nuclear.repo_2020.heat(0.0) > 0
        assert np.all(
            nuclear.repo_2020_conservative.heat(np.geomspace(1, 1e6, num=10))
        )
