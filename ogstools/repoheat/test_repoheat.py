"""Unit tests for repoheat."""

import unittest

import numpy as np

import ogstools.repoheat as repo

EPS = 1e-7


class RepoheatTest(unittest.TestCase):
    """Test case for repoheat."""

    def test_waste_heat(self):
        """Test temperature property."""
        for model in repo.models:
            for i in range(len(model.nuclide_powers)):
                assert model.heat(0.0, baseline=True, ncl_id=i)
            assert model.heat(0.0, baseline=True)
            assert model.heat(0.0)
        assert repo.repo_2020.heat(0.0)
        assert np.all(
            repo.repo_2020_conservative.heat(np.geomspace(1, 1e6, num=10))
        )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
