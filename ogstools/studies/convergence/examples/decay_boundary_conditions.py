from typing import TYPE_CHECKING

try:
    # this import is necessary for pip version of ogs
    import ogs.callbacks as OpenGeoSys
except ModuleNotFoundError:
    if not TYPE_CHECKING:
        # this import works for binary version of ogs
        import OpenGeoSys

import ogstools.physics.nuclearwasteheat as nuclear


class T_RepositorySourceTerm(OpenGeoSys.SourceTerm):
    def getFlux(self, t, coords, primary_vars):  # noqa: ARG002
        boundary_len = 1500  # m
        repo_edge_len = 1500  # m
        value = (
            nuclear.repo_2020_conservative.heat(t)
            / repo_edge_len
            / boundary_len
            / 2.0  # due to symmetry
        )
        derivative = [0.0] * len(primary_vars)
        return (value, derivative)


T_source_term = T_RepositorySourceTerm()
