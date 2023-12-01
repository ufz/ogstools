from typing import TYPE_CHECKING

try:
    # this import is necessary for pip version of ogs
    import ogs.callbacks as OpenGeoSys
except ModuleNotFoundError:
    if not TYPE_CHECKING:
        # this import works for binary version of ogs
        import OpenGeoSys

import ogstools.physics.nuclearwasteheat as nuclear


class T_RepositorySourceTerm(OpenGeoSys.BoundaryCondition):
    t_prev: float = 0.0
    t_current: float = 0.0
    use_temporal_split: bool = False
    repo = nuclear.repo_2020_conservative

    def update_t_prev(self, t):
        if self.t_current != t:
            self.t_prev = self.t_current
            self.t_current = t

    def temporal_split(self, t, split: float = 0.5):
        return self.t_prev + split * (t - self.t_prev)

    def getFlux(self, t, coords, primary_vars):  # noqa: ARG002
        self.update_t_prev(t)
        boundary_len = 1500  # m
        repo_edge_len = 1500  # m
        _t = self.temporal_split(t) if self.use_temporal_split else t
        value = self.repo.heat(_t) / repo_edge_len / boundary_len / 2.0
        # / 2.0 due to symmetry
        return (True, value, [0.0])


T_source_term = T_RepositorySourceTerm()
T_source_term.use_temporal_split = True
