import tempfile
from pathlib import Path

import pytest

import ogstools as ot

"""
##############
Below tests regarding FEFLOW models.
"""

pytest.importorskip("ifm")

from ogstools.examples import (  # noqa: E402 / because of ifm-skip
    feflow_model_2D_HT,
)


def get_attributes_of_project(prj: ot.Project, xpath: str) -> list:
    return [
        attribute.find("name").text
        for attribute in prj.tree.getroot().findall(xpath)
    ]


class Test_OGSModel:
    def setup_method(self):
        self.tempdir = Path(tempfile.mkdtemp("test_feflow_model"))
        self.feflow_HT = ot.FeflowModel(
            feflow_model_2D_HT, self.tempdir / "feflow_HT"
        )

    def test_from_feflow(self):
        ogs_model = ot.OGSModel.from_feflow_model(self.feflow_HT)

        for name, subdomain in ogs_model.subdomains.items():
            assert (
                subdomain.n_arrays == self.feflow_HT.subdomains[name].n_arrays
            )
            assert subdomain.n_cells == self.feflow_HT.subdomains[name].n_cells
            assert (
                subdomain.n_points == self.feflow_HT.subdomains[name].n_points
            )
        assert ogs_model.mesh.n_arrays == self.feflow_HT.mesh.n_arrays
        assert ogs_model.mesh.n_cells == self.feflow_HT.mesh.n_cells
        assert ogs_model.mesh.n_points == self.feflow_HT.mesh.n_points
        xpaths = [
            "./parameters/parameter",
            './media/medium[@id="0"]/properties/property',
        ]
        for xpath in xpaths:
            assert get_attributes_of_project(
                ogs_model.project, xpath
            ) == get_attributes_of_project(self.feflow_HT.project, xpath)

    def test_read_feflow(self):
        ogs_model_read = ot.OGSModel.read_feflow(feflow_model_2D_HT)
        ogs_model_read.output_path = self.tempdir / "feflow_HT_read"
        for name, subdomain in ogs_model_read.subdomains.items():
            assert (
                subdomain.n_arrays == self.feflow_HT.subdomains[name].n_arrays
            )
            assert subdomain.n_cells == self.feflow_HT.subdomains[name].n_cells
            assert (
                subdomain.n_points == self.feflow_HT.subdomains[name].n_points
            )
        assert ogs_model_read.mesh.n_arrays == self.feflow_HT.mesh.n_arrays
        assert ogs_model_read.mesh.n_cells == self.feflow_HT.mesh.n_cells
        assert ogs_model_read.mesh.n_points == self.feflow_HT.mesh.n_points
        xpaths = [
            "./parameters/parameter",
            './media/medium[@id="0"]/properties/property',
        ]
        for xpath in xpaths:
            assert get_attributes_of_project(
                ogs_model_read.project, xpath
            ) == get_attributes_of_project(self.feflow_HT.project, xpath)
