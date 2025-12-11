import shutil

import pytest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.mesh import create

meshpath = EXAMPLES_DIR / "meshlib"


class TestTetraeder:
    layerset = meshpath / "compose_geomodel/layersets.csv"
    materialset = meshpath / "compose_geomodel/materialset.csv"
    surfacedata = meshpath / "mesh1/surface_data/"

    @pytest.mark.tools()  # createTetgenSmeshFromRasters
    @pytest.mark.xfail(
        shutil.which("tetgen") is None, reason="Tetgen not installed"
    )
    def yz(self):
        mesh1_df = create.dataframe_from_csv(
            1, self.layerset, self.materialset, self.surfacedata
        )
        layer_set = create.LayerSet.from_pandas(mesh1_df)
        mesh = layer_set.to_region_tetrahedron(resolution=400).mesh

        assert len(mesh.cell_data["MaterialIDs"]) > 0
        assert mesh.number_of_cells > 10000
        assert mesh.number_of_cells < 30000
