from collections.abc import Callable, Iterator

import pytest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.mesh import create


@pytest.fixture(name="make_layerset", scope="module")
def layerset_factory() -> Iterator[Callable[[int], create.LayerSet]]:
    created = []
    meshpath = EXAMPLES_DIR / "meshlib"
    set_csv = meshpath / "compose_geomodel/layersets.csv"
    surfaces = meshpath / "mesh1/surface_data/"

    def make_layerset(layerset_id: int) -> create.LayerSet:
        ls_df = create.dataframe_from_csv(layerset_id, set_csv, surfaces)
        layerset = create.LayerSet.from_pandas(ls_df)
        created.append(layerset)
        return layerset

    yield make_layerset

    created.clear()
