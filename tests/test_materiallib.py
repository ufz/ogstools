from pathlib import Path

import pytest
import yaml  # type: ignore[import]

import ogstools as ot
from ogstools import examples
from ogstools.materiallib.core import component, components, material_manager
from ogstools.materiallib.core.material import Material
from ogstools.materiallib.core.media import MediaSet
from ogstools.materiallib.schema import process_schema, required_properties
from ogstools.ogs6py import Project


@pytest.fixture()
def make_material():
    """Factory that builds a Material object directly from properties."""

    def _make(properties: dict, name="test_material"):
        raw_data = {"name": name, "properties": properties}
        return Material(name=name, raw_data=raw_data)

    return _make


@pytest.fixture()
def write_yaml(tmp_path):
    """Write a dict to a YAML file in tmp_path and return the path."""

    def _write(filename: str, data: dict):
        path = tmp_path / filename
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        return path

    return _write


@pytest.fixture()
def example_materials(make_material):
    """Provide commonly used materials as dicts and as ready Material objects."""
    return {
        "granite": {
            "name": "granite",
            "properties": {"Density": {"type": "Constant", "value": 2700}},
        },
        "water": {
            "name": "water",
            "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
        },
        "sandstone": {
            "name": "sandstone",
            "properties": {"Density": {"type": "Constant", "value": 2200}},
        },
        "limestone": {
            "name": "limestone",
            "properties": {"Density": {"type": "Constant", "value": 2500}},
        },
        "clay": {
            "name": "clay",
            "properties": {"Density": {"type": "Constant", "value": 2400}},
        },
        "granite_obj": make_material(
            {"Density": {"type": "Constant", "value": 2700}}, name="granite"
        ),
        "water_obj": make_material(
            {"Viscosity": {"type": "Constant", "value": 1.0}}, name="water"
        ),
        "clay_obj": make_material(
            {"Density": {"type": "Constant", "value": 2400}}, name="clay"
        ),
    }


@pytest.fixture()
def make_db(write_yaml, tmp_path):
    """Factory that creates a MaterialManager from a dict of materials by writing them to tmp_path."""

    def _make(materials: dict[str, dict]):
        for name, data in materials.items():
            write_yaml(f"{name}.yml", data)
        return material_manager.MaterialManager(data_dir=tmp_path)

    return _make


@pytest.fixture()
def simple_schema():
    """Provide a minimal schema requiring only a 'Density' property and no phases."""
    return {"properties": ["Density"], "phases": []}


@pytest.fixture()
def default_materials():
    """Provide a minimal set of default materials (clay + water)."""
    return {
        "clay": {
            "name": "clay",
            "properties": {"Density": {"type": "Constant", "value": 2400}},
        },
        "water": {
            "name": "water",
            "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
        },
    }


@pytest.fixture()
def default_schema():
    """Provide a minimal process schema requiring Density and one AqueousLiquid phase."""
    return {
        "properties": ["Density"],
        "phases": [{"type": "AqueousLiquid", "properties": ["Viscosity"]}],
    }


@pytest.fixture()
def default_subdomains():
    """Provide a single subdomain definition pointing to clay."""
    return [{"subdomain": "region1", "material": "clay", "material_ids": [1]}]


@pytest.fixture()
def default_fluids():
    """Provide a single fluid definition mapping AqueousLiquid -> water."""
    return {"AqueousLiquid": "water"}


@pytest.fixture()
def make_filtered_db(write_yaml, tmp_path, monkeypatch):
    """Factory that builds a filtered MaterialManager with arbitrary materials, schema, and fluids.

    Parameters
    ----------
    materials : dict
        A mapping of material names to raw data dicts, written to YAML before loading the DB.
    schema : dict
        A process schema dict with required medium/phase/component properties.
    subdomains : list
        A list of subdomain definitions, passed to MaterialManager.filter.
    fluids : dict
        A mapping of phase types to material names, passed to MaterialManager.filter.
    diffusion : dict | None
        Optional diffusion coefficient data to be written to diffusion_coefficients.yml.
    """

    def _make(
        materials: dict,
        schema: dict,
        subdomains: list,
        fluids: dict,
        diffusion: dict | None = None,
    ):
        for name, data in materials.items():
            write_yaml(f"{name}.yml", data)

        db = material_manager.MaterialManager(data_dir=tmp_path)
        monkeypatch.setitem(material_manager.PROCESS_SCHEMAS, "dummy", schema)

        if diffusion:
            (tmp_path / "diffusion_coefficients.yml").write_text(
                yaml.safe_dump(diffusion)
            )
            monkeypatch.setattr(components.defs, "MATERIALS_DIR", str(tmp_path))

        return db.filter(process="dummy", subdomains=subdomains, fluids=fluids)

    return _make


class TestMaterialLib:
    def test_material_parses_properties_from_raw_data(self, make_material):
        """Material should correctly parse properties (including lists) from raw_data."""
        mat = make_material(
            {
                "Density": {"type": "Constant", "value": 2500},
                "Permeability": [
                    {"type": "Constant", "value": 1e-18},
                    {"type": "Constant", "value": 2e-18},
                ],
            }
        )
        props = mat.properties
        assert len(props) == 3
        assert any(p.name == "Density" for p in props)
        assert sum(p.name == "Permeability" for p in props) == 2

    def test_material_property_names_returns_all_names(self, make_material):
        """Material.property_names should return the names of all parsed properties."""
        mat = make_material(
            {
                "Density": {"type": "Constant", "value": 2500},
                "Viscosity": {"type": "Constant", "value": 1.0},
            }
        )
        assert set(mat.property_names()) == {"Density", "Viscosity"}

    def test_material_filter_properties_returns_only_allowed(
        self, make_material
    ):
        """Material.filter_properties should return a new Material with only allowed properties."""
        mat = make_material(
            {
                "Density": {"type": "Constant", "value": 2500},
                "Viscosity": {"type": "Constant", "value": 1.0},
            }
        )

        filtered = mat.filter_properties({"Density"})

        assert filtered.property_names() == ["Density"]

    def test_material_filter_properties_empty_set_returns_empty_material(
        self, make_material
    ):
        """Material.filter_properties with empty allowed set should return a Material without properties."""
        mat = make_material(
            {
                "Density": {"type": "Constant", "value": 2500},
            }
        )

        filtered = mat.filter_properties(set())

        assert filtered.properties == []
        assert filtered.property_names() == []

    def test_material_repr_contains_name_and_property_count(
        self, make_material
    ):
        """Material.__repr__ should show the material name and number of properties."""
        mat = make_material(
            {
                "Density": {"type": "Constant", "value": 2500},
                "Viscosity": {"type": "Constant", "value": 1.0},
            }
        )

        text = repr(mat)
        assert "test_material" in text
        assert "2 properties" in text

    def test_material_filter_process_respects_schema(self, make_material):
        """Material.filter_process should keep only properties listed in the process schema."""
        mat = make_material(
            {
                "Density": {"type": "Constant", "value": 2500},
                "Permeability": {"type": "Constant", "value": 1e-18},
                "Viscosity": {"type": "Constant", "value": 1.0},
            }
        )
        process_schema = {
            "properties": [
                "Density",
                "Viscosity",
            ],  # required medium-level properties
            "phases": [],
        }

        filtered = mat.filter_process(process_schema)

        names = filtered.property_names()
        assert "Density" in names
        assert "Viscosity" in names
        assert "Permeability" not in names


class TestMaterialManager:
    def test_materialdb_loads_yaml_files(self, tmp_path, write_yaml):
        """MaterialManager should load all YAML files in the given directory into Material objects."""
        write_yaml(
            "rock.yml",
            {
                "name": "granite",
                "properties": {"Density": {"type": "Constant", "value": 2700}},
            },
        )

        db = material_manager.MaterialManager(data_dir=tmp_path)
        mat = db.get_material("granite")

        assert mat is not None
        assert mat.name == "granite"
        assert "Density" in mat.property_names()

    def test_materialdb_get_material_returns_correct_object(
        self, tmp_path, write_yaml
    ):
        """MaterialManager.get_material should return the correct Material instance by name."""
        write_yaml(
            "water.yml",
            {
                "name": "water",
                "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
            },
        )

        db = material_manager.MaterialManager(data_dir=tmp_path)
        mat = db.get_material("water")

        assert mat is not None
        assert mat.name == "water"
        assert "Viscosity" in mat.property_names()

    def test_materialdb_list_materials_returns_all_names(
        self, tmp_path, write_yaml
    ):
        """MaterialManager.list_materials should return a list of all loaded material names."""
        mats = [
            {
                "name": "sandstone",
                "properties": {"Density": {"type": "Constant", "value": 2200}},
            },
            {
                "name": "limestone",
                "properties": {"Density": {"type": "Constant", "value": 2500}},
            },
        ]
        for m in mats:
            write_yaml(f"{m['name']}.yml", m)

        db = material_manager.MaterialManager(data_dir=tmp_path)
        names = db._list_materials()

        assert set(names) == {"sandstone", "limestone"}

    def test_materialdb_raises_if_no_yaml_files_found(self, tmp_path):
        """MaterialManager should raise FileNotFoundError if no YAML files exist in the directory."""
        with pytest.raises(FileNotFoundError):
            material_manager.MaterialManager(data_dir=tmp_path)

    def test_materialdb_skips_invalid_yaml_file(self, tmp_path):
        """MaterialManager should skip YAML files that do not parse to a dict."""
        (tmp_path / "invalid.yml").write_text("just_a_string")

        db = material_manager.MaterialManager(data_dir=tmp_path)
        assert db._list_materials() == []

    def test_materialdb_skips_yaml_file_without_name(
        self, tmp_path, write_yaml
    ):
        """MaterialManager should skip YAML files that do not contain a 'name' field."""
        write_yaml(
            "noname.yml",
            {"properties": {"Density": {"type": "Constant", "value": 2500}}},
        )

        db = material_manager.MaterialManager(data_dir=tmp_path)
        assert db._list_materials() == []

    def test_materialdb_repr_shows_materials_and_ids(
        self, tmp_path, write_yaml
    ):
        """MaterialManager.__repr__ should include subdomain ids and fluid materials in its string output."""
        write_yaml(
            "granite.yml",
            {
                "name": "granite",
                "properties": {"Density": {"type": "Constant", "value": 2700}},
            },
        )
        write_yaml(
            "water.yml",
            {
                "name": "water",
                "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
            },
        )

        db = material_manager.MaterialManager(data_dir=tmp_path)

        # solid + fluid
        filtered = material_manager.MaterialManager(
            data_dir=tmp_path,
            materials={
                "sub1": db.get_material("granite"),
                "AqueousLiquid": db.get_material("water"),
            },
            subdomain_ids={"sub1": 42},
        )
        text = repr(filtered)
        assert "granite" in text
        assert "water" in text
        assert "[42]" in text

        # only solid
        filtered = material_manager.MaterialManager(
            data_dir=tmp_path,
            materials={"sub1": db.get_material("granite")},
            subdomain_ids={"sub1": 42},
        )
        text = repr(filtered)
        assert "granite" in text
        assert "[42]" in text

        # only fluid
        filtered = material_manager.MaterialManager(
            data_dir=tmp_path,
            materials={"AqueousLiquid": db.get_material("water")},
            subdomain_ids={"sub1": 42},
        )
        text = repr(filtered)
        assert "water" in text
        assert "[42]" in text

    def test_materialdb_repr_shows_no_solids_if_subdomain_ids_empty(
        self, tmp_path, write_yaml
    ):
        """MaterialManager.__repr__ should state 'No solid or medium materials defined' if no subdomain_ids exist."""
        write_yaml(
            "fluidy.yml",
            {
                "name": "fluidy",
                "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
            },
        )

        db = material_manager.MaterialManager(
            data_dir=tmp_path,
            materials={
                "AqueousLiquid": material_manager.MaterialManager(
                    data_dir=tmp_path
                ).get_material("fluidy")
            },
            subdomain_ids={},
        )

        text = repr(db)
        assert "No solid or medium materials defined" in text

    def test_materialdb_list_ids_returns_ids(self, tmp_path, write_yaml):
        """MaterialManager.list_ids should return all stored material_ids."""
        write_yaml(
            "sandstone.yml",
            {
                "name": "sandstone",
                "properties": {"Density": {"type": "Constant", "value": 2200}},
            },
        )

        db = material_manager.MaterialManager(
            data_dir=tmp_path,
            materials={
                "subA": material_manager.MaterialManager(
                    data_dir=tmp_path
                ).get_material("sandstone")
            },
            subdomain_ids={"subA": 7},
        )
        assert db._list_ids() == [7]

    def test_materialdb_list_subdomains_returns_names(
        self, tmp_path, write_yaml
    ):
        """MaterialManager.list_subdomains should return all subdomain names."""
        write_yaml(
            "limestone.yml",
            {
                "name": "limestone",
                "properties": {"Density": {"type": "Constant", "value": 2500}},
            },
        )

        db = material_manager.MaterialManager(
            data_dir=tmp_path,
            materials={
                "domainX": material_manager.MaterialManager(
                    data_dir=tmp_path
                ).get_material("limestone")
            },
            subdomain_ids={"domainX": 3},
        )
        assert db._list_subdomains() == ["domainX"]


class TestMaterialManagerFilter:
    def test_materialdb_filter_includes_subdomains(
        self, tmp_path, write_yaml, monkeypatch
    ):
        """MaterialManager.filter should include materials for all given subdomains."""
        write_yaml(
            "clay.yml",
            {
                "name": "clay",
                "properties": {
                    "Density": {"type": "Constant", "value": 2400},
                    "Permeability": {"type": "Constant", "value": 1e-18},
                },
            },
        )
        db = material_manager.MaterialManager(data_dir=tmp_path)

        process_schema = {"properties": ["Density"], "phases": []}
        monkeypatch.setitem(
            material_manager.PROCESS_SCHEMAS, "dummy", process_schema
        )

        subdomains = [
            {"subdomain": "region1", "material": "clay", "material_ids": [0]}
        ]
        filtered = db.filter(process="dummy", subdomains=subdomains, fluids={})

        mat = filtered.get_material("region1")
        assert mat is not None
        assert mat.property_names() == ["Density"]
        assert filtered._list_subdomains() == ["region1"]
        assert filtered._list_ids() == [0]

    def test_materialdb_filter_includes_aqueous_liquid(
        self, tmp_path, write_yaml, monkeypatch
    ):
        """MaterialManager.filter should include an aqueous liquid fluid material."""
        write_yaml(
            "water.yml",
            {
                "name": "water",
                "properties": {
                    "Density": {"type": "Constant", "value": 1000},
                    "Viscosity": {"type": "Constant", "value": 1.0},
                },
            },
        )
        db = material_manager.MaterialManager(data_dir=tmp_path)

        process_schema = {"properties": ["Viscosity"], "phases": []}
        monkeypatch.setitem(
            material_manager.PROCESS_SCHEMAS, "dummy", process_schema
        )

        filtered = db.filter(
            process="dummy", subdomains=[], fluids={"AqueousLiquid": "water"}
        )

        assert "AqueousLiquid" in filtered.fluids
        assert isinstance(
            filtered.fluids["AqueousLiquid"], material_manager.Material
        )

    def test_materialdb_filter_includes_nonaqueous_liquid(
        self, tmp_path, write_yaml, monkeypatch
    ):
        """MaterialManager.filter should include a non-aqueous liquid fluid material."""
        write_yaml(
            "oil.yml",
            {
                "name": "oil",
                "properties": {
                    "Density": {"type": "Constant", "value": 800},
                    "Viscosity": {"type": "Constant", "value": 5.0},
                },
            },
        )
        db = material_manager.MaterialManager(data_dir=tmp_path)

        process_schema = {"properties": ["Viscosity"], "phases": []}
        monkeypatch.setitem(
            material_manager.PROCESS_SCHEMAS, "dummy", process_schema
        )

        filtered = db.filter(
            process="dummy", subdomains=[], fluids={"NonAqueousLiquid": "oil"}
        )

        assert "NonAqueousLiquid" in filtered.fluids
        assert isinstance(
            filtered.fluids["NonAqueousLiquid"], material_manager.Material
        )

    def test_materialdb_filter_raises_if_material_missing(
        self, tmp_path, write_yaml, monkeypatch
    ):
        """MaterialManager.filter should raise ValueError if a referenced material is not in the DB."""
        write_yaml(
            "present.yml",
            {
                "name": "present",
                "properties": {"Density": {"type": "Constant", "value": 1234}},
            },
        )
        db = material_manager.MaterialManager(data_dir=tmp_path)

        process_schema = {"properties": ["Density"], "phases": []}
        monkeypatch.setitem(
            material_manager.PROCESS_SCHEMAS, "dummy", process_schema
        )

        subdomains = [
            {
                "subdomain": ["regionX"],
                "material": "not_here",
                "material_ids": [0],
            }
        ]
        with pytest.raises(
            ValueError, match="Material 'not_here' not found in repository."
        ):
            db.filter(process="dummy", subdomains=subdomains, fluids={})

    def test_materialdb_filter_raises_if_fluid_missing(
        self, tmp_path, write_yaml, monkeypatch
    ):
        """MaterialManager.filter should raise ValueError if a referenced fluid material is not in the DB."""
        write_yaml(
            "rock.yml",
            {
                "name": "rock",
                "properties": {"Density": {"type": "Constant", "value": 2700}},
            },
        )
        db = material_manager.MaterialManager(data_dir=tmp_path)

        process_schema = {"properties": ["Density"], "phases": []}
        monkeypatch.setitem(
            material_manager.PROCESS_SCHEMAS, "dummy", process_schema
        )

        subdomains = [
            {"subdomain": "region1", "material": "rock", "material_ids": [0]}
        ]
        fluids = {"AqueousLiquid": "not_in_db"}

        with pytest.raises(
            ValueError,
            match="Fluid material 'not_in_db' not found in repository.",
        ):
            db.filter(process="dummy", subdomains=subdomains, fluids=fluids)

    def test_materialdb_filter_raises_if_process_not_found(
        self, tmp_path, write_yaml
    ):
        """MaterialManager.filter should raise ValueError if the process name is not in PROCESS_SCHEMAS."""
        write_yaml(
            "rock.yml",
            {
                "name": "rock",
                "properties": {"Density": {"type": "Constant", "value": 2700}},
            },
        )
        db = material_manager.MaterialManager(data_dir=tmp_path)

        subdomains = [
            {"subdomain": "region1", "material": "rock", "material_ids": [0]}
        ]
        with pytest.raises(
            ValueError,
            match="No process schema found for 'nonexistent_process'",
        ):
            db.filter(
                process="nonexistent_process", subdomains=subdomains, fluids={}
            )

    def test_materialdb_filter_and_repr_integration(
        self, tmp_path, write_yaml, monkeypatch
    ):
        """MaterialManager.__repr__ should correctly summarize a filtered database with solids and fluids."""
        write_yaml(
            "clay.yml",
            {
                "name": "clay",
                "properties": {
                    "Density": {"type": "Constant", "value": 2400},
                    "Permeability": {"type": "Constant", "value": 1e-18},
                },
            },
        )
        write_yaml(
            "water.yml",
            {
                "name": "water",
                "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
            },
        )

        db = material_manager.MaterialManager(data_dir=tmp_path)

        process_schema = {"properties": ["Density", "Viscosity"], "phases": []}
        monkeypatch.setitem(
            material_manager.PROCESS_SCHEMAS, "dummy", process_schema
        )

        subdomains = [
            {"subdomain": "region1", "material": "clay", "material_ids": [42]}
        ]
        fluids = {"AqueousLiquid": "water"}

        filtered = db.filter(
            process="dummy", subdomains=subdomains, fluids=fluids
        )
        text = repr(filtered)

        assert "region1" in text
        assert "[42]" in text
        assert "clay" in text
        assert "AqueousLiquid" in text
        assert "water" in text

    def test_required_property_names_collects_all_levels(self):
        """required_property_names should return a set of medium, phase, and component properties."""
        schema = {
            "properties": ["Density", "Porosity"],
            "phases": [
                {"type": "Solid", "properties": ["YoungsModulus"]},
                {
                    "type": "AqueousLiquid",
                    "properties": ["Viscosity"],
                    "components": {
                        "H2O": ["MolarMass"],
                        "NaCl": ["DiffusionCoefficient"],
                    },
                },
            ],
        }
        result = required_properties.required_property_names(schema)

        assert result == {
            "Density",
            "Porosity",
            "YoungsModulus",
            "Viscosity",
            "MolarMass",
            "DiffusionCoefficient",
        }

    def test_filter_preserves_scope(self, make_material):
        """Filtering a Material must preserve extra fields like 'scope'."""

        # Material mit Phase- und Medium-Scoped thermal_conductivity
        mat = make_material(
            {
                "thermal_conductivity": [
                    {"type": "Constant", "value": 1.7, "scope": "phase"},
                    {
                        "type": "EffectiveThermalConductivityPorosityMixing",
                        "scope": "medium",
                    },
                ]
            },
            name="mock",
        )

        # Nur thermal_conductivity erlauben
        filtered = mat.filter_properties({"thermal_conductivity"})

        # Check, dass beide Varianten drinbleiben
        scopes = [p.extra.get("scope") for p in filtered.properties]
        assert "phase" in scopes
        assert "medium" in scopes


class TestMedia:

    def test_media_create_no_process(self, tmp_path, write_yaml):
        """MediaSet should not be creatable from an unfiltered MaterialManager."""
        write_yaml(
            "clay.yml",
            {
                "name": "clay",
                "properties": {"Density": {"type": "Constant", "value": 2400}},
            },
        )
        write_yaml(
            "water.yml",
            {
                "name": "water",
                "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
            },
        )

        db = material_manager.MaterialManager(data_dir=tmp_path)
        with pytest.raises(
            ValueError,
            match="MediaSet can only be created from a filtered MaterialManager.",
        ):
            MediaSet(db)

    def test_media_create_and_repr(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
    ):
        """MediaSet should be creatable from a filtered MaterialManager and show correct repr output."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)

        # minimal sanity checks
        assert len(media) == 1
        text = repr(media)

        # MediaSet-level
        assert "MediaSet with 1 entries" in text

        # Medium-level
        assert "<Medium 'region1' (ID=1)>" in text
        assert "1 medium properties" in text
        assert "Density (Constant)" in text

        # Phase-level
        assert "1 phase(s)" in text
        assert "<Phase 'AqueousLiquid'>" in text
        assert "Viscosity (Constant)" in text

        # Components
        assert "no components" in text

    def test_media_len_and_iter(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
    ):
        """len() and iteration should reflect the number of Medium entries."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)

        assert len(media) == 1
        mediums = list(iter(media))
        assert len(mediums) == 1
        assert mediums[0].name == "region1"

    def test_media_getitem_and_contains(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
    ):
        """__getitem__ and __contains__ should allow lookup by subdomain name."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)

        assert "region1" in media
        medium = media["region1"]
        assert medium.material_id == 1

    def test_media_keys_values_items(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
    ):
        """keys(), values(), items() should behave like a dict."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)

        keys = media.keys()
        assert keys == ["region1"]

        values = media.values()
        assert len(values) == 1
        assert values[0].name == "region1"

        items = media.items()
        assert items[0][0] == "region1"
        assert items[0][1].material_id == 1

    def test_media_get_by_id(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
    ):
        """get_by_id should return the correct Medium or None."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)

        medium = media.get_by_id(1)
        assert medium is not None
        assert medium.name == "region1"

        assert media.get_by_id(99) is None

    def test_media_to_dict(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
    ):
        """to_dict should return a dict with subdomain names as keys and Mediums as values."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)

        dct = media.to_dict()
        assert "region1" in dct
        assert dct["region1"].material_id == 1

    def test_media_validate_medium_called(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
    ):
        """validate() should call validate() on each Medium."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)

        called = {"ok": False}

        def fake_validate():
            called["ok"] = True
            return True

        media._media[0].validate = fake_validate  # override
        media.validate()

        assert called["ok"] is True

    def test_media_builds_phases_from_fluids(self, make_filtered_db):
        """MediaSet should build Mediums with correct phases when fluids are provided."""
        materials = {
            "clay": {
                "name": "clay",
                "properties": {"Density": {"type": "Constant", "value": 2400}},
            },
            "water": {
                "name": "water",
                "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
            },
        }
        schema = {
            "properties": ["Density"],
            "phases": [{"type": "AqueousLiquid", "properties": ["Viscosity"]}],
        }
        subdomains = [
            {"subdomain": "region1", "material": "clay", "material_ids": [0]}
        ]
        fluids = {"AqueousLiquid": "water"}

        filtered = make_filtered_db(materials, schema, subdomains, fluids)
        media = MediaSet(filtered)

        assert len(media) == 1
        medium = next(iter(media))
        assert medium.aqueous is not None, "AqueousLiquid phase should exist"
        assert medium.aqueous.type == "AqueousLiquid"
        assert len(medium.aqueous.properties) == 1
        assert medium.aqueous.properties[0].name == "Viscosity"
        assert medium.properties[0].name == "Density"


class TestComponent:
    def test_component_initializes_with_schema(
        self, make_material, monkeypatch
    ):
        """Component should initialize and filter properties according to schema."""
        mat = make_material(
            {"Viscosity": {"type": "Constant", "value": 1.0}}, name="water"
        )

        schema = {
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "components": {"Solvent": ["Viscosity"]},
                }
            ]
        }
        monkeypatch.setitem(component.PROCESS_SCHEMAS, "dummy", schema)

        comp = component.Component(
            mat, "AqueousLiquid", "Solvent", "dummy", diffusion_coefficient=0.0
        )
        assert comp.validate() is True
        assert [p.name for p in comp.properties] == ["Viscosity"]

    def test_component_raises_if_schema_missing(self, make_material):
        """Component should raise ValueError if process schema is not found."""
        mat = make_material(
            {"Viscosity": {"type": "Constant", "value": 1.0}}, name="water"
        )

        with pytest.raises(
            ValueError, match="No process schema found for 'no_such_process'."
        ):
            component.Component(
                mat, "AqueousLiquid", "Solvent", "no_such_process", 0.0
            )

    def test_component_inserts_diffusion_property_for_vapour_role(
        self, make_material, monkeypatch
    ):
        """Component should insert diffusion property when phase=Gas and role=Vapour."""
        mat = make_material({}, name="co2")

        schema = {
            "phases": [{"type": "Gas", "components": {"Vapour": ["diffusion"]}}]
        }
        monkeypatch.setitem(component.PROCESS_SCHEMAS, "dummy", schema)

        comp = component.Component(
            mat, "Gas", "Vapour", "dummy", diffusion_coefficient=1.23e-5
        )
        assert any(p.name == "diffusion" for p in comp.properties)
        assert comp.D == 1.23e-5

    def test_component_validate_raises_for_wrong_phase(
        self, make_material, monkeypatch
    ):
        """Component.validate should raise if phase type not in schema."""
        mat = make_material(
            {"Viscosity": {"type": "Constant", "value": 1.0}}, name="water"
        )

        schema = {
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "components": {"Solvent": ["Viscosity"]},
                }
            ]
        }
        monkeypatch.setitem(component.PROCESS_SCHEMAS, "dummy", schema)

        comp = component.Component(mat, "Gas", "Solvent", "dummy", 0.0)

        with pytest.raises(
            ValueError,
            match="Component 'water' is in a phase 'Gas' not allowed for process.",
        ):
            comp.validate()

    def test_component_repr_contains_name_and_role(
        self, make_material, monkeypatch
    ):
        """__repr__ should include component name and role."""
        mat = make_material(
            {"Viscosity": {"type": "Constant", "value": 1.0}}, name="water"
        )

        schema = {
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "components": {"Solvent": ["Viscosity"]},
                }
            ]
        }
        monkeypatch.setitem(component.PROCESS_SCHEMAS, "dummy", schema)

        comp = component.Component(
            mat, "AqueousLiquid", "Solvent", "dummy", 0.0
        )
        text = repr(comp)
        assert "Component 'water'" in text
        assert "Role: Solvent" in text
        assert "Viscosity" in text


class TestComponents:
    def test_components_initializes_aqueousliquid(
        self, make_material, monkeypatch
    ):
        """Components should initialize Solute and Solvent for AqueousLiquid phase."""
        gas = make_material(
            {"Henry": {"type": "Constant", "value": 1.0}}, name="co2"
        )
        liquid = make_material(
            {"Viscosity": {"type": "Constant", "value": 1.0}}, name="water"
        )

        schema = {
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "components": {"Solute": [], "Solvent": ["Viscosity"]},
                },
            ]
        }
        monkeypatch.setitem(component.PROCESS_SCHEMAS, "dummy", schema)

        comps = components.Components(
            "AqueousLiquid", gas, liquid, "dummy", Diffusion_coefficient=1.0
        )
        assert comps.gas_component_obj.role == "Solute"
        assert comps.liquid_component_obj.role == "Solvent"

    def test_components_initializes_gas(self, make_material, monkeypatch):
        """Components should initialize Carrier and Vapour for Gas phase."""
        gas = make_material(
            {"MolarMass": {"type": "Constant", "value": 28}}, name="N2"
        )
        liquid = make_material(
            {"MolarMass": {"type": "Constant", "value": 18}}, name="H2O"
        )

        schema = {
            "phases": [
                {
                    "type": "Gas",
                    "components": {
                        "Carrier": ["MolarMass"],
                        "Vapour": ["MolarMass"],
                    },
                },
            ]
        }
        monkeypatch.setitem(component.PROCESS_SCHEMAS, "dummy", schema)

        comps = components.Components(
            "Gas", gas, liquid, "dummy", Diffusion_coefficient=1.0
        )
        assert comps.gas_component_obj.role == "Carrier"
        assert comps.liquid_component_obj.role == "Vapour"

    def test_components_raises_on_invalid_phase(self, make_material):
        """Components should raise ValueError if phase_type is not supported."""
        gas = make_material({}, name="co2")
        liquid = make_material({}, name="water")

        with pytest.raises(ValueError, match="Unsupported phase_type: Solid"):
            components.Components("Solid", gas, liquid, "dummy")

    def test_components_repr_includes_phase_and_components(
        self, make_material, monkeypatch
    ):
        """__repr__ should include phase type and component details."""
        gas = make_material(
            {"Henry": {"type": "Constant", "value": 1.0}}, name="co2"
        )
        liquid = make_material(
            {"Viscosity": {"type": "Constant", "value": 1.0}}, name="water"
        )

        schema = {
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "components": {"Solute": [], "Solvent": ["Viscosity"]},
                },
            ]
        }
        monkeypatch.setitem(component.PROCESS_SCHEMAS, "dummy", schema)

        comps = components.Components(
            "AqueousLiquid", gas, liquid, "dummy", Diffusion_coefficient=1.0
        )
        text = repr(comps)
        assert "AqueousLiquid" in text
        assert "co2" in text or "water" in text

    def test_get_binary_diffusion_coefficient_reads_file(
        self, make_material, tmp_path, monkeypatch
    ):
        """get_binary_diffusion_coefficient should read coefficient from YAML file."""
        schema = {
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "components": {"Solute": [], "Solvent": []},
                },
            ]
        }
        monkeypatch.setitem(component.PROCESS_SCHEMAS, "dummy", schema)

        diffusion_data = {"AqueousLiquid": {"water": {"co2": 1.9e-9}}}
        (tmp_path / "diffusion_coefficients.yml").write_text(
            yaml.safe_dump(diffusion_data)
        )
        monkeypatch.setattr(components.defs, "MATERIALS_DIR", str(tmp_path))

        gas = make_material({}, name="co2")
        liquid = make_material({}, name="water")

        comps = components.Components(
            "AqueousLiquid", gas, liquid, "dummy", Diffusion_coefficient=None
        )
        D = comps.get_binary_diffusion_coefficient("AqueousLiquid")
        assert D == 1.9e-9

    def test_get_binary_diffusion_coefficient_unit_raises_if_missing(
        self, make_material, tmp_path, monkeypatch
    ):
        """get_binary_diffusion_coefficient should raise if YAML entry is missing."""
        diffusion_data = {"AqueousLiquid": {"water": {}}}
        (tmp_path / "diffusion_coefficients.yml").write_text(
            yaml.safe_dump(diffusion_data)
        )
        monkeypatch.setattr(components.defs, "MATERIALS_DIR", str(tmp_path))

        fake = components.Components.__new__(components.Components)
        fake.gas_component = make_material({}, name="co2")
        fake.liquid_component = make_material({}, name="water")

        with pytest.raises(
            ValueError,
            match="No AqueousLiquid-phase diffusion coefficient found",
        ) as excinfo:
            fake.get_binary_diffusion_coefficient("AqueousLiquid")

        assert "No AqueousLiquid-phase diffusion coefficient" in str(
            excinfo.value
        )

    def test_get_binary_diffusion_coefficient_integration_fails_on_init(
        self, make_material, tmp_path, monkeypatch
    ):
        """Integration test: Components.__init__ should fail when diffusion coefficient is missing."""
        schema = {
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "components": {"Solute": [], "Solvent": []},
                },
            ]
        }
        monkeypatch.setitem(component.PROCESS_SCHEMAS, "dummy", schema)

        diffusion_data = {"AqueousLiquid": {"water": {}}}
        (tmp_path / "diffusion_coefficients.yml").write_text(
            yaml.safe_dump(diffusion_data)
        )
        monkeypatch.setattr(components.defs, "MATERIALS_DIR", str(tmp_path))

        gas = make_material({}, name="co2")
        liquid = make_material({}, name="water")

        with pytest.raises(
            ValueError,
            match="No AqueousLiquid-phase diffusion coefficient found for the pair 'water / co2'",
        ) as excinfo:
            components.Components(
                "AqueousLiquid",
                gas,
                liquid,
                "dummy",
                Diffusion_coefficient=None,
            )

        assert "No AqueousLiquid-phase diffusion coefficient" in str(
            excinfo.value
        )

    def test_get_binary_diffusion_coefficient_integration_success(
        self, make_material, tmp_path, monkeypatch
    ):
        """Integration test: Components.__init__ should assign diffusion properties for both gas and vapour."""
        schema = {
            "phases": [
                {
                    "type": "Gas",
                    "components": {
                        "Carrier": ["diffusion"],
                        "Vapour": ["diffusion"],
                    },
                },
            ]
        }
        monkeypatch.setitem(component.PROCESS_SCHEMAS, "dummy", schema)

        diffusion_data = {"Gas": {"CO2": {"H2O": 1.0e-6}}}
        (tmp_path / "diffusion_coefficients.yml").write_text(
            yaml.safe_dump(diffusion_data)
        )
        monkeypatch.setattr(components.defs, "MATERIALS_DIR", str(tmp_path))

        gas = make_material({}, name="CO2")
        liquid = make_material({}, name="H2O")

        comps = components.Components(
            "Gas", gas, liquid, "dummy", Diffusion_coefficient=None
        )

        assert comps.gas_component_obj is not None
        assert comps.liquid_component_obj is not None

        D = comps.get_binary_diffusion_coefficient("Gas")
        assert pytest.approx(1.0e-6) == D
        assert pytest.approx(1.0e-6) == comps.liquid_component_obj.D

        gas_props = [p.name for p in comps.gas_component_obj.properties]
        liquid_props = [p.name for p in comps.liquid_component_obj.properties]
        assert "diffusion" in gas_props
        assert "diffusion" in liquid_props


class TestMedium:
    def test_validate_returns_true(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
    ):
        """Medium.validate() should return True if properties and phases are valid."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)
        medium = media["region1"]
        assert medium.validate() is True

    def test_validate_medium_raises_if_invalid(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
    ):
        """Medium.validate() should return True if properties and phases are valid."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)
        medium = media["region1"]

        # validate() override to return False
        medium.validate = lambda: False

        # MediaSet.validate_medium should now raise ValueError
        with pytest.raises(
            ValueError, match="Medium 'region1' failed validation."
        ):
            media.validate_medium(medium)

    def test_validate_medium_raises_if_broken_after_creation(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
    ):
        """MediaSet.validate_medium should raise ValueError if a Medium becomes invalid after creation."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)
        medium = media["region1"]

        # break the medium intentionally
        medium.properties.clear()

        with pytest.raises(
            ValueError,
            match=r"Medium 'region1' is missing required properties: \['Density'\]",
        ) as exc:
            media.validate_medium(medium)

        assert "missing required properties" in str(exc.value).lower()

    def test_media_import_exports_media_block(
        self,
        make_filtered_db,
        default_materials,
        default_schema,
        default_subdomains,
        default_fluids,
        tmp_path,
    ):
        """Project.set_media should import a MediaSet and export a <media> block into the Project XML with correct content."""
        filtered = make_filtered_db(
            default_materials,
            default_schema,
            default_subdomains,
            default_fluids,
        )
        media = MediaSet(filtered)

        prj = Project()
        prj.set_media(media)  # statt media.to_project(prj)

        xml_file = tmp_path / "test_export.prj"
        prj.write_input(xml_file)
        xml_text = xml_file.read_text()

        assert "<media>" in xml_text
        assert '<medium id="1">' in xml_text or '<medium id="0">' in xml_text
        assert "<name>Density</name>" in xml_text
        assert "<value>2400</value>" in xml_text
        assert "<type>AqueousLiquid</type>" in xml_text
        assert "<name>Viscosity</name>" in xml_text
        assert "<value>1.0</value>" in xml_text

    def test_media_import_exports_with_components_two_phase(
        self, make_filtered_db, tmp_path
    ):
        """Project.set_media should export a TH2M-style Medium with Solid + Gas + AqueousLiquid phases and components."""
        materials = {
            "clay": {
                "name": "clay",
                "properties": {"Density": {"type": "Constant", "value": 2400}},
            },
            "CO2": {
                "name": "CO2",
                "properties": {"Density": {"type": "Constant", "value": 1.8}},
            },
            "H2O": {
                "name": "H2O",
                "properties": {"Density": {"type": "Constant", "value": 1.0}},
            },
        }
        schema = {
            "properties": ["Density"],
            "phases": [
                {"type": "Solid", "properties": ["Density"], "components": {}},
                {
                    "type": "Gas",
                    "properties": [],
                    "components": {
                        "Carrier": ["diffusion"],
                        "Vapour": ["diffusion"],
                    },
                },
                {
                    "type": "AqueousLiquid",
                    "properties": [],
                    "components": {
                        "Solute": ["diffusion"],
                        "Solvent": ["diffusion"],
                    },
                },
            ],
        }
        diffusion = {
            "Gas": {"CO2": {"H2O": 1.0e-6}},
            "AqueousLiquid": {"H2O": {"CO2": 1.9e-9}},
        }
        subdomains = [
            {"subdomain": "region1", "material": "clay", "material_ids": [0]}
        ]
        fluids = {"Gas": "CO2", "AqueousLiquid": "H2O"}

        filtered = make_filtered_db(
            materials, schema, subdomains, fluids, diffusion=diffusion
        )
        media = MediaSet(filtered)

        prj = Project()
        prj.set_media(media)

        xml_file = tmp_path / "export.prj"
        prj.write_input(xml_file)
        xml_str = xml_file.read_text()

        # MediaSet + Medium
        assert "<media>" in xml_str
        assert '<medium id="0">' in xml_str

        # Medium-level property
        assert "<name>Density</name>" in xml_str
        assert "<value>2400</value>" in xml_str

        # Solid phase
        assert "<type>Solid</type>" in xml_str

        # AqueousLiquid phase + components
        assert "<type>AqueousLiquid</type>" in xml_str
        assert "<component>" in xml_str
        assert "<name>CO2</name>" in xml_str
        assert "<name>H2O</name>" in xml_str

        # Gas phase
        assert "<type>Gas</type>" in xml_str

        # Diffusion values
        assert "<value>0.0</value>" in xml_str
        assert "<value>1e-06</value>" in xml_str


class TestMediumPhaseMatrix:
    """Covers the four canonical phase/component configurations in Medium._build_phases."""

    def test_one_phase_no_components(self, make_filtered_db):
        """Case 1: One phase, no components (AqueousLiquid only)."""
        materials = {
            "clay": {
                "name": "clay",
                "properties": {"Density": {"type": "Constant", "value": 2400}},
            },
            "water": {
                "name": "water",
                "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
            },
        }
        schema = {
            "properties": ["Density"],
            "phases": [{"type": "AqueousLiquid", "properties": ["Viscosity"]}],
        }
        subdomains = [
            {"subdomain": "region1", "material": "clay", "material_ids": [0]}
        ]
        fluids = {"AqueousLiquid": "water"}

        filtered = make_filtered_db(materials, schema, subdomains, fluids)
        media = MediaSet(filtered)

        medium = next(iter(media))
        aqueous = medium.get_phase("AqueousLiquid")
        assert aqueous is not None
        assert (
            not hasattr(aqueous, "components") or len(aqueous.components) == 0
        )

    def test_two_phase_no_components(self, make_filtered_db):
        """Case 2: Two phases, no components (Gas + AqueousLiquid)."""
        materials = {
            "clay": {
                "name": "clay",
                "properties": {"Density": {"type": "Constant", "value": 2400}},
            },
            "water": {
                "name": "water",
                "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
            },
            "co2": {
                "name": "co2",
                "properties": {
                    "MolarMass": {"type": "Constant", "value": 44.0}
                },
            },
        }
        schema = {
            "properties": ["Density"],
            "phases": [
                {"type": "AqueousLiquid", "properties": ["Viscosity"]},
                {"type": "Gas", "properties": ["MolarMass"]},
            ],
        }
        subdomains = [
            {"subdomain": "region1", "material": "clay", "material_ids": [0]}
        ]
        fluids = {"AqueousLiquid": "water", "Gas": "co2"}

        filtered = make_filtered_db(materials, schema, subdomains, fluids)
        media = MediaSet(filtered)

        medium = next(iter(media))
        assert medium.get_phase("AqueousLiquid") is not None
        assert medium.get_phase("Gas") is not None

    def test_two_phase_with_components(self, make_filtered_db):
        """Case 3: Two phases with components (TH2M style)."""
        materials = {
            "clay": {
                "name": "clay",
                "properties": {"Density": {"type": "Constant", "value": 2400}},
            },
            "water": {
                "name": "water",
                "properties": {
                    "specific_heat_capacity": {"type": "Constant", "value": 1.0}
                },
            },
            "co2": {
                "name": "co2",
                "properties": {
                    "molar_mass": {"type": "Constant", "value": 44.0}
                },
            },
        }
        schema = {
            "properties": ["Density"],
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "properties": [],
                    "components": {
                        "Solute": ["diffusion"],
                        "Solvent": ["specific_heat_capacity"],
                    },
                },
                {
                    "type": "Gas",
                    "properties": [],
                    "components": {
                        "Carrier": ["molar_mass"],
                        "Vapour": ["diffusion"],
                    },
                },
            ],
        }
        diffusion = {
            "Gas": {"co2": {"water": 1e-6}},
            "AqueousLiquid": {"water": {"co2": 1.9e-9}},
        }
        subdomains = [
            {"subdomain": "region1", "material": "clay", "material_ids": [0]}
        ]
        fluids = {"AqueousLiquid": "water", "Gas": "co2"}

        filtered = make_filtered_db(
            materials, schema, subdomains, fluids, diffusion=diffusion
        )
        media = MediaSet(filtered)

        medium = next(iter(media))
        gas = medium.get_phase("Gas")
        aq = medium.get_phase("AqueousLiquid")

        assert gas is not None
        assert aq is not None

        assert any(
            prop.name == "diffusion"
            for comp in gas.components
            for prop in comp.properties
        )
        assert any(
            prop.name == "diffusion"
            for comp in aq.components
            for prop in comp.properties
        )

    def test_two_phase_with_components_allows_empty_carrier(
        self, make_filtered_db
    ):
        """Gas phase schema with Carrier: [] should still create a valid Component without properties."""
        materials = {
            "clay": {
                "name": "clay",
                "properties": {"Density": {"type": "Constant", "value": 2400}},
            },
            "water": {
                "name": "water",
                "properties": {
                    "specific_heat_capacity": {"type": "Constant", "value": 1.0}
                },
            },
            "co2": {
                "name": "co2",
                "properties": {
                    "molar_mass": {"type": "Constant", "value": 44.0}
                },
            },
        }
        schema = {
            "properties": ["Density"],
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "properties": [],
                    "components": {
                        "Solute": ["diffusion"],
                        "Solvent": ["specific_heat_capacity"],
                    },
                },
                {
                    "type": "Gas",
                    "properties": [],
                    "components": {"Carrier": [], "Vapour": ["diffusion"]},
                },
            ],
        }
        diffusion = {
            "Gas": {"co2": {"water": 1e-6}},
            "AqueousLiquid": {"water": {"co2": 1.9e-9}},
        }
        subdomains = [
            {"subdomain": "region1", "material": "clay", "material_ids": [0]}
        ]
        fluids = {"AqueousLiquid": "water", "Gas": "co2"}

        filtered = make_filtered_db(
            materials, schema, subdomains, fluids, diffusion=diffusion
        )
        media = MediaSet(filtered)

        medium = next(iter(media))
        gas = medium.get_phase("Gas")
        aq = medium.get_phase("AqueousLiquid")
        assert gas is not None
        assert aq is not None

        assert any(
            prop.name == "diffusion"
            for comp in gas.components
            for prop in comp.properties
        )
        assert any(
            prop.name == "diffusion"
            for comp in aq.components
            for prop in comp.properties
        )

    def test_two_phase_with_components_allows_empty_solvent(
        self, make_filtered_db
    ):
        """AqueousLiquid phase schema with Solvent: [] should still create a valid Component without properties."""
        materials = {
            "clay": {
                "name": "clay",
                "properties": {"Density": {"type": "Constant", "value": 2400}},
            },
            "water": {
                "name": "water",
                "properties": {
                    "specific_heat_capacity": {"type": "Constant", "value": 1.0}
                },
            },
            "co2": {
                "name": "co2",
                "properties": {
                    "molar_mass": {"type": "Constant", "value": 44.0}
                },
            },
        }
        schema = {
            "properties": ["Density"],
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "properties": [],
                    "components": {"Solute": ["diffusion"], "Solvent": []},
                },
                {
                    "type": "Gas",
                    "properties": [],
                    "components": {
                        "Carrier": ["molar_mass"],
                        "Vapour": ["diffusion"],
                    },
                },
            ],
        }
        diffusion = {
            "Gas": {"co2": {"water": 1e-6}},
            "AqueousLiquid": {"water": {"co2": 1.9e-9}},
        }
        subdomains = [
            {"subdomain": "region1", "material": "clay", "material_ids": [0]}
        ]
        fluids = {"AqueousLiquid": "water", "Gas": "co2"}

        filtered = make_filtered_db(
            materials, schema, subdomains, fluids, diffusion=diffusion
        )
        media = MediaSet(filtered)

        medium = next(iter(media))
        gas = medium.get_phase("Gas")
        aq = medium.get_phase("AqueousLiquid")
        assert gas is not None
        assert aq is not None
        assert any(
            prop.name == "diffusion"
            for comp in gas.components
            for prop in comp.properties
        )
        assert any(
            prop.name == "diffusion"
            for comp in aq.components
            for prop in comp.properties
        )

    def test_two_phase_with_components_allows_empty_roles(
        self, make_filtered_db
    ):
        """Two-phase schema with empty component roles should still yield valid Components without properties."""
        materials = {
            "clay": {
                "name": "clay",
                "properties": {"Density": {"type": "Constant", "value": 2400}},
            },
            "water": {
                "name": "water",
                "properties": {
                    "specific_heat_capacity": {"type": "Constant", "value": 1.0}
                },
            },
            "co2": {
                "name": "co2",
                "properties": {
                    "molar_mass": {"type": "Constant", "value": 44.0}
                },
            },
        }
        schema = {
            "properties": ["Density"],
            "phases": [
                {
                    "type": "AqueousLiquid",
                    "properties": [],
                    "components": {"Solute": ["diffusion"], "Solvent": []},
                },
                {
                    "type": "Gas",
                    "properties": [],
                    "components": {"Carrier": [], "Vapour": ["diffusion"]},
                },
            ],
        }
        diffusion = {
            "Gas": {"co2": {"water": 1e-6}},
            "AqueousLiquid": {"water": {"co2": 1.9e-9}},
        }
        subdomains = [
            {"subdomain": "region1", "material": "clay", "material_ids": [0]}
        ]
        fluids = {"AqueousLiquid": "water", "Gas": "co2"}

        filtered = make_filtered_db(
            materials, schema, subdomains, fluids, diffusion=diffusion
        )
        media = MediaSet(filtered)

        medium = next(iter(media))
        gas = medium.get_phase("Gas")
        aq = medium.get_phase("AqueousLiquid")
        assert gas is not None
        assert aq is not None
        assert any(
            prop.name == "diffusion"
            for comp in gas.components
            for prop in comp.properties
        )
        assert any(
            prop.name == "diffusion"
            for comp in aq.components
            for prop in comp.properties
        )

    @pytest.mark.xfail(reason="Single-phase multi-component not yet supported")
    def test_one_phase_with_components_xfail(self, make_filtered_db):
        """Case 4: One phase with multiple components should raise NotImplementedError."""
        materials = {
            "water": {
                "name": "water",
                "properties": {"Viscosity": {"type": "Constant", "value": 1.0}},
            },
            "o2": {
                "name": "o2",
                "properties": {
                    "MolarMass": {"type": "Constant", "value": 32.0}
                },
            },
            "n2": {
                "name": "n2",
                "properties": {
                    "MolarMass": {"type": "Constant", "value": 28.0}
                },
            },
        }
        schema = {
            "properties": [],
            "phases": [
                {
                    "type": "Gas",
                    "components": {"O2": ["diffusion"], "N2": ["diffusion"]},
                }
            ],
        }
        subdomains = []
        fluids = {"Gas": "o2"}  # intentionally incomplete

        filtered = make_filtered_db(materials, schema, subdomains, fluids)
        with pytest.raises(NotImplementedError):
            MediaSet(filtered)


class TestBuiltinProcessSchemas:
    def test_all_schemas_are_dicts(self):
        """All process schemas should be dicts with 'phases' and 'properties' keys."""
        for name, schema in process_schema.PROCESS_SCHEMAS.items():
            assert isinstance(schema, dict), f"{name} schema must be a dict"
            assert "phases" in schema, f"{name} schema missing 'phases'"
            assert "properties" in schema, f"{name} schema missing 'properties'"
            assert isinstance(schema["phases"], list)
            assert isinstance(schema["properties"], list)

    def test_phases_have_required_structure(self):
        """Every phase must have a 'type' and optional 'properties' and 'components'."""
        for name, schema in process_schema.PROCESS_SCHEMAS.items():
            for phase in schema["phases"]:
                assert "type" in phase, f"{name} has phase without type"
                assert isinstance(phase["type"], str)
                assert (
                    "properties" in phase
                ), f"{name} phase {phase['type']} missing 'properties'"
                assert isinstance(phase["properties"], list)

                # components are optional but if present must be dict[str, list]
                if "components" in phase:
                    comps = phase["components"]
                    assert isinstance(comps, dict)
                    for role, props in comps.items():
                        assert isinstance(role, str)
                        assert isinstance(props, list)
                        # no empty property names
                        assert all(isinstance(p, str) and p for p in props)

    def test_properties_are_strings(self):
        """Top-level and phase properties must always be non-empty strings."""
        for name, schema in process_schema.PROCESS_SCHEMAS.items():
            for p in schema["properties"]:
                assert isinstance(
                    p, str
                ), f"{name} has non-string property {p!r}"
                assert p.strip(), f"{name} has empty/whitespace property {p!r}"
            for phase in schema["phases"]:
                for p in phase.get("properties", []):
                    assert isinstance(p, str)
                    assert p.strip()

    def test_no_duplicate_phase_types_in_schema(self):
        """Each schema should not contain the same phase type multiple times."""
        for name, schema in process_schema.PROCESS_SCHEMAS.items():
            types = [phase["type"] for phase in schema["phases"]]
            assert len(types) == len(
                set(types)
            ), f"{name} has duplicate phases {types}"

    def test_specific_known_schema_th2m(self):
        """TH2M_PT schema should contain Solid, Gas, and AqueousLiquid phases with components."""
        th2m = process_schema.PROCESS_SCHEMAS["TH2M_PT"]
        types = {phase["type"] for phase in th2m["phases"]}
        assert {"Solid", "Gas", "AqueousLiquid"}.issubset(types)

        # Gas must have Carrier + Vapour components
        gas = next(p for p in th2m["phases"] if p["type"] == "Gas")
        assert "components" in gas
        assert set(gas["components"].keys()) == {"Carrier", "Vapour"}

        # AqueousLiquid must have Solute + Solvent
        aq = next(p for p in th2m["phases"] if p["type"] == "AqueousLiquid")
        assert "components" in aq
        assert set(aq["components"].keys()) == {"Solute", "Solvent"}


class TestOgstoolsInternalDB:

    def test_materialdb_loads_builtin_materials(self):
        """Check that the internal material DB loads non-empty set of materials."""
        db = material_manager.MaterialManager()
        assert len(db.materials_db) > 0

    def test_media_import_with_builtin_schema(self, tmp_path):
        """Integration: Project can import builtin DB + schema."""
        db = material_manager.MaterialManager()
        # pick ein Schema + Subdomain
        # schema_def = process_schema.PROCESS_SCHEMAS["TH2M_PT"]
        subdomains = [
            {
                "subdomain": "region1",
                "material": "bentonite",
                "material_ids": [0],
            }
        ]
        fluids = {"AqueousLiquid": "water", "Gas": "carbon_dioxide"}

        filtered = db.filter(
            process="TH2M_PT", subdomains=subdomains, fluids=fluids
        )
        from ogstools.materiallib.core.media import MediaSet

        media = MediaSet(filtered)

        prj = Project()
        prj.set_media(media)

        xml_file = tmp_path / "internal_test.prj"
        prj.write_input(xml_file)
        text = xml_file.read_text()

        assert "<media>" in text
        assert "AqueousLiquid" in text
        assert "Gas" in text

    def create_temporary_mesh(self, tmp_path):

        mesh_path = tmp_path / "domain.msh"

        ot.meshlib.rect(
            (6, 4),
            5,
            n_layers=2,
            structured_grid=True,
            order=1,
            out_name=mesh_path,
            jiggle=0.06,
        )
        meshes = ot.Meshes.from_gmsh(mesh_path)
        meshes.save(tmp_path)

    @pytest.mark.system()
    def test_media_project_import_and_run(self, tmp_path):
        """System test: build MediaSet from builtin DB, PRJ import media, and run OGS."""

        # --- Setup: Materials and Schema ---
        db = material_manager.MaterialManager()
        process = "TH2M_PT"

        subdomains = [
            {
                "subdomain": "test_subdomain_1",
                "material": "opalinus_clay",
                "material_ids": [0],
            },
            {
                "subdomain": "test_subdomain_2",
                "material": "bentonite",
                "material_ids": [1],
            },
        ]
        fluids = {"AqueousLiquid": "water", "Gas": "carbon_dioxide"}

        # --- Build MediaSet ---
        filtered = db.filter(
            process=process, subdomains=subdomains, fluids=fluids
        )
        media = MediaSet(filtered)

        # --- Prepare Project ---
        prj_file = tmp_path / "default.prj"

        print(f"Prj file: {prj_file}")
        model = Project(
            output_file=prj_file, input_file=examples.prj_th2m_phase_transition
        )

        # inject <media> section
        model.set_media(media)
        model.write_input()

        # Create mesh and export as VTU
        self.create_temporary_mesh(tmp_path)

        # --- Run OGS ---
        model.run_model(write_logs=True, args=f"-m {tmp_path} -o {tmp_path}")

        # --- Check run result ---
        if model.process is None:
            pytest.fail("OGS process did not start.")

        rc = (
            model.process.returncode
            if model.process.poll() is not None
            else model.process.wait()
        )
        if rc != 0:
            log = Path(model.logfile)
            tail = (
                log.read_text(encoding="utf-8").splitlines()[-80:]
                if log.exists()
                else []
            )
            pytest.fail(
                f"OGS exited with code {rc}.\n"
                + (
                    "Log tail:\n" + "\n".join(tail)
                    if tail
                    else "No logfile found."
                )
            )

        # --- Sanity checks ---
        text = prj_file.read_text()
        assert "<media>" in text
        assert "AqueousLiquid" in text
        assert "Gas" in text
        assert "opalinus_clay" not in text  # only IDs are exported
