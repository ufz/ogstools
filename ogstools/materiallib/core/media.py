# ogstools/materiallib/core/media.py

from collections import defaultdict
from collections.abc import Iterator

from ogstools.materiallib.core.material_list import MaterialList
from ogstools.materiallib.core.medium import Medium
from ogstools.ogs6py import Project
from ogstools.ogs6py.media import (
    Media as OGS6pyMedia,
)


class Media:
    def __init__(self, materials_list: MaterialList):
        self._media: list[Medium] = []
        self._name_map: dict[str, Medium] = {}
        self.process = materials_list.process

        for name in sorted(
            materials_list.materials,
            key=lambda n: materials_list.material_ids[n],
        ):
            mat = materials_list.materials[name]
            mat_id = materials_list.material_ids[name]

            medium = Medium(
                material_id=mat_id,
                material=mat,
                name=name,
                fluids=materials_list.fluid_materials,
                process=self.process,
            )

            if not self.validate_medium(medium):
                msg = f"Medium '{name}' (ID={mat_id}) is invalid."
                raise ValueError(msg)

            self._media.append(medium)
            self._name_map[name] = medium

            # self.validate()

    def __iter__(self) -> Iterator[Medium]:
        return iter(self._media)

    def __getitem__(self, key: str) -> Medium:
        return self._name_map[key]

    def __len__(self) -> int:
        return len(self._media)

    def __contains__(self, name: str) -> bool:
        return name in self._name_map

    def keys(self) -> list[str]:
        return list(self._name_map.keys())

    def values(self) -> list[Medium]:
        return list(self._name_map.values())

    def items(self) -> list[tuple[str, Medium]]:
        return list(self._name_map.items())

    def to_dict(self) -> dict[str, Medium]:
        return dict(self._name_map)

    def get_by_id(self, material_id: int) -> Medium | None:
        return next(
            (m for m in self._media if m.material_id == material_id), None
        )

    def to_prj(self, prj: Project) -> None:

        prj.remove_element("./media")
        prj.add_block("media", parent_xpath=".")

        prj.media = OGS6pyMedia(prj.tree)

        for medium in self._media:
            medium.to_prj(prj)

    @classmethod
    def from_prj(cls, prj: Project, process: str) -> "Media":
        # TODO: parse all <media><medium> entries and reconstruct Medium/Phase/Component/Property
        _ = prj, process  # prevent unused arg warnings
        msg = "from_prj() not implemented yet."
        raise NotImplementedError(msg)

    # def validate(self) -> bool:
    #     for medium in self._media:
    #         for phase in medium.get_phases():
    #             # Regeln aus dem Prozess-Schema extrahieren
    #             expected_phase_schemas = [
    #                 p for p in PROCESS_SCHEMAS[self.process]["phases"]
    #                 if p["type"] == phase.type
    #             ]
    #             if not expected_phase_schemas:
    #                 raise ValueError(f"Phase '{phase.type}' not allowed in process '{self.process}'")

    #             expected = expected_phase_schemas[0]
    #             # Komponenten validieren
    #             for component in phase.components:
    #                 allowed_props = expected.get("components", {}).get(component.role, [])
    #                 component.validate(allowed_props)
    #     return True

    # def validate(self) -> bool:
    #     for medium in self._media:
    #         medium.validate()
    #     return True

    def validate(self) -> bool:
        ambiguous = self.find_ambiguous_properties()
        if ambiguous:
            lines = ["Found ambiguous properties:"]
            for medium_name, props in ambiguous.items():
                lines.append(f"  - Medium: {medium_name}")
                for prop_name, variants in props.items():
                    lines.append(f"    • Property: {prop_name}")
                    for i, prop in enumerate(variants, 1):
                        desc = f"type: {prop.type}"
                        if prop.extra:
                            desc += " | " + ", ".join(
                                f"{k}={v}" for k, v in prop.extra.items()
                            )
                        lines.append(f"      [{i}] {desc}")
            raise ValueError("\n".join(lines))

        for medium in self._media:
            medium.validate()
        return True

    def validate_medium(self, medium: Medium) -> bool:
        return medium.validate()

    def find_ambiguous_properties(self) -> dict[str, dict[str, list[str]]]:
        """Finds properties that are defined multiple times for the same medium."""

        issues = defaultdict(lambda: defaultdict(list))

        for medium in self._media:
            props_by_name = defaultdict(list)
            for prop in medium.properties:
                props_by_name[prop.name].append(prop)
            for name, variants in props_by_name.items():
                if len(variants) > 1:
                    issues[medium.name][name] = variants

        return dict(issues)

    def __repr__(self) -> str:
        lines = [f"<Media with {len(self)} entries>"]
        for medium in self._media:
            lines.append(f"  ├─ [{medium.material_id}] {medium.name}")
        return "\n".join(lines)
