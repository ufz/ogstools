# ogstools/materiallib/core/media.py

from collections.abc import Iterator

from ogstools.materiallib.core.material_list import MaterialList
from ogstools.materiallib.core.medium import Medium


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

            self._media.append(medium)
            self._name_map[name] = medium

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

    def __repr__(self) -> str:
        lines = [f"<Media with {len(self)} entries>"]
        for medium in self._media:
            lines.append(f"  ├─ [{medium.material_id}] {medium.name}")
        return "\n".join(lines)
