from typing import Any

from ogstools.ogs6py import Project


class Property:
    def __init__(self, name: str, type_: str, value: Any = None, **extra: Any):
        # def __init__(self, name: str, type_: str, value: float | None = None, **extra):
        self.name = name
        self.type = type_
        self.value = value
        self.extra = extra  # e.g. unit, slope, source, ...

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.type}
        if self.value is not None:
            d["value"] = self.value
        d.update(self.extra)
        return d

    def to_prj(
        self,
        prj: Project,
        medium_id: str,
        phase_type: str | None = None,
        component_name: str | None = None,
    ) -> None:
        args = {
            "medium_id": medium_id,
            "name": self.name,
            "type": self.type,
            "value": self.value,
            **self.extra,
        }
        if phase_type is not None:
            args["phase_type"] = phase_type
        if component_name is not None:
            args["component_name"] = component_name

        prj.media.add_property(**args)

    def __repr__(self) -> str:
        return f"<Property '{self.name}' type={self.type} value={self.value}>"
