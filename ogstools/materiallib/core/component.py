from typing import Any


class Component:
    def __init__(self, name: str, properties: list | None = None):
        self.name = name
        self.properties = properties or []

    def add_property(self, prop: Any) -> None:
        self.properties.append(prop)
