from .component import Component
from .material import Material, RawMaterial


class Phase:
    def __init__(
        self,
        type_: str,
        material: Material = None,
        properties=None,
        components=None,
    ):
        self.type = type_
        self.properties = properties or []
        self.components = components or []

        if material:
            if isinstance(material, RawMaterial):
                msg = "Cannot build Phase from RawMaterial. Use a filtered Material from MaterialList."
                raise TypeError(msg)
            if not isinstance(material, Material):
                msg = f"Expected a Material instance, got {type(material)}"
                raise TypeError(msg)

            self._load_from_material(material)

    def _load_from_material(self, material: Material):
        """Assign properties from a validated, filtered Material object."""
        self.properties.extend(material.get_properties())

    def add_property(self, prop):
        self.properties.append(prop)

    def add_component(self, component: Component):
        self.components.append(component)

    def __repr__(self):
        return f"<Phase '{self.type}' with {len(self.properties)} properties and {len(self.components)} components>"


# class Phase:
#     def __init__(self, type_: str, material: Material = None, properties=None, components=None):
#         self.type = type_
#         self.properties = properties or []
#         self.components = components or []

#         if material:
#             self._load_from_material(material)

#     def _load_from_material(self, material: Material):
#         """Filter material properties relevant to this phase (basic version)."""
#         for prop in material.get_properties():
#             self.properties.append(prop)

#     def add_property(self, prop):
#         self.properties.append(prop)

#     def add_component(self, component: Component):
#         self.components.append(component)
