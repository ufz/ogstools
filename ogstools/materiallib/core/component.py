class Component:
    def __init__(self, name: str, properties: list = None):
        self.name = name
        self.properties = properties or []

    def add_property(self, prop):
        self.properties.append(prop)
