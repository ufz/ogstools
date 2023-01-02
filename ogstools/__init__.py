from importlib import metadata

__version__ = metadata.version(__package__)
__authors__ = metadata.metadata(__package__)["Author"]

del metadata  # optional, avoids polluting the results of dir(__package__)
