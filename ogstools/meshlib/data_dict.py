from collections.abc import Callable, Iterator, MutableMapping, Sequence
from typing import Any

import numpy as np

from .mesh import Mesh


class DataDict(MutableMapping):
    "Mapping that works like both a dict and a mutable object."

    def __init__(
        self, ms: Sequence[Mesh], get_data: Callable[[Mesh], dict]
    ) -> None:
        self.ms = ms
        self.get_data = get_data

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, float | int):
            if key in (mesh0_data := self.get_data(self.ms[0])):
                len_vals = np.shape(mesh0_data[key])[0]
            else:
                len_vals = -1
            value = np.tile(value, (len(self.ms), len_vals))
        for mesh, value_t in zip(self.ms, value, strict=True):
            self.get_data(mesh)[key] = value_t

    def __getitem__(self, key: str) -> np.ndarray:
        return np.asarray([self.get_data(mesh)[key] for mesh in self.ms])

    def __delitem__(self, key: str) -> None:
        for mesh in self.ms:
            del self.get_data(mesh)[key]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.get_data(self.ms[0]))

    def __len__(self) -> int:
        return len(self.get_data(self.ms[0]))
