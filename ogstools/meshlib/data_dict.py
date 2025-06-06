# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from collections.abc import Callable, Iterator, MutableMapping, Sequence
from typing import Any

import numpy as np

from ogstools.variables import Variable

from .mesh import Mesh


class DataDict(MutableMapping):
    "Mapping that works like both a dict and a mutable object."

    def __init__(
        self,
        ms: Sequence[Mesh],
        get_data: Callable[[Mesh], dict],
        array_len: int | None,
    ) -> None:
        self.ms = ms
        self.get_data = get_data
        self.array_len = array_len

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, float | int):
            if key in (mesh0_data := self.get_data(self.ms[0])):
                len_vals = np.shape(mesh0_data[key])[0]
            else:
                len_vals = -1
            value = np.tile(value, (len(self.ms), len_vals))
        if self.array_len is not None:
            value = np.reshape(value, (len(self.ms), self.array_len, -1))
        for mesh, value_t in zip(self.ms, value, strict=True):
            self.get_data(mesh)[key] = value_t

    def __getitem__(self, key: str | Variable) -> np.ndarray:
        if isinstance(key, Variable):
            if key.output_name in self:
                return np.asarray(
                    [self.get_data(mesh)[key.output_name] for mesh in self.ms]
                )
            return key.transform(self.ms)
        return np.asarray([self.get_data(mesh)[key] for mesh in self.ms])

    def __delitem__(self, key: str | Variable) -> None:
        if isinstance(key, Variable):
            key = key.output_name if key.output_name in self else key.data_name
        for mesh in self.ms:
            del self.get_data(mesh)[key]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.get_data(self.ms[0]))

    def __len__(self) -> int:
        return len(self.get_data(self.ms[0]))

    def __contains__(self, key: object) -> bool:
        data = self.get_data(self.ms[0])
        if isinstance(key, Variable):
            return key.output_name in data or key.data_name in data
        return key in self.get_data(self.ms[0])
