# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class UniformDistribution:
    lower: float
    upper: float


@dataclass(frozen=True)
class NormalDistribution:
    mean: float
    stddev: float


@dataclass(frozen=True)
class LogUniformDistribution:
    lower: float
    upper: float


@dataclass(frozen=True)
class LogNormalDistribution:
    mean: float
    stddev: float


Distribution = (
    UniformDistribution
    | NormalDistribution
    | LogUniformDistribution
    | LogNormalDistribution
)


_DISTRIBUTION_TYPES: dict[str, type[Distribution]] = {
    "uniform": UniformDistribution,
    "normal": NormalDistribution,
    "loguniform": LogUniformDistribution,
    "lognormal": LogNormalDistribution,
}


def _distribution_fields(distribution_type: str) -> tuple[str, ...]:
    cls = _DISTRIBUTION_TYPES[distribution_type]
    return tuple(field.name for field in fields(cls))


def _validate_distribution_mapping(data: dict[str, Any]) -> str:
    type_ = data.get("type")
    if not isinstance(type_, str):
        msg = "Distribution mapping must define a string 'type'."
        raise ValueError(msg)

    if type_ not in _DISTRIBUTION_TYPES:
        known_types = ", ".join(sorted(_DISTRIBUTION_TYPES))
        msg = (
            f"Unknown distribution type '{type_}'. "
            f"Known types are: {known_types}."
        )
        raise ValueError(msg)

    expected_fields = _distribution_fields(type_)
    allowed_keys = {"type", *expected_fields}
    unknown_keys = set(data) - allowed_keys
    if unknown_keys:
        msg = (
            f"Distribution of type '{type_}' contains unknown key(s): "
            f"{', '.join(sorted(unknown_keys))}."
        )
        raise ValueError(msg)

    missing_keys = set(expected_fields) - set(data)
    if missing_keys:
        msg = (
            f"Distribution of type '{type_}' is missing required key(s): "
            f"{', '.join(sorted(missing_keys))}."
        )
        raise ValueError(msg)

    return type_


def parse_distribution(data: dict[str, Any]) -> Distribution:
    type_ = _validate_distribution_mapping(data)
    cls = _DISTRIBUTION_TYPES[type_]
    kwargs = {name: data[name] for name in _distribution_fields(type_)}
    return cls(**kwargs)


def serialize_distribution(distribution: Distribution) -> dict[str, Any]:
    for type_name, cls in _DISTRIBUTION_TYPES.items():
        if isinstance(distribution, cls):
            return {"type": type_name, **asdict(distribution)}

    msg = f"Unsupported distribution object of type {type(distribution)!r}."
    raise TypeError(msg)
