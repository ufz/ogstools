from __future__ import annotations

import pytest

from ogstools.materiallib.distributions import (
    Distribution,
    LogNormalDistribution,
    LogUniformDistribution,
    NormalDistribution,
    UniformDistribution,
    parse_distribution,
    serialize_distribution,
)

_DISTRIBUTION_CASES = [
    (
        {"type": "uniform", "lower": 0.1, "upper": 0.2},
        UniformDistribution(lower=0.1, upper=0.2),
    ),
    (
        {"type": "normal", "mean": 5.0, "stddev": 1.5},
        NormalDistribution(mean=5.0, stddev=1.5),
    ),
    (
        {"type": "loguniform", "lower": 1.0e-12, "upper": 1.0e-10},
        LogUniformDistribution(lower=1.0e-12, upper=1.0e-10),
    ),
    (
        {"type": "lognormal", "mean": 4.8e7, "stddev": 5.0e6},
        LogNormalDistribution(mean=4.8e7, stddev=5.0e6),
    ),
]


@pytest.mark.parametrize(
    ("data", "expected"),
    _DISTRIBUTION_CASES,
)
def test_parse_distribution(
    data: dict[str, float | str], expected: Distribution
) -> None:
    distribution = parse_distribution(data)

    assert distribution == expected


@pytest.mark.parametrize(
    ("expected", "distribution"),
    [(data, distribution) for data, distribution in _DISTRIBUTION_CASES],
)
def test_serialize_distribution(
    expected: dict[str, float | str], distribution: Distribution
) -> None:
    data = serialize_distribution(distribution)

    assert data == expected


@pytest.mark.parametrize(
    "distribution",
    [(distribution,) for _, distribution in _DISTRIBUTION_CASES],
)
def test_distribution_roundtrip(distribution: Distribution) -> None:
    roundtrip = parse_distribution(serialize_distribution(distribution))

    assert roundtrip == distribution


def test_parse_distribution_rejects_missing_type() -> None:
    with pytest.raises(
        ValueError, match="Distribution mapping must define a string 'type'"
    ):
        parse_distribution({"lower": 0.1, "upper": 0.2})


def test_parse_distribution_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown distribution type"):
        parse_distribution({"type": "triangle", "lower": 0.1, "upper": 0.2})


def test_parse_distribution_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="contains unknown key"):
        parse_distribution(
            {"type": "uniform", "lower": 0.1, "upper": 0.2, "mean": 0.15}
        )


def test_parse_distribution_rejects_missing_required_keys() -> None:
    with pytest.raises(ValueError, match="missing required key"):
        parse_distribution({"type": "normal", "mean": 5.0})
