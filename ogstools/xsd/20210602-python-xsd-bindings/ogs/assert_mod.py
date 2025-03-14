from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConstantParameter:
    class Meta:
        name = "constantParameter"

    name: str | None = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    value: str | None = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    type: str = field(
        init=False,
        default="Constant",
        metadata={
            "type": "Attribute",
        },
    )


@dataclass
class CurveParameter:
    class Meta:
        name = "curveParameter"

    name: str | None = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    curve: str | None = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    parameter: str | None = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    type: str = field(
        init=False,
        default="CurveScaled",
        metadata={
            "type": "Attribute",
        },
    )


@dataclass
class OpenGeoSysProjectType:
    parameters: Optional["OpenGeoSysProjectType.Parameters"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    any_element: list[object] = field(
        default_factory=list,
        metadata={
            "type": "Wildcard",
            "namespace": "##any",
        },
    )

    @dataclass
    class Parameters:
        parameter: list[ConstantParameter | CurveParameter] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 1,
            },
        )


@dataclass
class OpenGeoSysProject(OpenGeoSysProjectType):
    pass
