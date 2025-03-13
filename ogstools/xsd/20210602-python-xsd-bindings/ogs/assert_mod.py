from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ConstantParameter:
    class Meta:
        name = "constantParameter"

    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        }
    )
    value: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        }
    )
    type: str = field(
        init=False,
        default="Constant",
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class CurveParameter:
    class Meta:
        name = "curveParameter"

    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        }
    )
    curve: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        }
    )
    parameter: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        }
    )
    type: str = field(
        init=False,
        default="CurveScaled",
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class OpenGeoSysProjectType:
    parameters: Optional["OpenGeoSysProjectType.Parameters"] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        }
    )
    any_element: List[object] = field(
        default_factory=list,
        metadata={
            "type": "Wildcard",
            "namespace": "##any",
        }
    )

    @dataclass
    class Parameters:
        parameter: List[Union[ConstantParameter, CurveParameter]] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 1,
            }
        )


@dataclass
class OpenGeoSysProject(OpenGeoSysProjectType):
    pass
