# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from .boundary import Layer
from .boundary_set import LayerSet


def refine(layerset: LayerSet, factor: int) -> LayerSet:
    """
    Refine the provided LayerSet by increasing the number of subdivisions.

    This function takes a LayerSet and refines it by increasing the number of subdivisions
    in each layer. The factor parameter determines the degree of refinement.

    :param layerset: The original LayerSet to be refined.
    :param factor:   The refinement factor for the number of subdivisions.

    :returns: A new LayerSet with increased subdivisions for each layer.
    """

    def refined_num_subsections(num_subsections: int, factor: int) -> int:
        return (num_subsections + 1) * factor - 1

    out = [
        Layer(
            layer.top,
            layer.bottom,
            material_id=layer.material_id,
            num_subdivisions=refined_num_subsections(
                layer.num_subdivisions, factor
            ),
        )
        for layer in layerset.layers
    ]

    return LayerSet(layers=out)
