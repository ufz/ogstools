from .boundary import Layer
from .boundary_set import LayerSet


def refine(layerset: LayerSet, factor: int) -> LayerSet:
    """
    Refine the provided LayerSet by increasing the number of subdivisions.

    This function takes a LayerSet and refines it by increasing the number of subdivisions
    in each layer. The factor parameter determines the degree of refinement.

    Args:
        layerset (LayerSet): The original LayerSet to be refined.
        factor (int): The refinement factor to increase the number of subdivisions.

    Returns:
        LayerSet: A new LayerSet with increased subdivisions for each layer.
    """

    def refined_num_subsections(num_subsections, factor):
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
