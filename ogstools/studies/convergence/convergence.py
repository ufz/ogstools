from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

from ogstools.propertylib import Property


def resample(
    topology: pv.DataSet, meshes: list[pv.DataSet]
) -> list[pv.DataSet]:
    meshes_resampled = []
    for mesh in meshes:
        mesh_temp = deepcopy(topology)
        mesh_temp.clear_point_data()
        mesh_temp = mesh_temp.sample(mesh, pass_cell_data=False)
        meshes_resampled += [mesh_temp]

    return meshes_resampled


def add_grid_spacing(mesh: pv.DataSet) -> pv.DataSet:
    dim = mesh.get_cell(0).dimension
    key = ["Length", "Area", "Volume"][dim - 1]
    _mesh = mesh.compute_cell_sizes()
    _mesh.cell_data["grid_spacing"] = _mesh.cell_data[key] ** (1.0 / dim)
    return _mesh


def grid_convergence(
    meshes: list[pv.DataSet],
    property: Property,
    topology: pv.DataSet,
    refinement_ratio: float,
) -> pv.DataSet:
    """
    Calculate the grid convergence field for the given meshes on the topology.

    The calculation is based on the last three of the given meshes.
    For more information on this topic see
    <https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html> or
    <https://curiosityfluids.com/2016/09/09/establishing-grid-convergence/>.

    :param meshes:              At least three meshes with constant refinement.
    :param property:            The property to be extrapolated.
    :param topology:            The topology to evaluate.
    :param refinement_ratio:    If not given, it is calculated automatically

    returns:                    Grid convergence field of the given property.
    """
    assert len(meshes) >= 3
    cast = property.magnitude.strip_units
    result = deepcopy(topology)
    result.clear_point_data()
    result.clear_cell_data()
    _meshes = resample(topology=topology, meshes=meshes)
    f3 = cast(_meshes[-3].point_data[property.data_name])
    f2 = cast(_meshes[-2].point_data[property.data_name])
    f1 = cast(_meshes[-1].point_data[property.data_name])
    r = np.ones(f1.shape) * refinement_ratio
    a = f3 - f2
    b = f2 - f1
    zeros = np.zeros_like
    ones = np.ones_like
    c = np.divide(a, b, out=ones(a), where=(b != 0))
    with np.errstate(divide="ignore"):
        p = np.log(np.abs(c)) / np.log(r)
    rpm1 = r**p - 1
    _gci23 = np.divide(np.abs(a), f3, out=zeros(a), where=(f3 != 0.0))
    _gci12 = np.divide(np.abs(b), f2, out=zeros(a), where=(f2 != 0.0))
    gci23 = np.divide(_gci23, rpm1, out=zeros(a), where=(rpm1 != 0.0))
    gci12 = np.divide(_gci12, rpm1, out=zeros(a), where=(rpm1 != 0.0))
    conv_ratio = np.divide(
        gci23, gci12 * r**p, out=ones(a), where=((gci12 * r**p) != 0.0)
    )

    result["r"] = r
    result["p"] = p
    result["gci23"] = gci23
    result["gci12"] = gci12
    result["grid_convergence"] = conv_ratio

    return result


def richardson_extrapolation(
    meshes: list[pv.DataSet],
    property: Property,
    topology: pv.DataSet,
    refinement_ratio: float,
) -> pv.DataSet:
    """
    Estimate a better approximation of a property on a mesh.

    This function calculates the Richardson Extrapolation based on the change
    in results in the last three of the given meshes.
    For more information on this topic see
    <https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html> or
    <https://curiosityfluids.com/2016/09/09/establishing-grid-convergence/>.

    :param meshes:              At least three meshes with constant refinement.
    :param property:            The property to be extrapolated.
    :param topology:            The topology on which the extrapolation is done.
    :param refinement_ratio:    Refinement ratio (spatial or temporal).

    :returns:                   Richardson extrapolation of the given property.
    """
    _meshes = resample(topology, meshes[-2:])
    m1 = _meshes[-1]
    m2 = _meshes[-2]
    f1 = m1.point_data[property.data_name]
    f2 = m2.point_data[property.data_name]
    results = grid_convergence(meshes, property, topology, refinement_ratio)
    r = results["r"].astype(np.float64)
    p = results["p"]
    rpm1 = r**p - 1
    diff = f1 - f2
    delta = np.divide(
        diff.T, rpm1, out=np.zeros_like(f1.T), where=(rpm1 != 0)
    ).T
    results.point_data[property.data_name] = f1 + delta
    return results


def convergence_metrics(
    meshes: list[pv.DataSet],
    reference: pv.DataSet,
    property: Property,
    timestep_sizes: list[float],
) -> pd.DataFrame:
    """
    Calculate convergence metrics for a given reference and property.

    :param meshes:      The List of meshes which is analyzed for convergence.
    :param reference:   The reference Dataset to compare against.
    :param property:    The property of interest.

    :returns:           A pandas Dataframe containing all metrics.
    """

    def _data(m: pv.DataSet):
        return property.magnitude.strip_units(m.point_data[property.data_name])

    x = [np.mean(add_grid_spacing(mesh)["grid_spacing"]) for mesh in meshes]
    x_str = "mean element length"
    if all(xi == x[0] for xi in x):
        x = timestep_sizes
        x_str = "time step size"
    x += [0.0]
    _meshes = meshes + [reference]
    maxs = [np.max(_data(m)) for m in _meshes]
    mins = [np.min(_data(m)) for m in _meshes]
    rel_errs_max = np.abs(1.0 - maxs / maxs[-1])
    rel_errs_min = np.abs(1.0 - mins / mins[-1])
    rel_errs_l2 = []
    for mesh in resample(reference, _meshes):
        rel_errs_l2 += [
            np.linalg.norm(_data(reference) - _data(mesh), axis=0, ord=2)
            / np.linalg.norm(_data(reference), axis=0, ord=2)
        ]
    data = np.column_stack(
        (x, maxs, mins, rel_errs_max, rel_errs_min, rel_errs_l2)
    )
    columns = [x_str, "maximum", "minimum"] + [
        f"rel. error ({x})" for x in ["max", "min", "L2 norm"]
    ]

    return pd.DataFrame(data, columns=columns)


def log_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    if np.all(np.isnan(y)):
        return 0.0, y
    indices = np.invert(np.isnan(x))
    _x = x[indices]
    _y = y[indices]
    params = np.polyfit(np.log10(_x), np.log10(_y), 1)
    fit_vals = 10 ** (params[0] * np.log10(x) + params[1])
    return params[0], fit_vals


def plot_convergence(metrics: pd.DataFrame, property: Property) -> plt.Figure:
    "Plot the absolute values of the convergence metrics."
    fig, axes = plt.subplots(2, 1, sharex=True)
    metrics.iloc[:-1].plot(ax=axes[0], x=0, y=1, c="r", style="-o", grid=True)
    axes[0].plot(metrics.iloc[-1, 0], metrics.iloc[-1, 1], "r^")
    axes[0].legend(["maximum", "Richardson\nextrapolation"])
    metrics.iloc[:-1].plot(ax=axes[1], x=0, y=2, c="b", style="-o", grid=True)
    axes[1].plot(metrics.iloc[-1, 0], metrics.iloc[-1, 2], "b^")
    axes[1].legend(["minimum", "Richardson\nextrapolation"])
    y_label = property.output_name + " / " + property.output_unit
    fig.supylabel(y_label, fontsize="medium")
    fig.tight_layout()
    return fig


def plot_convergence_errors(metrics: pd.DataFrame) -> plt.Figure:
    "Plot the relative errors of the convergence metrics in loglog scale."
    plot_df = metrics.replace(0.0, np.nan)
    plot_df[plot_df < 1e-12] = np.nan
    x_vals = plot_df.iloc[:, 0].to_numpy()
    fig, ax = plt.subplots()
    for i, c in enumerate("rbg"):
        j = i + 3
        order_p, fit_vals = log_fit(x_vals, plot_df.iloc[:, j].to_numpy())
        err_str = ["max", "min", "L2"][i]
        label = f"$\\varepsilon_{{rel}}^{{{err_str}}} (p={order_p:.2f})$"
        plot_df.plot(
            ax=ax, x=0, y=j, c=c, style="o", grid=True, loglog=True, label=label
        )
        ax.loglog(x_vals, fit_vals, c + "--")
    fig.tight_layout()
    return fig
