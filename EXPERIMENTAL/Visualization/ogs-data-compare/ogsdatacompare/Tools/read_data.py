import os
import vtuIO
from .detect_experiments import getfilesbyextension


# TODO: add option to use different tools for
def read_data(experiments_to_include, field_to_read, points, results_folder):
    SEC2A = 1 / (365 * 24 * 60 * 60)  # time in seconds to time in years
    reference_experiment = experiments_to_include[0]
    # Load different experiments
    experiments_results = {}
    experiments_results_param = {}
    for experiment_name in experiments_to_include:
        # data_path = os.path.join(os.getcwd(), experiment_name, 'Results', file_name)  # Select by experiment folder
        file_name = getfilesbyextension(os.path.join(os.getcwd(), results_folder, experiment_name))
        data_path = os.path.join(results_folder, experiment_name, file_name[0])  # Select by results folder
        data_file = vtuIO.PVDIO(data_path, dim=2)
        experiments_results[experiment_name] = data_file
        experiments_results_param[experiment_name] \
            = data_file.read_time_series(field_to_read,
                                         pts=points)

    point_names = list(experiments_results_param[reference_experiment].keys())
    time_steps = experiments_results[reference_experiment].timesteps * SEC2A

    return experiments_results, experiments_results_param, point_names, time_steps


def read_data_full(path, parameter_to_read, experiments_results):
    data_full = {}
    for experiment in experiments_results.keys():
        data_full[experiment] = {}
        for vtu_file in experiments_results[experiment].vtufilenames:
            vtu_data = vtuIO.VTUIO(os.path.join(path, experiment, vtu_file), dim=2)
            data_full[experiment][vtu_file] = vtu_data.get_point_field(parameter_to_read)

    return data_full
