import os
import types

import matplotlib.pyplot as plt
import itertools
import numpy as np

from .Tools.points import load_points, plot_points
from .Tools.read_data import read_data, read_data_full
from .Tools.detect_experiments import detect_experiments, extract_params
from .Tools.legend_without_duplicate_labels import legend_without_duplicate_labels
from .Tools.match_time_step import match_time_step
from .Tools.get_analytical_model import get_analytica_model


class OGSDataCompare:

    def __init__(self, settings):

        self.constants = types.SimpleNamespace()
        self.constants.SEC2A = 1 / (365 * 24 * 60 * 60)
        self.constants.PA2MPA = 1e-6

        self.config = types.SimpleNamespace()
        # Mandatory
        self.config.path = os.path.join('')
        self.config.parameter = None
        self.config.points_file = None
        # Optional
        self.config.points_plot = False
        self.config.plot_legend = False
        self.config.color_ref = None
        self.config.colors = None
        self.config.plot_ext = None
        self.config.include_legend = False
        self.config.reference_experiment = None
        self.__read_config(settings)
        self.config.output_path = os.path.join(self.config.path, 'Results')

        self.points = self.read_points()

        self.data = types.SimpleNamespace()
        self.data.experiments = None
        self.data.experiments_results = None
        self.data.experiments_results_param = None
        self.data.point_names = None
        self.data.time_steps = None

    def read_points(self):
        points = load_points(self.config.points_file)

        if self.config.points_plot is True:
            plot_points(points, self.config.path)

        return points

    def read_data(self):
        print('Reading data... ')
        self.data.experiments = detect_experiments(self.config.path)

        if self.config.reference_experiment is None:
            self.config.reference_experiment = self.data.experiments[0]

        self.data.experiments_results, self.data.experiments_results_param, self.data.point_names, self.data.time_steps = \
            read_data(self.data.experiments,
                      self.config.parameter,
                      self.points,
                      self.config.path)
        print('Finished reading data.')

    def read_data_full(self):
        print('Reading data... ')
        self.data.experiments_results_full = read_data_full(self.config.path,
                                                            self.config.parameter,
                                                            self.data.experiments_results)

    def ogs_compare(self, analytical_ref=False, ogs_model=None):

        if self.config.reference_experiment is not None:
            experiment_name = self.config.reference_experiment
        else:
            experiment_name = self.data.experiments[0]

        # Plot everything
        sub_params_n = self.data.experiments_results_param[experiment_name][self.data.point_names[0]].shape
        if len(sub_params_n) > 1:
            sub_params_n = sub_params_n[1]
        else:
            sub_params_n = 1
        for sub_param in range(sub_params_n):

            plt.rcParams['figure.figsize'] = (15, 15)
            fig, ax = plt.subplots(4, 4)  # (2, 3) # (4, 4)

            # Plot reference line first
            experiments_to_include_new = None
            k = 0
            for i, j in itertools.product(range(ax.shape[0]), range(ax.shape[1])):
                point_name = self.data.point_names[k]
                # THIS FUNCTIONALITY IS NOT AVAILABLE NOW. SET analytical_ref TO FALSE.
                # See Tools\get_analytical_model.py for explanation.
                if analytical_ref is True:
                    exp_data_plot = get_analytica_model(self.config.parameter, self.points[point_name][0],
                                                        self.points[point_name][1], self.points[point_name][2],
                                                        self.data.time_steps / self.constants.SEC2A,
                                                        ogs_model=ogs_model)
                    experiments_to_include_new = self.data.experiments.copy()
                    label_temp = 'analytical_model'
                else:
                    if sub_params_n > 1:
                        exp_data_plot = self.data.experiments_results_param[self.config.reference_experiment][
                                            point_name][:, sub_param]
                    else:
                        exp_data_plot = self.data.experiments_results_param[self.config.reference_experiment][
                            point_name]
                    experiments_to_include_new = self.data.experiments.copy()
                    experiments_to_include_new.remove(self.config.reference_experiment)
                    label_temp = self.data.experiments[0]
                ax[i][j].plot(self.data.time_steps, exp_data_plot, color=self.config.color_ref,
                              label=label_temp)
                k += 1

            # Plot curves from other experiments
            k = 0

            for i, j in itertools.product(range(ax.shape[0]), range(ax.shape[1])):
                point_name = self.data.point_names[k]
                color_iterator = 0
                for experiment_name in experiments_to_include_new:
                    print(experiment_name)

                    if sub_params_n > 1:
                        exp_data_plot = self.data.experiments_results_param[experiment_name][point_name][:, sub_param]
                    else:
                        exp_data_plot = self.data.experiments_results_param[experiment_name][point_name]
                    ax[i][j].plot(self.data.time_steps, exp_data_plot, color=self.config.colors[color_iterator],
                                  label=experiments_to_include_new[color_iterator])
                    color_iterator += 1
                ax[i][j].set_title(point_name)
                k += 1

            # Other plot management steps
            # lines, labels = ax[i][j].get_legend_handles_labels()
            if self.config.include_legend is True:
                # fig.legend(lines, labels, loc='lower center', ncol=4)
                fig = legend_without_duplicate_labels(fig)
            plt.setp(ax[-1, :], xlabel='Time [a]')
            plt.setp(ax[:, 0], ylabel=self.config.parameter)  # TODO add units
            if sub_params_n > 1:
                fig.suptitle(self.config.parameter + '_' + str(sub_param))
            else:
                fig.suptitle(self.config.parameter)

            output_file_name = "{}_compare_{}".format(self.config.parameter, str(sub_param))
            self.__save_figure(fig, output_file_name)

    def ogs_compare_separate(self):

        sub_params_n = self.data.experiments_results_param[self.config.reference_experiment][
            self.data.point_names[0]].shape
        if len(sub_params_n) > 1:
            sub_params_n = sub_params_n[1]
        else:
            sub_params_n = 1

        for sub_param in range(sub_params_n):
            for point_name in self.data.point_names:
                fig_sep, ax_sep = plt.subplots(1)

                # Plot reference experiment
                if sub_params_n > 1:
                    exp_data_plot = self.data.experiments_results_param[self.config.reference_experiment][point_name][:,
                                    sub_param]
                else:
                    exp_data_plot = self.data.experiments_results_param[self.config.reference_experiment][point_name]
                ax_sep.plot(self.data.time_steps, exp_data_plot,
                            color=self.config.colors[0],
                            label=self.data.experiments[0])

                color_iterator = 0
                for experiment_name in self.data.experiments:
                    if sub_params_n > 1:
                        exp_data_plot = self.data.experiments_results_param[experiment_name][point_name][:, sub_param]
                    else:
                        exp_data_plot = self.data.experiments_results_param[experiment_name][point_name]
                    ax_sep.plot(self.data.time_steps, exp_data_plot,
                                color=self.config.colors[color_iterator + 1],
                                label=self.data.experiments[color_iterator])

                    color_iterator += 1
                    r = np.sqrt(
                        self.points[point_name][0] ** 2 + self.points[point_name][1] ** 2 + self.points[point_name][
                            2] ** 2)
                    ax_sep.set_title("{} at {}, r={:.2f}m".format(self.config.parameter, point_name, r))

                lines, labels = ax_sep.get_legend_handles_labels()
                fig_sep.legend(lines, labels, loc='center right', ncol=1, bbox_to_anchor=(1.17, 0.5))
                ax_sep.set_xlabel('t / a')
                ax_sep.set_ylabel(self.config.parameter)  # TODO add units

                fig_sep.set_size_inches(8, 8)
                fig_sep.set_dpi(200)
                fig_sep.tight_layout()

                output_file_name = "{}_compare_separate_{}_{}".format(self.config.parameter,
                                                                      str(sub_param),
                                                                      point_name)
                self.__save_figure(fig_sep, output_file_name)

        if self.config.plot_legend is True:
            # Plot and save legend as separate file

            fig_legend = plt.figure(figsize=(5, 2.1))
            fig_legend.legend(lines, labels)
            fig_legend.tight_layout()
            fig_legend.show()

            output_file_name = "compare_separate_legend"
            self.__save_figure(fig_legend, output_file_name)

    def ogs_compare_time_point(self, time_step, analytical_ref=False, ogs_model=None):

        # Prepare results
        sub_params_n = self.data.experiments_results_param[self.config.reference_experiment][
            self.data.point_names[0]].shape
        if len(sub_params_n) > 1:
            sub_params_n = sub_params_n[1]
        else:
            sub_params_n = 1

        for point_name in self.data.point_names:
            for time_step_temp in time_step:
                time_step_id = match_time_step(self.data.time_steps, time_step_temp)

                for sub_param in range(sub_params_n):
                    k = 0
                    x_positions = []
                    y_values = []
                    x_axis_labels = []
                    for experiment_name in self.data.experiments_results_param.keys():
                        x_positions.append(k)
                        if sub_params_n > 1:
                            y_values.append(
                                self.data.experiments_results_param[experiment_name][point_name][time_step_id][
                                    sub_param])
                            x_axis_labels.append(experiment_name)
                        else:
                            y_values.append(
                                self.data.experiments_results_param[experiment_name][point_name][time_step_id])
                            x_axis_labels.append(experiment_name)
                        k += 1

                    # Plot the results
                    # plt.rcParams['figure.figsize'] = (15, 15)
                    plt.rcParams.update({'figure.autolayout': True})
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(x_positions, y_values, 'rx', label=experiment_name)

                    plt.xticks(np.arange(0, k, 1), x_axis_labels, rotation=90)
                    ax.set_ylabel(self.config.parameter)
                    ax.set_xlabel('Experiment name')

                    output_file_name = "{}_compare_time_point_{}_t_{:.2f}_{}".format(self.config.parameter,
                                                                                     str(sub_param),
                                                                                     self.data.time_steps[time_step_id],
                                                                                     point_name)

                    self.__save_figure(fig, output_file_name)

    def ogs_compare_3d_view(self, time_step=None):

        # Prepare results
        sub_params_n = self.data.experiments_results_param[self.config.reference_experiment][
            self.data.point_names[0]].shape
        if len(sub_params_n) > 1:
            sub_params_n = sub_params_n[1]
        else:
            sub_params_n = 1

        for point_name in self.data.point_names:
            for time_step_temp in time_step:
                time_step_id = match_time_step(self.data.time_steps, time_step_temp)

                for sub_param in range(sub_params_n):
                    a_param_values = []
                    b_param_values = []
                    z_values = []

                    for experiment_name in self.data.experiments_results_param.keys():
                        a_param, b_param = extract_params(experiment_name)
                        a_param_name = a_param['name']
                        a_param_values.append(a_param['value'])
                        b_param_name = b_param['name']
                        b_param_values.append(b_param['value'])
                        if sub_params_n > 1:
                            z_values.append(
                                self.data.experiments_results_param[experiment_name][point_name][time_step_id][
                                    sub_param])

                        else:
                            z_values.append(
                                self.data.experiments_results_param[experiment_name][point_name][time_step_id])

                    # Plot the results
                    # plt.rcParams['figure.figsize'] = (15, 15)
                    #plt.rcParams.update({'figure.autolayout': True})
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(a_param_values, b_param_values, z_values, 'rx', label=experiment_name)

                    ax.set_xlabel(a_param_name)
                    ax.set_ylabel(b_param_name)
                    ax.set_zlabel(self.config.parameter)

                    ax.view_init(azim=20, elev=45)

                    #fig.set_size_inches(6, 6)
                    #fig.set_dpi(150)

                    output_file_name = "{}_3D_values_{}_t_{:.2f}_{}".format(self.config.parameter,
                                                                            str(sub_param),
                                                                            self.data.time_steps[time_step_id],
                                                                            point_name)

                    self.__save_figure(fig, output_file_name)

    def plot_min_max(self):
        # Get the data
        for experiment in self.data.experiments_results:
            data_max = np.ones([len(self.data.experiments_results_full[experiment].keys())]) * np.NINF
            data_min = np.ones([len(self.data.experiments_results_full[experiment].keys())]) * np.PINF
            for data_ts, time_step_id in zip(self.data.experiments_results_full[experiment].keys(),
                                             range(len(self.data.experiments_results_full[experiment].keys()))):
                data_max[time_step_id] = self.data.experiments_results_full[experiment][data_ts].max()
                data_min[time_step_id] = self.data.experiments_results_full[experiment][data_ts].min()

            # Plot
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            ax1.plot(self.data.time_steps * self.constants.SEC2A, data_max, 'b-', label="toc_max")
            ax2.plot(self.data.time_steps * self.constants.SEC2A, data_min, 'r-', label="toc_min")

            ax1.set_title('Change of max and min value of {} \n (depending on time step)'.format(self.config.parameter))
            ax1.set_xlabel('Time [a]')
            ax1.set_ylabel('{} max []'.format(self.config.parameter), color='b')
            ax2.set_ylabel('{} min []'.format(self.config.parameter), color='r')

            plt.tight_layout()
            plt.show()

            output_file_name = "{}_min_max_{}".format(self.config.parameter,
                                                      experiment)

            self.__save_figure(fig, output_file_name)

    def __save_figure(self, fig, output_file_name):
        for ext in self.config.plot_ext:
            fig.savefig(os.path.join(self.config.output_path,
                                     "{}.{}".format(output_file_name, ext)),
                        bbox_inches='tight')

    def __read_config(self, settings):
        # Those are mandatory and have no default value
        missing_setup = []

        if 'parameter' in settings.keys():
            self.config.parameter = settings['parameter']
        else:
            missing_setup.append('parameter')

        if 'path' in settings.keys():
            self.config.path = settings['path']
        else:
            missing_setup.append('path')

        if 'points_file' in settings.keys():
            self.config.points_file = settings['points_file']
        else:
            missing_setup.append('points_file')

        if len(missing_setup) == 0:
            print('Following mandatory fields are missing: ')
            print(missing_setup)

        # Those are optional and have default values
        if 'points_plot' in settings.keys():
            self.config.points_plot = settings['points_plot']

        if 'plot_legend' in settings.keys():
            self.config.plot_legend = settings['plot_legend']

        if 'color_ref' in settings.keys():
            self.config.color_ref = settings['color_ref']
        else:
            self.config.color_ref = 'red'

        if 'colors' in settings.keys():
            self.config.colors = settings['colors']
        else:
            self.config.colors = [plt.cm.nipy_spectral(i) for i in np.linspace(0, 1, 50)]
            '''
            self.config.colors = ['blue', 'orange', 'teal', 'violet', 'purple', 'bisque',
                                  'gold', 'yellowgreen', 'aqua', 'indigo', 'grey', 'olive',
                                  'sienna', 'green', 'salmon', 'fuchsia', 'black', 'orchid',
                                  'lime', 'azure', 'orangered', 'coral', 'turquoise', 'wheat',
                                  'ivory', 'brown', 'crimson', 'magenta', 'maroon', 'silver',
                                  'tan']
                                  '''

        if 'plot_ext' in settings.keys():
            self.config.plot_ext = settings['plot_ext']
        else:
            self.config.plot_ext = ['png']

        if 'include_legend' in settings.keys():
            self.config.include_legend = settings['include_legend']

        if 'reference_experiment' in settings.keys():
            self.config.reference_experiment = settings['reference_experiment']
