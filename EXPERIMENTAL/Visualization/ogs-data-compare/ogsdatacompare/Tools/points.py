#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:20:36 2022

@author: felikskiszkurno
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


def load_points(file_path):
    '''
    Load list of points from csv files exported from Paraview into format
    acceptable by vtuIO

    Parameters
    ----------
    file_path : str
        file name or path to csv file.

    Returns
    -------
    points : dict
        points stored in a dict.

    '''
    points = {}

    points_df = pd.read_csv(file_path)
    #points_df = points_df.sort_values(by=["Points_Magnitude"], ascending=True)
    points_x = points_df["Points_0"].to_numpy()
    points_y = points_df["Points_1"].to_numpy()
    points_z = points_df["Points_2"].to_numpy()

    for point_id in range(len(points_df.index)):
        point_name = "pt_{p_num}".format(p_num=f'{int(point_id):03d}')
        points[point_name] = (points_x[point_id],
                              points_y[point_id],
                              points_z[point_id])
    return points


def plot_points(points_dict, experiments_folder, exp_size=None):
    x = []
    y = []
    labels = []
    for point in points_dict:
        x.append(points_dict[point][0])
        y.append(points_dict[point][1])
        labels.append(point)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, point_label in enumerate(labels):
        ax.annotate(point_label, (x[i], y[i]))
    if exp_size is not None:
        ax.set_xlim([exp_size[0], exp_size[1]])
        ax.set_ylim([exp_size[2], exp_size[3]])
    ax.set_title('Positions of points')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    # fig.show()
    for ext in ['png']:
        fig.savefig(os.path.join(experiments_folder, 'Results', 'points_overview.' + ext))
