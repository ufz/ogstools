from os import scandir as scandir
from os import listdir as listdir
import pandas as pd


def getfilesbyextension(folder, ext="pvd"):
    files = listdir(folder)
    pvdfiles = []
    for file in files:
        if file.split(".")[-1] == ext:
            pvdfiles.append(file)
    del files
    return pvdfiles


def sort_folders(folder_list):

    folders_df = pd.DataFrame(columns=['num_ID', 'Name'])

    i = 0
    for f in folder_list:
        folders_df = pd.concat([folders_df, pd.DataFrame({'num_ID': [i], 'Name': [f]})])
        i += 1

    folders_df = folders_df.sort_values(['Name'])

    folder_list_sorted = []
    for index, row in folders_df.iterrows():
        folder_list_sorted.append(folder_list[row['num_ID']])

    return folder_list_sorted


def detect_experiments(path, sorted_by='name'):
    exp_folders = []
    sub_dirs = scandir(path)
    for f in sub_dirs:
        if f.is_dir():
            potential_pvd_files = getfilesbyextension(f, ext="pvd")
            if len(potential_pvd_files) == 1:
                exp_folders.append(f.name)
    if sorted_by == 'name':
        exp_folders_sorted = sort_folders(exp_folders)
    else:
        exp_folders_sorted = exp_folders

    return exp_folders_sorted


def extract_params(folder_name):

    param_a = {}
    param_b = {}

    param_a['name'] = folder_name.split("_")[0]
    param_a['value'] = float(folder_name.split("_")[1])
    param_b['name'] = folder_name.split("_")[-2]
    param_b['value'] = float(folder_name.split("_")[-1])

    return param_a, param_b
