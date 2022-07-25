from ogsdatacompare.ogs_data_compare import OGSDataCompare
import os


settings = {'parameter': 'pressure',
            #'path': os.path.join('/home/kiszkurn/Dev/Data/Experiment1'),
            #'points_file': os.path.join('/home/kiszkurn/Dev/Data', 'points_isotropic.csv'),
            'path': os.path.join(os.path.join('Documentation', 'Example', 'Data')),
            'points_file': os.path.join('Documentation', 'Example', 'Data', 'points.csv'),

            'plot_ext': ['png'],
            'include_legend': True
            }

data_compare = OGSDataCompare(settings)
data_compare.read_data()

data_compare.ogs_compare_separate()
data_compare.ogs_compare_time_point(time_step=[2.3, 2.7])
data_compare.ogs_compare_3d_view(time_step=[2.3, 2.7])  # time_step=[1.0, 2.0, 3.0, 4.0, 5.0])

data_compare.read_data_full()
data_compare.plot_min_max()
