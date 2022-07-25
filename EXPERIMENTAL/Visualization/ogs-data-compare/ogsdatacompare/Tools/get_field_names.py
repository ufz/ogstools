import vtuIO
import os


def get_field_names(pvdio_object):

    file_name = os.path.join(pvdio_object.folder,
                             pvdio_object.vtufilenames[0])
    data = vtuIO.VTUIO(file_name, dim=2)

    return data.get_point_field_names()
