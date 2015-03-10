__author__ = 'jt306'

def get_full_path(relative_path):
    from socket import gethostname
    import os

    h = gethostname()
    if 'apollo' in h or 'node' in h:
        prefix = '/mnt/lustre/scratch/inf/jt306'
    else:
        prefix = '/Volumes/LocalDataHD/jt306'
    return os.path.join(prefix, relative_path)