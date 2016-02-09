'''
Takes the end of a path (relative_path) and appends the correct start, depending on host (cluster or local machine)
'''

__author__ = 'jt306'
def get_full_path(relative_path):
    from socket import gethostname
    import os

    h = gethostname()
    if 'apollo' in h or 'node' in h:
        print('apollo in hostname')
        prefix = '/lustre/scratch/inf/jt306'
    else:
        prefix = '/Volumes/LocalDataHD/jt306'
    return os.path.join(prefix, relative_path)