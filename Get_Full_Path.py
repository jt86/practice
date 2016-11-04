'''
Takes the end of a path (relative_path) and appends the correct start, depending on host (cluster or local machine)
'''

__author__ = 'jt306'
def get_full_path(relative_path):
    s
    import os

    h = gethostname()
    print (h)
    if 'apollo' in h or 'node' in h:
        print('apollo in hostname')
        prefix = '/lustre/scratch/inf/jt306'
    else:
        prefix = '/Volumes/LocalDataHD/j/jt/jt306'
    return os.path.join(prefix, relative_path)