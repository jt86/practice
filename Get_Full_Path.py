'''
Takes the end of a path (relative_path) and appends the correct start, depending on host (cluster or local machine)
'''
import socket
__author__ = 'jt306'
def get_full_path(relative_path):

    import os

    h = socket.gethostname()
    if 'apollo' in h or 'node' in h:
        print('apollo in hostname')
        prefix = '/lustre/scratch/inf/jt306'
    elif 'Joes-iMac' in h:
        prefix = '/Users/joe/'
    else:
        prefix = '/Volumes/LocalDataHD/j/jt/jt306'
    return os.path.join(prefix, relative_path)

print (socket.gethostname())