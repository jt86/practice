__author__ = 'jt306'
import logging


def get_full_path(relative_path):
    logging.info('getting full path')
    from socket import gethostname
    import os

    h = gethostname()
    if 'apollo' in h or 'node' in h:
        logging.info('apollo in hostname')
        prefix = '/mnt/lustre/scratch/inf/jt306'
    else:
        prefix = '/Volumes/LocalDataHD/jt306'
    return os.path.join(prefix, relative_path)