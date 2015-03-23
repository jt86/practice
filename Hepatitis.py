import csv
import numpy as np
from Get_Full_Path import get_full_path
import logging


def get_hepatitis_data(ignore_missing=False):
    with open(get_full_path("Desktop/Privileged_Data/new_data/hepatitis_data.csv"), "r+") as infile:
        features_array = []
        reader = csv.reader(infile, dialect=csv.excel_tab)
        for row in reader:
            features_array.append(str(row).translate(None, "[]'").split(","))

        if ignore_missing == True:
            logging.info('ignoring certain features with missing values')
            # features_array=np.ma.array(features_array, mask = False)
            # features_array.mask[16,19] = True
            feats_to_delete = [18]
            features_array = np.delete(features_array, feats_to_delete, 1)
            logging.info(features_array.shape)
        print
        features_array = np.array([line for line in features_array if '?' not in line])

        labels_array = np.array(features_array[:, 0], dtype=int)

        logging.info(labels_array.shape)
        features_array = np.array(features_array[:, 1:], dtype=float)
        logging.info(features_array.shape)
        return features_array, labels_array


get_hepatitis_data(True)
