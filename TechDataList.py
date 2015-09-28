__author__ = 'jt306'

from Get_Full_Path import  get_full_path


def get_tech_address(line_num):
    with open(get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/techtc_files_49.txt"), "r") as infile:
        line = infile.readlines()[line_num]
        print (get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/{}/vectors.dat".format(line[:-1])))

