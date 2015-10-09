__author__ = 'jt306'
from Get_Full_Path import get_full_path

def get_words(line_num):
    with open(get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/techtc_files_49.txt"), "r") as infile:
        line = infile.readlines()[line_num]
        return (get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/{}/features.idx".format(line.strip('\r\n'))))

def get_words_indices(line_num):
    words_indices_dict={}
    with open(get_words(line_num), "r")as infile:
        print (infile)
        for row_num, line in enumerate(infile):
            if row_num>8:
                index,word = line.split()[0],line.split()[1]
                # print ('word',word,'index',index)
                if len(word)<5:
                    print ('short word',word)
                    words_indices_dict[index]=word
    print (len(words_indices_dict))
    return words_indices_dict

get_words_indices(10)