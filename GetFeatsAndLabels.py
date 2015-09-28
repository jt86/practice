__author__ = 'jt306'
import logging
import os, sys
import argparse

from Arcene import get_arcene_data
from Gisette import get_gisette_data
from Get_Full_Path import get_full_path
from Madelon import get_madelon_data
from Dorothea import get_dorothea_data
from Vote import get_vote_data
from Heart import get_heart_data
from Haberman import get_haberman_data
from Crx import get_crx_data
from Mushroom import get_mushroom_data
from Hepatitis import get_hepatitis_data
from Cancer import get_cancer_data
from Bankruptcy import get_bankruptcy_data
from Spambase import get_spambase_data
from Musk2 import get_musk2_data
from Musk1 import get_musk1_data
from Ionosphere import get_ionosphere_data
from HillValley import get_hillvalley_data
from Wine import get_wine_data
from Dexter_NEW1 import get_dexter_data
import time

def get_feats_and_labels(input):
    print('input', input)
    if input == 'arcene':
        features_array, labels_array = get_arcene_data()
        tuple = (919, 9920, 1000)
    elif input == 'gisette':
        features_array, labels_array = get_gisette_data()
        tuple = [450, 5000, 500]
    elif input == 'madelon':
        features_array, labels_array = get_madelon_data()
        tuple = [45, 500, 50]
    elif input == 'dorothea':
        features_array, labels_array = get_dorothea_data()
        tuple = [9000,100000,10000]
    elif input == 'vote':
        features_array, labels_array = get_vote_data()
        tuple = [1, 16, 1]
    elif input == 'heart' or input == 'heart2':
        features_array, labels_array = get_heart_data()
        tuple = [1, 13, 1]
    elif input == 'haberman':
        features_array, labels_array = get_haberman_data()
        tuple = [1,4,1]
    elif input == 'crx':
        features_array, labels_array = get_crx_data()
        tuple = [1, 42, 5]
    elif input == 'mushroom':
        features_array, labels_array = get_mushroom_data()
    elif input == 'hepatitis':
        features_array, labels_array = get_hepatitis_data()
        tuple = [1,18,2]
    elif input == 'cancer':
        features_array, labels_array = get_cancer_data()
        tuple = [15, 153, 15]
    elif input == 'bankruptcy':
        features_array, labels_array = get_bankruptcy_data()
        tuple = [1, 18, 1]
    elif input == 'spambase':
        features_array, labels_array = get_spambase_data()
        tuple = [5, 58, 10]
    elif input == 'musk2':
        features_array, labels_array = get_musk2_data()
        tuple = [16,169, 16]
    elif input == 'musk1':
        features_array, labels_array = get_musk1_data()
        tuple = [16, 166, 16]
    elif input == 'ionosphere':
        features_array, labels_array = get_ionosphere_data()
        tuple = [3, 34, 3]
    elif input == 'hillvalley':
        features_array, labels_array = get_hillvalley_data()
        tuple = [9, 101, 10]
    elif input == 'wine':
        features_array, labels_array = get_wine_data()
        tuple = [1, 13, 1]
    elif input == 'dexter':
        features_array, labels_array = get_dexter_data()
        tuple = [1950,20000,2000]
    else:
        features_array, labels_array,tuple = None,None,None
    return features_array, labels_array, tuple