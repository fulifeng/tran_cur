# -*- coding: utf-8 -*-
# import numpy as np
import csv
import numpy as np
import os
import re
from sklearn import metrics
from util import feature_scale


class name_matcher:
    def __init__(self, fname):
        # uni_names = np.genfromtxt(fname, dtype=str, delimiter=',')
        with open(fname, 'r') as csvfile:
            uni_names = csv.reader(csvfile)
            # print 'shape:', uni_names.line_num
            try:
                self.eng_dict = {}  # given english name, return chinese name
                for univ in uni_names:
                    # print univ
                    if not univ[1] == '0':
                        # self.eng_dict[re.sub('\(.+?\)', '', univ[1].replace("（", "(").replace('）', ')').
                        #                      replace('`', '\'').replace('’', '\'').lower().replace('&', 'and')).strip()] = univ[0]
                        # if univ[0] == '南京工业大学':
                        #     print clean_uni_name_en(univ[1])
                        self.eng_dict[clean_uni_name_en(univ[1])] = univ[0]
            except csv.Error as e:
                print str(e)
        print 'english name dictionary size:', len(self.eng_dict)

    def get_chi_name(self, eng_name):
        if eng_name.lower() in self.eng_dict.keys():
            return self.eng_dict[eng_name.lower()]
        else:
            return None


def clean_uni_name_en(uni_name):
    if type(uni_name) is unicode:
        uni_name = uni_name.encode('utf8')
    # if uni_name == 'Northeastern University - China':       # In USN_2017
    #     clean_name = 'Northeastern University'
    # elif uni_name == 'Southwest University - China':        # In USN_2017
    #     clean_name = 'Southwest University'
    if uni_name == 'China University of Geosciences (Wuhan)':     # In THE_2017
        clean_name = 'China University of Geosciences,Wuhan'
    elif uni_name == 'China University of Petroleum (Beijing)':     # In THE_2017
        clean_name = 'China University of Petroleum,Beijing'
    elif uni_name == 'Fourth Military Medical University':
        clean_name = 'The Fourth Military Medical University'
    elif uni_name == 'Second Military Medical University':
        clean_name = 'The Second Military Medical University'
    else:
        clean_name = re.sub('\(.+?\)', '', uni_name.replace("（", "(").replace('）', ')')).\
            replace(' ', ' ').replace(' - China', '').\
            replace('Defence', 'Defense').\
            replace('，', ',').replace(', ', ',').\
            replace('`', '\'').replace('’', '\'').\
            replace('&', ' and ').replace('  ', ' ').strip().lower()
    return clean_name


def get_fnames(path_sel, kw_filter):
    fnames = []
    for f in os.listdir(path_sel):
        if f.endswith(kw_filter):
            fnames.append(f)
    return fnames


def initialize_ranks(rank_list_path, rank_tables):
    # read ranks from the given rank_list_path
    rank_lists = []
    rank_dicts = []
    for rank_name in rank_tables:
        data = np.genfromtxt(os.path.join(rank_list_path, rank_name + '.csv'), dtype=str, delimiter=',')
        print 'matrix read with shape:', data.shape
        rank_lists.append(data)
        rank_dict = {}
        for univ in data:
            rank_dict[univ[0]] = int(univ[1])
        rank_dicts.append(rank_dict)
    # size check
    for i in range(len(rank_tables)):
        assert len(rank_lists[i]) == len(rank_dicts[i]), 'size of dictionary and list are not equal: %d\t%d' % \
                                                                   (len(rank_dicts[i]), len(rank_dicts[i]))
        print len(rank_dicts[i])
    return rank_lists, rank_dicts


def cal_median(fea_matrix_fname):
    X = np.genfromtxt(fea_matrix_fname, delimiter=',', dtype=float)
    feature_scale(X)
    print 'data shape:', X.shape
    euc_dis = metrics.pairwise.euclidean_distances(X, X)
    print float(np.median(euc_dis))


if __name__ == '__main__':
    # nama = name_matcher('/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/eng_name.csv')
    # print nama.get_chi_name('Central China Normal University')
    cal_median('/home/ffl/nus/MM/cur_trans/data/prepared/feature_all_nmf_0.02.csv')