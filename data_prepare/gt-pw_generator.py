# -*- coding: utf-8 -*-
import csv
import math
import numpy as np
try:
    set
except NameError:
    from sets import Set as set
from util import clean_uni_name, longest_common_substring

class gt_generator:
    def __init__(self, fnames):
        self.fnames = fnames
        self.ranks = []
        # self.inter_lists = []

    def generate(self):
        # read multiple rankings
        inter_list_initialized = False
        for fname in self.fnames:
            data = np.genfromtxt(fname, delimiter=',', dtype=str)
            print 'read a matrix with shape:', data.shape, 'from:', fname
            cur_rank = {}
            for i in range(data.shape[0]):
                cur_rank[clean_uni_name(data[i][1])] = int(data[i][0])
            print 'current rank size:', len(cur_rank)
            self.ranks.append(cur_rank)
            # # initialize inter_list
            # if inter_list_initialized == False:
            #     inter_list_initialized = True
            #     for i in range(data.shape[0]):
            #         self.inter_lists.append(data[i][0])

        # get the common names that occur in all rankings
        common_names = set()
        list_count = len(self.ranks)
        for uni_name in self.ranks[0].iterkeys():
            is_common = True
            for i in range(1, list_count):
                if not uni_name in self.ranks[i]:
                    is_common = False
                    break
            if is_common:
                common_names.add(uni_name)
        print '#commont university names: ', len(common_names)

        '''This doesn't work as too many false positive samples: shi fan da xue, gong cheng da xue, and etc.'''
        # # check potential repeat university names
        # for name1 in common_names:
        #     half_len = len(name1) / 2
        #     for name2 in common_names:
        #         if not name1 == name2:
        #             if len(longest_common_substring(name1, name2)) > half_len:
        #                 print 'potential pair:', name1, name2

        # traverse the rankings and print the names that do not belong to the common names
        for i in range(len(self.ranks)):
            rank_list = self.ranks[i]
            occur_alone_count = 0
            for uni_name in rank_list.iterkeys():
                if not uni_name in common_names:
                    print uni_name
                    occur_alone_count += 1
            print '#universities only occurred in:', self.fnames[i], '\t', occur_alone_count, '\n'

        # enumerate all common name pairs and write the ground truth into 'ground_truth.csv'
        rank_count = len(self.ranks)
        pairs_compared = 0
        pairs_possitive = 0
        gt_list = []
        for ind1, name1 in enumerate(common_names):
            for ind2, name2 in enumerate(common_names):
                if ind1 < ind2:
                    pairs_compared += 1
                    # check whether all rank lists consistently rank them
                    better = True
                    worse = True
                    for i in range(rank_count):
                        # name1 is better than name2
                        if self.ranks[i][name1] > self.ranks[i][name2]:
                            better = False
                        # name2 is better than name1
                        if self.ranks[i][name1] < self.ranks[i][name2]:
                            worse = False
                    # name1 and name2 are equally ranked in all the lists
                    if better == True and worse == True:
                        print name1, name2, 'have same rank in all the lists'
                        cur_pair = [name1, name2, 0]
                        for i in range(rank_count):
                            cur_pair.append(str(self.ranks[i][name2] - self.ranks[i][name1]))
                        gt_list.append(cur_pair)
                    # name1 is better
                    if better == True and worse == False:
                        cur_pair = [name1, name2, 1]
                        for i in range(rank_count):
                            cur_pair.append(str(self.ranks[i][name2] - self.ranks[i][name1]))
                        gt_list.append(cur_pair)
                        pairs_possitive += 1
                    # name2 is better
                    if better == False and worse == True:
                        cur_pair = [name2, name1, 1]
                        for i in range(rank_count):
                            cur_pair.append(str(self.ranks[i][name1] - self.ranks[i][name2]))
                        gt_list.append(cur_pair)
                        pairs_possitive += 1
                    # name 1 and name 2 are not distinguished
                    if better == False and worse == False:
                        cur_pair = [name1, name2, 0]
                        for i in range(rank_count):
                            cur_pair.append(str(self.ranks[i][name2] - self.ranks[i][name1]))
                        gt_list.append(cur_pair)
        print 'pairs compared:', pairs_compared
        print 'pairs generated:', len(gt_list)
        print 'pairs labeled as 1:', pairs_possitive
        np.savetxt('ground_truth.csv', gt_list, fmt='%s', delimiter=',')

if __name__ == "__main__":
    # default parameters
    # cuaa_2016.csv  ipin_2016.csv  wsl_2016.csv  wuhan_2016.csv
    # fnames = ['cuaa_2016.csv', 'transferred_wsl_2016.csv', 'wuhan_2016.csv']
    fnames = ['cuaa_2015.csv', 'transferred_wsl_2015.csv', 'wuhan_2015.csv']
    g = gt_generator(fnames)
    g.generate()
