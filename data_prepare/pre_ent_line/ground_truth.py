# -*- coding: utf-8 -*-
import copy
import MySQLdb
import numpy as np
import operator
import os
import random
import re
import sys
import traceback
# from MySQLdb import cursors
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from util import clean_uni_name, read_ranking_list
from util_expand import name_matcher, clean_uni_name_en


class gt_constructor:
    def __init__(self, rank_tables):
        self.rank_tables = rank_tables
        self.rank_dicts = []
        self.rank_lists = []

    '''
        Select universities from university_yiben.csv by filtering out
        military and art universities. (Not used here)
    '''
    def select_can_univ(self, uni_list_path):
        merged_rank_list = np.genfromtxt(os.path.join(uni_list_path, 'university_merged_list.csv'), dtype=str, delimiter=',')
        print 'read a matrix with shape:', merged_rank_list.shape
        merged_rank_dict = read_ranking_list(os.path.join(uni_list_path, 'university_merged_list.csv'), dtype=int)
        yiben_set = set(np.genfromtxt(os.path.join(uni_list_path, 'university_yiben.csv'), dtype=str))
        print '#yiben universities:', len(yiben_set)
        yiben_not_covered = 0
        for uni in yiben_set:
            if uni not in merged_rank_dict.keys():
                yiben_not_covered += 1
                print uni
        data = np.genfromtxt('/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/univ_category.csv', delimiter=',',
                             dtype=str)
        print 'read a matrix with shape: ', data.shape
        univ_types = {}
        for d in data:
            univ_types[clean_uni_name(d[0])] = d[1]
        arts_chi_str = '艺术'
        with open(os.path.join(uni_list_path, 'candidate_universites.csv'), 'w') as fout:
            for uni in yiben_set:
                if not uni in univ_types.keys():
                    fout.write('%s\n' % (uni))
                    print 'without category:', uni
                    continue
                if not univ_types[uni] == arts_chi_str:
                    fout.write('%s\n' % (uni))

    '''
        Generate pair-wise ground truth. (Not used here)
    '''
    def generate_pw_gt(self, rank_list_path):
        common_names = set()
        # data = np.genfromtxt(os.path.join(rank_list_path, 'top_university.csv'), dtype=str, delimiter=',')
        data = np.genfromtxt(os.path.join(rank_list_path, 'candidate_universites.csv'), dtype=str, delimiter=',')
        # print data.shape, data.shape[0], len(data.shape)
        if len(data.shape) == 1:
            common_names = set(data)
        else:
            for uni_kv in data:
                common_names.add(uni_kv[0])
        print '#universities selected:', len(common_names)
        # read entrance lines
        lin_cou, ent_lins = self._read_entrance_lines()

        # enumerate all common name pairs and write the ground truth into 'ground_truth.csv'
        rank_count = lin_cou
        pairs_compared = 0
        pairs_possitive = 0
        pairs_negative = 0
        gt_list = []
        better_counts = [0] * (lin_cou + 1)
        worse_counts = [0] * (lin_cou + 1)
        tie_counts = [0] * (lin_cou + 1)
        occur_counts = [0] * (lin_cou + 1)
        for ind1, name1 in enumerate(common_names):
            for ind2, name2 in enumerate(common_names):
                if ind1 < ind2:
                    pairs_compared += 1
                    # check whether all rank lists consistently rank them
                    better = 0
                    worse = 0
                    tie = 0
                    for i in range(rank_count):
                        # name1 is better than name2
                        if not (name1 in ent_lins.keys() and name2 in ent_lins.keys()):
                            continue
                        rank1 = ent_lins[name1][i]
                        rank2 = ent_lins[name2][i]
                        if rank1 == -1 or rank2 == -1:
                            continue
                        if rank1 > rank2:
                            better += 1
                        # name2 is better than name1
                        if rank1 < rank2:
                            worse += 1
                        if rank1 == rank2:
                            tie += 1
                            # print rank1, rank2
                    occur = better + worse + tie
                    better_counts[better] += 1
                    worse_counts[worse] += 1
                    tie_counts[tie] += 1
                    occur_counts[occur] += 1
                    # if not occur == 0:
                    if occur > 9:
                        if float(better) / occur > 0.6:
                            label = 1
                        if float(worse) / occur > 0.6:
                            label = -1
                            # label = self._label_pair(better, worse, occur)
                        if label == 1:
                            pairs_possitive += 1
                            gt_list.append([name1, name2, label, occur, better])
                        elif label == -1:
                            pairs_negative += 1
                            # print name1, name2
                            gt_list.append([name1, name2, label, occur, worse])
                    else:
                        print name1, name2
        print 'better', ','.join(map(str, better_counts))
        print 'worse', ','.join(map(str, worse_counts))
        print 'tie', ','.join(map(str, tie_counts))
        print 'occur', ','.join(map(str, occur_counts))
        print 'pairs compared:', pairs_compared
        print 'pairs generated:', len(gt_list)
        print 'pairs labeled as 1:', pairs_possitive
        print 'pairs labeled as -1:', pairs_negative
        # np.savetxt(os.path.join(rank_list_path, 'ground_truth/pair_wise_top_university.csv'), gt_list, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(rank_list_path, 'ground_truth/pair_wise_first_level.csv'), gt_list, fmt='%s', delimiter=',')

    '''
        Generate pair-wise ground truth with expansion on pairs between top-50
        universities and the remaining. (Not used here)
    '''
    def generate_pw_gt_expand(self, rank_list_path):
        top_unis = set()
        data = np.genfromtxt(os.path.join(rank_list_path, 'top_university.csv'), dtype=str, delimiter=',')
        # print data.shape, data.shape[0], len(data.shape)
        if len(data.shape) == 1:
            top_unis = set(data)
        else:
            for uni_kv in data:
                top_unis.add(uni_kv[0])
        print '#universities selected:', len(top_unis)
        data = np.genfromtxt(os.path.join(rank_list_path, 'candidate_universites.csv'), dtype=str, delimiter=',')
        print '#candidate universities:', len(data)
        bot_unis = set()
        for uni_kv in data:
            if not uni_kv in top_unis:
                bot_unis.add(uni_kv)
        print '#bottom universities:', len(bot_unis)

        # read entrance lines
        lin_cou, ent_lins = self._read_entrance_lines()
        # enumerate all common name pairs and write the ground truth into 'ground_truth.csv'
        # rank_count = len(self.rank_lists)
        rank_count = lin_cou
        pairs_compared = 0
        pairs_possitive = 0
        pairs_negative = 0
        gt_list = []
        better_counts = [0] * (lin_cou + 1)
        worse_counts = [0] * (lin_cou + 1)
        tie_counts = [0] * (lin_cou + 1)
        occur_counts = [0] * (lin_cou + 1)
        name_pairs = []
        for ind1, name1 in enumerate(top_unis):
            for ind2, name2 in enumerate(bot_unis):
                name_pairs.append((name1, name2))
        for ind1, name1 in enumerate(top_unis):
            for ind2, name2 in enumerate(top_unis):
                if ind1 < ind2:
                    name_pairs.append((name1, name2))
        # for ind1, name1 in enumerate(top_unis):
        #     for ind2, name2 in enumerate(bot_unis):
                ## if ind1 < ind2:
        for name1, name2 in name_pairs:
            pairs_compared += 1
            # check whether all rank lists consistently rank them
            better = 0
            worse = 0
            tie = 0
            for i in range(rank_count):
                # name1 is better than name2
                if not (name1 in ent_lins.keys() and name2 in ent_lins.keys()):
                    continue
                rank1 = ent_lins[name1][i]
                rank2 = ent_lins[name2][i]
                if rank1 == -1 or rank2 == -1:
                    continue
                if rank1 > rank2:
                    better += 1
                # name2 is better than name1
                if rank1 < rank2:
                    worse += 1
                if rank1 == rank2:
                    tie += 1
                # print rank1, rank2
            occur = better + worse + tie
            better_counts[better] += 1
            worse_counts[worse] += 1
            tie_counts[tie] += 1
            occur_counts[occur] += 1
            # if not occur == 0:
            if occur > 9:
                if float(better) / occur > 0.6:
                    label = 1
                if float(worse) / occur > 0.6:
                    label = -1
            # label = self._label_pair(better, worse, occur)
                if label == 1:
                    pairs_possitive += 1
                    gt_list.append([name1, name2, label, occur, better])
                elif label == -1:
                    pairs_negative += 1
                    # print name1, name2
                    gt_list.append([name1, name2, label, occur, worse])
            else:
                print name1, name2
        print 'better', ','.join(map(str, better_counts))
        print 'worse', ','.join(map(str, worse_counts))
        print 'tie', ','.join(map(str, tie_counts))
        print 'occur', ','.join(map(str, occur_counts))
        print 'pairs compared:', pairs_compared
        print 'pairs generated:', len(gt_list)
        print 'pairs labeled as 1:', pairs_possitive
        print 'pairs labeled as -1:', pairs_negative
        ofname = os.path.join(rank_list_path, 'ground_truth', 'pair_wise_top_university_expand.csv')
        # print rank_list_path, ofname
        np.savetxt(ofname, gt_list, fmt='%s', delimiter=',')

    '''
        Generate pair-wise ground truth for each province and each category (
        literal/science)
    '''
    def generate_pw_gt_province(self, rank_list_path):
        top_unis = set()
        data = np.genfromtxt(os.path.join(rank_list_path, 'top_university.csv'), dtype=str, delimiter=',')
        # print data.shape, data.shape[0], len(data.shape)
        if len(data.shape) == 1:
            top_unis = set(data)
        else:
            for uni_kv in data:
                top_unis.add(uni_kv[0])
        print '#universities selected:', len(top_unis)
        data = np.genfromtxt(os.path.join(rank_list_path, 'candidate_universites.csv'), dtype=str, delimiter=',')
        can_unis = set(data)
        print '#candidate universities:', len(can_unis)
        bot_unis = set()
        for uni_kv in data:
            if not uni_kv in top_unis:
                bot_unis.add(uni_kv)
        print '#bottom universities:', len(bot_unis)

        # read entrance lines
        lin_cou, ent_lins = self._read_entrance_lines()
        # read titles
        titles = []
        with open('/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/university_lines_batch_2015.csv') as fin:
            line = fin.readline()
            titles = line.strip().split(',')
            titles = titles[1:len(titles)]
        print '\n'.join(titles)
        name_pairs = []
        for ind1, name1 in enumerate(can_unis):
            for ind2, name2 in enumerate(can_unis):
                if ind1 < ind2:
                    name_pairs.append((name1, name2))
        # enumerate all common name pairs and write the ground truth into 'ground_truth.csv'
        # rank_count = len(self.rank_lists)
        for lin_ind in range(lin_cou):
            rank_count = 1
            pairs_compared = 0
            pairs_possitive = 0
            pairs_negative = 0
            gt_list = []
            better_counts = [0] * (rank_count + 1)
            worse_counts = [0] * (rank_count + 1)
            tie_counts = [0] * (rank_count + 1)
            occur_counts = [0] * (rank_count + 1)
            ffb = open(os.path.join(rank_list_path, 'ground_truth', 'gt_pw_province', 'pair_wise_first_level_' + titles[lin_ind] + '.csv'), 'w')
            ftue = open(os.path.join(rank_list_path, 'ground_truth', 'gt_pw_province', 'pair_wise_top_university_expand_' + titles[lin_ind] +
                                     '.csv'), 'w')
            for name1, name2 in name_pairs:
                pairs_compared += 1
                # check whether all rank lists consistently rank them
                better = 0
                worse = 0
                tie = 0
                for i in range(lin_ind, lin_ind + 1):
                    # name1 is better than name2
                    if not (name1 in ent_lins.keys() and name2 in ent_lins.keys()):
                        continue
                    rank1 = ent_lins[name1][i]
                    rank2 = ent_lins[name2][i]
                    if rank1 == -1 or rank2 == -1:
                        continue
                    if rank1 > rank2:
                        better += 1
                    # name2 is better than name1
                    if rank1 < rank2:
                        worse += 1
                    if rank1 == rank2:
                        tie += 1
                        # print rank1, rank2
                occur = better + worse + tie
                better_counts[better] += 1
                worse_counts[worse] += 1
                tie_counts[tie] += 1
                occur_counts[occur] += 1
                # if not occur == 0:
                if occur > 0:
                    temp_pair = []
                    if float(better) / occur > 0.6:
                        label = 1
                    if float(worse) / occur > 0.6:
                        label = -1
                        # label = self._label_pair(better, worse, occur)
                    if label == 1:
                        pairs_possitive += 1
                        temp_pair = [name1, name2, label]
                    elif label == -1:
                        pairs_negative += 1
                        temp_pair = [name1, name2, label]
                    if label == 1 or label == -1:
                        ffb.write(name1 + ',' + name2 + ',' + str(label) + '\n')
                        if not (name1 in bot_unis and name2 in bot_unis):
                            ftue.write(name1 + ',' + name2 + ',' + str(label) + '\n')
                # else:
                #     print name1, name2
            print 'better', ','.join(map(str, better_counts))
            print 'worse', ','.join(map(str, worse_counts))
            print 'tie', ','.join(map(str, tie_counts))
            print 'occur', ','.join(map(str, occur_counts))
            print 'pairs compared:', pairs_compared
            print 'pairs generated:', len(gt_list)
            print 'pairs labeled as 1:', pairs_possitive
            print 'pairs labeled as -1:', pairs_negative
            print '______________________________________'
            ffb.close()
            ftue.close()
            # if lin_ind > 4:
            #     break

    '''
        Merge the selected ranking lists into a heuristic one by a recurrent
        algorithm (Not used here)
    '''
    def heuristic_rank(self, rank_list_path):
        self._initialize_ranks(rank_list_path)
        '''
            For each rank list, we first create a list with set elements: [set([universities with the highest rank]),
                                                                            set([universities with the second highest rank])]
            The algorithm will select the university that occurs most frequently among top set of all lists and remove the university from all
            lists, then remove empty sets from all lists. It is worth noting that the most frequently occurred university may not be unique.
        '''
        # construct the list with set elements
        rank_set_lists = []
        for rank_list in self.rank_lists:
            prev_rank = int(rank_list[0][1])
            rank_set = set()
            rank_set_list = []
            for uni in rank_list:
                rank = int(uni[1])
                if rank == prev_rank:
                    rank_set.add(uni[0])
                else:
                    rank_set_list.append(rank_set)
                    rank_set = set()
                    rank_set.add(uni[0])
                    prev_rank = rank
            rank_set_list.append(rank_set)
            rank_set_lists.append(rank_set_list)
            # print '#rank sets:', len(rank_set_list)
            # for uni in rank_set_list[len(rank_set_list) - 1]:
            #     print uni
        # generate the ranking list
        gen_rank_list = []
        gen_rank_dict = {}
        init_rank = 1
        while True:
            # find the occur frequency of top universitites
            top_unis = set()
            for i in range(len(rank_set_lists)):
                top_unis = top_unis.union(rank_set_lists[i][0])
            unis_freq = {}
            for uni in top_unis:
                unis_freq[uni] = 0
            for i in range(len(rank_set_lists)):
                for uni in rank_set_lists[i][0]:
                    unis_freq[uni] += 1
            top_unis_sorted = sorted(unis_freq.items(), key=operator.itemgetter(1), reverse=True)
            # update generate ranking list and dictionary
            max_freq = top_unis_sorted[0][1]
            uni_sel = []
            for uni_kv in top_unis_sorted:
                if uni_kv[1] == max_freq:
                    gen_rank_list.append(uni_kv[0])
                    gen_rank_dict[uni_kv[0]] = init_rank
                    uni_sel.append(uni_kv[0])
                else:
                    break
            init_rank += 1
            # update the rank_set_lists, remove selected universities from each list and empty sets
            for i in range(len(rank_set_lists)):
                for rank_set in rank_set_lists[i]:
                    for uni in uni_sel:
                        if uni in rank_set:
                            rank_set.remove(uni)
                rank_set_lists[i] = [item for item in rank_set_lists[i] if len(item) > 0]
                print len(rank_set_lists[i])
            rank_set_lists = [item for item in rank_set_lists if len(item) > 0]
            if len(rank_set_lists) == 0:
                break
        # write out
        with open('university_merged_list.csv', 'w') as fout:
            for uni in gen_rank_list:
                fout.write('%s,%d\n' % (uni, gen_rank_dict[uni]))

    '''
        Post-process the merged ranking list generated by heuristic_rank
        function by filtering out military and art universities as well as
        universities not belongs to the first level.
    '''
    def post_merged_list(self, uni_list_path):
        merged_rank_list = np.genfromtxt(os.path.join(uni_list_path, 'university_merged_list.csv'), dtype=str, delimiter=',')
        print 'read a matrix with shape:', merged_rank_list.shape
        merged_rank_dict = read_ranking_list(os.path.join(uni_list_path, 'university_merged_list.csv'), dtype=int)
        yiben_set = set(np.genfromtxt(os.path.join(uni_list_path, 'university_yiben.csv'), dtype=str))
        print '#yiben universities:', len(yiben_set)
        data = np.genfromtxt('/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/univ_category.csv', delimiter=',', dtype=str)
        print 'read a matrix with shape: ', data.shape
        univ_types = {}
        for d in data:
            univ_types[clean_uni_name(d[0])] = d[1]
        arts_chi_str = '艺术'

        '''here we have a couple of actions to do:
            filter out military universities
            filter out art universities
            filter out universities that are not belongs to the first level
        '''
        print '#initial universities in the merged list:', len(merged_rank_dict)
        for uni in merged_rank_dict.keys():
            if not uni in yiben_set:
                del merged_rank_dict[uni]
        print '#yiben universities and not military in the merged list:', len(merged_rank_dict)
        for uni in merged_rank_dict.keys():
            if uni not in univ_types.keys():
                print 'category missing:', uni
                continue
            if univ_types[uni] == arts_chi_str:
                print uni
                del merged_rank_dict[uni]
        print '#yiben unviersities and not art universities in the merged list:', len(merged_rank_dict)

        # post-processing rank
        sorted_merged_rank = sorted(merged_rank_dict.items(), key=operator.itemgetter(1))
        prev_rank = 1
        ori_pre_rank = 1
        # for i in range(len(sorted_merged_rank)):
        #     if not sorted_merged_rank[i][1] == prev_rank:
        #         sorted_merged_rank[i][1] = i + 1
        with open('university-selected_merged_list.csv', 'w') as fout:
            for i in range(len(sorted_merged_rank)):
                if not sorted_merged_rank[i][1] == ori_pre_rank:
                    fout.write('%s,%d\n' %(sorted_merged_rank[i][0], i + 1))
                    prev_rank = i + 1
                    ori_pre_rank = sorted_merged_rank[i][1]
                else:
                    fout.write('%s,%d\n' % (sorted_merged_rank[i][0], prev_rank))

    '''
        Randomly split the a dataset into training set and testing set (Not
        used here)
    '''
    def split_tra_dev(self, gt_fname, tra_percent = 70):
        train_lines = []
        develop_lines = []
        with open(gt_fname) as fin:
            lines = fin.readlines()
            for line in lines:
                if random.randint(0, 99) < 70:
                    train_lines.append(line)
                else:
                    develop_lines.append(line)
        print '#training samples: %d, #testing samples: %d' % (len(train_lines), len(develop_lines))
        with open(gt_fname.replace('.csv', '_tr.csv'), 'w') as fout:
            for line in train_lines:
                fout.write(line + '\n')
        with open(gt_fname.replace('.csv', '_dev.csv'), 'w') as fout:
            for line in develop_lines:
                fout.write(line + '\n')

    '''
        Read in historical ranking lists (Not used here)
    '''
    def _initialize_ranks(self, rank_list_path):
        # read ranks from the given rank_list_path
        for rank_name in self.rank_tables:
            data = np.genfromtxt(os.path.join(rank_list_path, rank_name + '.csv'), dtype=str, delimiter=',')
            print 'matrix read with shape:', data.shape
            self.rank_lists.append(data)
            rank_dict = {}
            for univ in data:
                rank_dict[univ[0]] = int(univ[1])
            self.rank_dicts.append(rank_dict)
        # size check
        for i in range(len(self.rank_tables)):
            assert len(self.rank_lists[i]) == len(self.rank_dicts[i]), 'size of dictionary and list are not equal: %d\t%d' % \
                                                                       (len(self.rank_dicts[i]), len(self.rank_dicts[i]))
            print len(self.rank_dicts[i])

    '''
        Heuristically label a pair of universities (Not used here)
    '''
    def _label_pair(self, better, worse, occur):
        ranks = len(self.rank_lists)
        assert ranks > 2, 'too few ranking list provided to generate pair-wise ground truth'
        for i in range(3, ranks):
            if occur == 3:
                if better == 3:
                    return 1
                elif worse == 3:
                    return -1
                else:
                    return 0
            elif occur > 3:
                if occur - better < 2:
                    return 1
                elif occur - worse < 2:
                    return -1
                else:
                    return 0
            else:
                return 0

    '''
        Read universities' collage entrance lines in 2015
    '''
    def _read_entrance_lines(self):
        data = np.genfromtxt(os.path.join('/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features',
                                          'university_lines_batch_2015.csv'), dtype=str, delimiter=',')
        # process a little
        ent_lins = {}
        for i in range(1, data.shape[0]):
            temp = []
            for j in range(1, data.shape[1]):
                temp.append(int(data[i][j]))
            ent_lins[data[i][0]] = copy.copy(temp)
        return data.shape[1] - 1, ent_lins

if __name__ == '__main__':
    rank_tables = ['cuaa_2016',
                   'wsl_2017',
                   'rank_2017',
                   # 'ipin_2016',
                   # 'arwu_2016',
                   'qs_2016',
                   'usn_2017']# ,
                   #'the_2017']
    gtc = gt_constructor(rank_tables)
    # gtc.export_rank_lists('/home/ffl/nus/MM/cur_trans/data/prepared/rank_lists',
    #                       '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/eng_name.csv')
    # gtc.heuristic_rank('/home/ffl/nus/MM/cur_trans/data/prepared/rank_lists')
    # gtc.post_merged_list('/home/ffl/nus/MM/cur_trans/data/prepared')
    # gtc.select_can_univ('/home/ffl/nus/MM/cur_trans/data/prepared')
    # gtc.generate_pw_gt('/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction')
    # gtc.generate_pw_gt_expand('/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction')
    # gtc.split_tra_dev('/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/ground_truth/pair_wise_top_university_expand.csv')
    gtc.generate_pw_gt_province('/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction')