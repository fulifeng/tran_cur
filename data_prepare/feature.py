import copy
from nmf import get_missing_entry
import numpy as np
import os
import read_feature as rf
from util import clean_uni_name, feature_scale
class feature:
    def __init__(self):
        self.work_dir = '/home/ffl/nus/MM/cur_trans/data/prepared/'
        self.sel_uni_fname = os.path.join(self.work_dir, 'candidate_universites.csv')
        self.feature_fname_fname = '/home/ffl/nus/MM/cur_trans/data/prepared/ground_truth/feature_fnames_ori.json'
        self.reader = rf.feature_reader(self.feature_fname_fname, self.sel_uni_fname, read_type=1, feature_path=os.path.join(self.work_dir,
                                                                                                                             'ori_feature'))
        # self.feature_nmf_fname = os.path.join(self.work_dir, 'feature_all_nmf_0.02.csv')

    def format_rl_feature(self, gt_fname):
        # currently, read original feature, after NMF, use the output of NMF
        # data = self.reader.read_feature()
        # print 'data shape:', data.shape
        feature = np.genfromtxt(os.path.join(self.work_dir, 'feature_all_nmf_0.02.csv'), delimiter=',', dtype=float)
        print 'feature shape:', feature.shape

        # read in selected universities
        sel_unis = np.genfromtxt(self.sel_uni_fname, dtype=str, delimiter=',')
        print '#selected universities:', len(sel_unis)
        uni_index = {}
        for ind, uni in enumerate(sel_unis):
            uni_index[uni] = ind

        # read ground truth pair
        data = np.genfromtxt(os.path.join(self.work_dir, 'ground_truth/', gt_fname), delimiter=',', dtype=str)
        print 'ground truth pair read in with shape:', data.shape
        positive_pair = 0
        negative_pair = 0
        gt_pairs = []
        for pair in data:
            if pair[2] == '1':
                positive_pair += 1
                gt_pair = [pair[0], pair[1]]
            elif pair[2] == '-1':
                negative_pair += 1
                gt_pair = [pair[1], pair[0]]
            else:
                print 'unexpected pair:', pair
                break
            gt_pairs.append(gt_pair)
        print 'ground truth pairs: #all %d, #positive %d, #negative %d' % (len(gt_pairs), positive_pair, negative_pair)

        # write to RankLib format
        with open(os.path.join(self.work_dir, 'ground_truth/', gt_fname.replace('.csv', '_feature.csv')), 'w') as fout:
            for ind, gtp in enumerate(gt_pairs):
                rl_str_pos = self._rl_feature(2, ind + 1, feature[uni_index[gtp[0]]])
                rl_str_neg = self._rl_feature(1, ind + 1, feature[uni_index[gtp[1]]])
                fout.write(rl_str_pos + '\n' + rl_str_neg + '\n')

    def format_rl_feature_test(self, uni_list_fname = None):
        feature = np.genfromtxt(os.path.join(self.work_dir, 'feature_all_nmf_0.02.csv'), delimiter=',', dtype=float)
        print 'feature shape:', feature.shape

        # read in selected universities
        all_unis = np.genfromtxt(self.sel_uni_fname, dtype=str, delimiter=',')
        print '#all universities:', len(all_unis)
        uni_index = {}
        for ind, uni in enumerate(all_unis):
            uni_index[uni] = ind

        # if no university is further selected, then all universities in the self.sel_uni_fname are used for testing
        if uni_list_fname is None:
            sel_unis = copy.copy(all_unis)
            uni_list_fname = self.sel_uni_fname
        else:
            uni_list_fname = os.path.join(self.work_dir, uni_list_fname)
            data = np.genfromtxt(uni_list_fname, dtype=str, delimiter=',')
            if len(data.shape) == 1:
                sel_unis = copy.copy(data)
            else:
                sel_unis = []
                for uni_kv in data:
                    sel_unis.append(uni_kv[0])
        print '#selected universities:', len(sel_unis)

        # write to RankLib format
        with open(uni_list_fname.replace('.csv', '_feature.csv'), 'w') as fout:
            for uni in sel_unis:
                rl_str = self._rl_feature(1, 1, feature[uni_index[uni]])
                fout.write(rl_str + '\n')

    def nmf_com_feature(self):
        # data_fname = '/home/ffl/nus/MM/complementary/chinese_university_ranking/experiment/matrix_fea    ture/university_lines_aver_trans.csv'
        # data = np.genfromtxt(data_fname, delimiter=',', dtype=float)
        data = self.reader.read_feature()
        print 'data shape:', data.shape
        # feature_scale(data)
        R = copy.copy(data)
        k = data.shape[1]
        iters = 500
        alpha = 0.0004
        beta = 0.1
        print 'latent concepts:', k, 'iter:', iters, 'alpha:', alpha, 'beta:', beta
        R = get_missing_entry(R, k=k, steps=iters, beta=beta, missing_denotation=-1)
        np.savetxt('../feature_all_nmf_' + '{:.4f}'.format(beta) + '.csv', R, delimiter=',')

    # def selelct_feature(self):
    #     sel_unis = np.genfromtxt(self.sel_uni_fname, dtype=str, delimiter=',')
    #     print '#selected universities:', len(sel_unis)
    #     fin_unis = np.genfromtxt(self.final_list_fname, dtype=str, delimiter=',')
    #     print '#previous final list:', len(fin_unis)
    #     uni_index = {}
    #     for ind, uni in enumerate(fin_unis):
    #         uni_index[clean_uni_name(uni)] = ind
    #     fea_mat = self.reader.read_feature()
    #     print 'feature matrix shape:', fea_mat.shape
    #     sel_fea_mat = []
    #     for uni in sel_unis:
    #         # assert uni in uni_index.keys(), 'university %s not in previous final university list' % (uni)
    #         if not uni in uni_index.keys():
    #             print uni
    #         else:
    #             sel_fea_mat.append(fea_mat[uni_index[uni]])
    #     print '#features:', len(sel_fea_mat)
    #     np.savetxt(os.path.join(self.work_dir, 'feature.csv'), sel_fea_mat, fmt='%s', delimiter=',')

    def transfer_feature(self):
        final_list_fname = self.sel_uni_fname
        feature_fname_fname = '/home/ffl/nus/MM/cur_trans/data/prepared/ground_truth/feature_fnames_ori.json'
        reader = rf.feature_reader(feature_fname_fname, final_list_fname, read_type=5)
        in_path = '/home/ffl/nus/MM/complementary/chinese_university_ranking/experiment/ori_feature/'
        out_path = '/home/ffl/nus/MM/cur_trans/data/prepared/ori_feature/'
        reader.transfer_feature(in_path, out_path)

    def _rl_feature(self, rel_score, qid, fea):
        rl_str = str(rel_score) + ' qid:' + str(qid)
        for fid, fea_value in enumerate(fea):
            rl_str += ' ' + str(fid + 1) + ':' + str(fea_value)
        return rl_str

if __name__ == '__main__':
    f = feature()
    # f.selelct_feature()
    # f.transfer_feature()
    # f.nmf_com_feature()
    # f.format_rl_feature()
    # print f._rl_feature(3.5, 12, [1, 2, 0.1, 4])
    # f.format_rl_feature('pair_wise_top_university.csv')
    # f.format_rl_feature('pair_wise_top_university_all.csv')
    # f.format_rl_feature_test('ground_truth/top_university.csv')
    f.format_rl_feature_test()