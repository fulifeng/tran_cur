import copy
from nmf import get_missing_entry
import numpy as np
import os
import read_feature as rf
import random
from util import clean_uni_name, feature_scale
class feature:
    def __init__(self):
        # self.work_dir = '/home/ffl/nus/MM/cur_trans/data/prepared/'
        self.work_dir = '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/'
        self.sel_uni_fname = os.path.join(self.work_dir, 'candidate_universites.csv')
        self.feature_fname_fname = '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/ground_truth/feature_fnames_ori.json'
        self.reader = rf.feature_reader(self.feature_fname_fname, self.sel_uni_fname, read_type=1, feature_path=os.path.join(self.work_dir,
                                                                                                                             'ori_feature'))
        # self.feature_nmf_fname = os.path.join(self.work_dir, 'feature_all_nmf_0.02.csv')

    '''
        Write samples (label and features) into libsvm format
    '''
    def format_libsvm_feature_province(self):
        # read in feature
        feature = np.genfromtxt(os.path.join(self.work_dir, 'feature_all_nmf_0.0200.csv'), delimiter=',', dtype=float)
        print 'feature shape:', feature.shape

        # read in entrance line (label)
        data = np.genfromtxt(os.path.join('/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features',
                                          'university_lines_batch_2015.csv'), dtype=str, delimiter=',')
        # process a little
        ent_lins = {}
        for i in range(1, data.shape[0]):
            temp = []
            for j in range(1, data.shape[1]):
                temp.append(int(data[i][j]))
            ent_lins[data[i][0]] = copy.copy(temp)

        # read in selected (candidate) universities
        all_unis = np.genfromtxt(self.sel_uni_fname, dtype=str, delimiter=',')
        print '#all universities:', len(all_unis)
        uni_index = {}
        for ind, uni in enumerate(all_unis):
            uni_index[uni] = ind

        # read top universities
        top_unis = set()
        data = np.genfromtxt(os.path.join(self.work_dir, 'top_university.csv'), dtype=str, delimiter=',')
        # print data.shape, data.shape[0], len(data.shape)
        if len(data.shape) == 1:
            top_unis = set(data)
        else:
            for uni_kv in data:
                top_unis.add(uni_kv[0])
        print '#universities selected:', len(top_unis)

        # generate training, developing, and testing file by province
        titles = ['2015_51_2', '2015_51_6', '2015_50_1', '2015_50_5', '2015_61_1', '2015_61_5', '2015_62_1', '2015_62_5', '2015_63_1',
                  '2015_63_5', '2015_64_1', '2015_64_5', '2015_53_1', '2015_53_5', '2015_52_1', '2015_52_5', '2015_21_1', '2015_21_5',
                  '2015_22_1', '2015_22_5', '2015_23_1', '2015_23_5', '2015_46_1', '2015_46_5', '2015_44_1', '2015_44_5', '2015_45_1',
                  '2015_45_5', '2015_42_1', '2015_42_5', '2015_43_1', '2015_43_5', '2015_41_1', '2015_41_5', '2015_11_11', '2015_11_15',
                  '2015_13_1', '2015_13_5', '2015_12_1', '2015_12_5', '2015_15_1', '2015_15_5', '2015_14_1', '2015_14_5', '2015_33_1',
                  '2015_33_5', '2015_32_1', '2015_32_5', '2015_31_1', '2015_31_5', '2015_37_1', '2015_37_5', '2015_36_1', '2015_36_5',
                  '2015_35_1', '2015_35_5', '2015_34_11', '2015_34_15', '2015_54_1', '2015_54_5', '2015_65_11', '2015_65_15']
        print 'train, dev, test'
        for i, title in enumerate(titles):
            if os.path.isfile(os.path.join(self.work_dir, 'pow_ground_truth', 'tr_' + title + '.csv')):
                continue
            ftr = open(os.path.join(self.work_dir, 'pow_ground_truth', 'tr_' + title + '.csv'), 'w')
            ftra = open(os.path.join(self.work_dir, 'pow_ground_truth', 'tr_all_' + title + '.csv'), 'w')
            fdev = open(os.path.join(self.work_dir, 'pow_ground_truth', 'dev_' + title + '.csv'), 'w')
            fte = open(os.path.join(self.work_dir, 'pow_ground_truth', 'te_' + title + '.csv'), 'w')
            # train_all = [] #elements (label, index)
            # test = []
            test_cou, train_cou, dev_cou = 0, 0, 0
            for uni_kv in uni_index.iteritems():
                if ent_lins[uni_kv[0]][i] == -1:
                    continue
                if uni_kv[0] in top_unis:
                    # train_all.append((ent_lins[uni_kv[0]][i], uni_kv[1]))
                    ftra.write(self._libsvm_feature(ent_lins[uni_kv[0]][i], feature[uni_kv[1]]) + '\n')
                    if random.randint(0, 1) == 0 or random.randint(0, 1) == 0:
                        # is train
                        ftr.write(self._libsvm_feature(ent_lins[uni_kv[0]][i], feature[uni_kv[1]]) + '\n')
                        train_cou += 1
                    else:
                        # is development
                        fdev.write(self._libsvm_feature(ent_lins[uni_kv[0]][i], feature[uni_kv[1]]) + '\n')
                        dev_cou += 1
                else:
                    # test.append((ent_lins[uni_kv[0]][i], uni_kv[1]))
                    fte.write(self._libsvm_feature(ent_lins[uni_kv[0]][i], feature[uni_kv[1]]) + '\n')
                    test_cou += 1
            # print '#train samples: %d\t#develop samples: %d\t#test samples:%d' % (train_cou, dev_cou, test_cou)
            print '%d,%d,%d' % (train_cou, dev_cou, test_cou)
            ftr.close()
            ftra.close()
            fdev.close()
            fte.close()
            # if i > 3:
            #     break

    '''
        Write samples (label and features) into libsvm format
    '''
    def format_rl_feature(self, gt_fname):
        # currently, read original feature, after NMF, use the output of NMF
        # data = self.reader.read_feature()
        # print 'data shape:', data.shape
        # feature = np.genfromtxt(os.path.join(self.work_dir, 'feature_all_nmf_0.0200.csv'), delimiter=',', dtype=float)
        feature = np.genfromtxt(os.path.join(self.work_dir, 'feature_all_nmf_0.0200_ent-lin.csv'), delimiter=',', dtype=float)
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

    '''
        Write samples (label and features) into libsvm format by province
    '''
    def format_rl_feature_province(self, gt_fname):
        # currently, read original feature, after NMF, use the output of NMF
        # data = self.reader.read_feature()
        # print 'data shape:', data.shape
        feature = np.genfromtxt(os.path.join(self.work_dir, 'feature_all_nmf_0.0200.csv'), delimiter=',', dtype=float)
        print 'feature shape:', feature.shape

        # read in selected universities
        sel_unis = np.genfromtxt(self.sel_uni_fname, dtype=str, delimiter=',')
        print '#selected universities:', len(sel_unis)
        uni_index = {}
        for ind, uni in enumerate(sel_unis):
            uni_index[uni] = ind

        titles = ['2015_51_2', '2015_51_6', '2015_50_1', '2015_50_5', '2015_61_1', '2015_61_5', '2015_62_1', '2015_62_5', '2015_63_1',
                  '2015_63_5', '2015_64_1', '2015_64_5', '2015_53_1', '2015_53_5', '2015_52_1', '2015_52_5', '2015_21_1', '2015_21_5',
                  '2015_22_1', '2015_22_5', '2015_23_1', '2015_23_5', '2015_46_1', '2015_46_5', '2015_44_1', '2015_44_5', '2015_45_1',
                  '2015_45_5', '2015_42_1', '2015_42_5', '2015_43_1', '2015_43_5', '2015_41_1', '2015_41_5', '2015_11_11', '2015_11_15',
                  '2015_13_1', '2015_13_5', '2015_12_1', '2015_12_5', '2015_15_1', '2015_15_5', '2015_14_1', '2015_14_5', '2015_33_1',
                  '2015_33_5', '2015_32_1', '2015_32_5', '2015_31_1', '2015_31_5', '2015_37_1', '2015_37_5', '2015_36_1', '2015_36_5',
                  '2015_35_1', '2015_35_5', '2015_34_11', '2015_34_15', '2015_54_1', '2015_54_5', '2015_65_11', '2015_65_15']
        for i in range(len(titles)):
            if os.path.isfile(os.path.join(self.work_dir, 'ground_truth', 'gt_pw_province',
                                        gt_fname.replace('.csv', '_' + titles[i] + '_feature.csv'))):
                continue
            # read ground truth pair
            data = np.genfromtxt(os.path.join(self.work_dir, 'ground_truth', 'gt_pw_province',
                                              gt_fname.replace('.csv', '_' + titles[i] + '.csv')), delimiter=',', dtype=str)
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
            with open(os.path.join(self.work_dir, 'ground_truth', 'gt_pw_province', gt_fname.replace('.csv', '_' + titles[i] + '_feature.csv')),
                      'w') as fout:
                for ind, gtp in enumerate(gt_pairs):
                    rl_str_pos = self._rl_feature(2, ind + 1, feature[uni_index[gtp[0]]])
                    rl_str_neg = self._rl_feature(1, ind + 1, feature[uni_index[gtp[1]]])
                    fout.write(rl_str_pos + '\n' + rl_str_neg + '\n')
            # if i > 4:
            #     break

    '''
        Write testing samples (label and features) into libsvm format
    '''
    def format_rl_feature_test(self, uni_list_fname = None):
        # feature = np.genfromtxt(os.path.join(self.work_dir, 'feature_all_nmf_0.0200.csv'), delimiter=',', dtype=float)
        feature = np.genfromtxt(os.path.join(self.work_dir, 'feature_all_nmf_0.0200_ent-lin.csv'), delimiter=',', dtype=float)
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

    '''
        Complete missing entries with non-negative matrix factorization
    '''
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
        beta = 0.001
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

    '''
        Concatenate features from different sources, do column normalization
        [0, 1]
    '''
    def transfer_feature(self):
        final_list_fname = self.sel_uni_fname
        feature_fname_fname = '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/ground_truth/feature_fnames_ori.json'
        reader = rf.feature_reader(feature_fname_fname, final_list_fname, read_type=5)
        in_path = '/home/ffl/nus/MM/complementary/chinese_university_ranking/experiment/ori_feature/'
        out_path = '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/ori_feature/'
        reader.transfer_feature(in_path, out_path)

    def _libsvm_feature(self, label, feature):
        rl_str = str(label)
        for fid, fea_value in enumerate(feature):
            rl_str += ' ' + str(fid + 1) + ':' + str(fea_value)
        return rl_str

    def _rl_feature(self, rel_score, qid, fea):
        rl_str = str(rel_score) + ' qid:' + str(qid)
        for fid, fea_value in enumerate(fea):
            rl_str += ' ' + str(fid + 1) + ':' + str(fea_value)
        return rl_str

'''
    One time code, select collage entrance line features
'''
def split_feature(feature_fname = 'feature_all_nmf_0.0200.csv'):
    work_dir = '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction'
    feature = np.genfromtxt(os.path.join(work_dir, feature_fname), delimiter=',', dtype=str)
    print 'feature shape:', feature.shape
    ent_lin_feature = feature[:, 17:79]
    print 'ent_line feature shape:', ent_lin_feature.shape
    other_feature = np.concatenate((feature[:, 0:17], feature[:, 79:feature.shape[1]]), axis=1)
    print 'other feature shape:', other_feature.shape
    # write out
    np.savetxt(os.path.join(work_dir, feature_fname.replace('.csv', '_ent-lin.csv')), ent_lin_feature, delimiter=',', fmt='%s')

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
    # f.format_rl_feature_test()

    # entrance line prediction, vote ground truth with entrance line
    # f.format_rl_feature_test()
    # f.format_rl_feature('pair_wise_top_university_expand_tr.csv')
    # f.format_rl_feature('pair_wise_top_university_expand_dev.csv')
    # f.format_rl_feature('pair_wise_top_university_expand.csv')
    # f.format_rl_feature_province('pair_wise_top_university_expand.csv')
    # split_feature()
    f.format_libsvm_feature_province()