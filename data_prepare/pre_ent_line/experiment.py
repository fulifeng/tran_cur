import copy
import json
import numpy as np
import operator
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import evaluator
from util import read_ranking_list


class experiment:
    def __init__(self, work_path, tool_path, train_fname, test_fname, gt_fname, rl_path, rl_names, sel_uni_fname):
        self.work_path = work_path
        self.tool_path = tool_path
        self.train_fname = train_fname
        self.test_fname = test_fname
        self.gt_fname = gt_fname
        self.rl_path = rl_path
        self.rl_names = rl_names
        os.chdir(self.work_path)

        # read selected universities
        self.sel_unis = []
        data = np.genfromtxt(sel_uni_fname, dtype=str, delimiter=',')
        if len(data.shape) == 1:
            self.sel_unis = copy.copy(data)
        else:
            for uni_kv in data:
                self.sel_unis.append(uni_kv[0])
        print '#universities selected:', len(self.sel_unis)

        # initialize evaluator
        self.evaluator = evaluator.evaluator(self.rl_path, self.rl_names, self.gt_fname)

    def ensemble(self, rsvm_tran_out, rnet_tran_out, ratio):
        assert ratio < 1 and ratio > 0, 'ratio: %f unexpected' % ratio
        rsvm = read_ranking_list(rsvm_tran_out, float)
        rnet = read_ranking_list(rnet_tran_out, float)
        assert len(rsvm) == len(rnet), 'length mismatch'
        rmerged = {}
        for r_kv in rsvm.iteritems():
            if r_kv[0] not in rnet.keys():
                print '%s not found in result of rank_net' % (r_kv[0])
                exit()
            rmerged[r_kv[0]] = r_kv[1] * ratio + rnet[r_kv[0]] * (1 - ratio)
        sor_gen_rl = sorted(rmerged.items(), key=operator.itemgetter(1), reverse=True)
        np.savetxt('ens_' + str(ratio) + '.csv', sor_gen_rl, fmt='%s', delimiter=',')
        # evaluate
        acc, cors = self.evaluator.evaluate(rmerged)
        # print acc, cors
        return acc, cors

    def ens_par_tun(self, rsvm_tran_out, rnet_tran_out):
        ratio = 0.001
        with open('ens_par_tun.log', 'w') as fout:
            best_acc = -1
            for i in range(1, 10):
                acc, cors = runner.ensemble(rsvm_tran_out, rnet_tran_out, ratio * i)
                print '-----------------------------------------------'
                if acc > best_acc:
                    print 'better performance with ratio', ratio * i
                    fout.write('better performance with ratio %f\n' % ratio * i)
                    best_acc = acc
                print acc, cors
                print '-----------------------------------------------'
                fout.write('-----------------------------------------------\nratio:' + str(ratio * i) + '\nperformance:' +
                           str(acc) + '\n' + json.dumps(cors) + '-----------------------------------------------\n')
            fout.write('best performance: %f\n' % best_acc)
        print 'best performance:', best_acc

    def predict(self, tool, mod_out_fname):
        gen_rl_fname = mod_out_fname.replace('mod_', 'gen_')
        if tool == 'rank_svm':
            command = os.path.join(self.tool_path, 'svm_rank_classify') + ' ' + self.test_fname + ' ' + mod_out_fname + ' ' + gen_rl_fname
        elif tool == 'rank_net':
            # java -jar ../../tool/trunk/bin/RankLib.jar -load model -rank ../../data/prepared/candidate_universites_feature.csv -score out
            command = 'java -jar ' + os.path.join(self.tool_path, 'RankLib.jar') + ' -load ' + mod_out_fname + ' -rank ' + self.test_fname + \
                      ' -score ' + gen_rl_fname
        else:
            command = 'no such command'
        os.system(command)
        return gen_rl_fname

    def rank_net_par_tun(self):
        # parameters
        epoch = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
        node = [5, 10, 15, 20, 30, 40, 50, 60]
        lr = [0.00001, 0.00003, 0.00005, 0.00007, 0.00009, 0.0001]
        paras_com = []
        temp_paras = {}
        for ev in epoch:
            temp_paras['epoch'] = ev
            for nv in node:
                temp_paras['node'] = nv
                for lv in lr:
                    temp_paras['lr'] = lv
                    paras_com.append(copy.copy(temp_paras))
        print '%d parameter combinations are generated' % (len(paras_com))

        with open('rank_net_par_tun.log', 'w') as fout:
            best_acc = -1
            best_rl_fname = 'svm_predictions'
            for temp_paras in paras_com:
                mod_out_fname = self.train('rank_net', temp_paras)
                gen_rl_fname = self.predict('rank_net', mod_out_fname)
                acc, cors = self.test('rank_net', gen_rl_fname)
                print '-----------------------------------------------'
                if acc > best_acc:
                    print 'better performance in', gen_rl_fname
                    fout.write('better performance in %s\n' % gen_rl_fname)
                    best_rl_fname = gen_rl_fname
                    best_acc = acc
                print temp_paras
                print acc, cors
                print '-----------------------------------------------'
                fout.write('-----------------------------------------------\nparameters:' + json.dumps(temp_paras) + '\nperformance:' +
                           str(acc) + '\n' + json.dumps(cors) + '-----------------------------------------------\n')
            fout.write('best performance: %f\n' % best_acc)
            fout.write('best generate list: %s\n' % best_rl_fname)
        print 'best performance:', best_acc
        print 'best generate list:', best_rl_fname

    def rank_svm_par_tun(self):
        # parameters
        c = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
        e = [0.01, 0.001, 0.0001]
        # t = [1]
        t = []
        g = [1.77342514625 / 8, 1.77342514625 / 4, 1.77342514625 / 2, 1.77342514625, 1.77342514625 * 2, 1.77342514625 * 4, 1.77342514625 * 8]
        d = [2]
        paras_com = []
        temp_paras = {}
        for cv in c:
            temp_paras['c'] = cv
            for ev in e:
                temp_paras['e'] = ev
                paras_com.append(copy.copy(temp_paras))
                # for tv in t:
                #     temp_paras['t'] = tv
                #     if tv == 0:
                #         paras_com.append(copy.copy(temp_paras))
                #     elif tv == 1:
                #         for dv in d:
                #             temp_paras['d'] = dv
                #             paras_com.append(copy.copy(temp_paras))
                #     elif tv == 2:
                #         for gv in g:
                #             temp_paras['g'] = gv
                #             paras_com.append(copy.copy(temp_paras))
        print '%d parameter combinations are generated' % (len(paras_com))

        with open('rank_svm_par_tun_t1.log', 'w') as fout:
            best_acc = -1
            best_rl_fname = 'svm_predictions'
            for temp_paras in paras_com:
                mod_out_fname = self.train('rank_svm', temp_paras)
                gen_rl_fname = self.predict('rank_svm', mod_out_fname)
                acc, cors = self.test('rank_svm', gen_rl_fname)
                print '-----------------------------------------------'
                if acc > best_acc:
                    print 'better performance in', gen_rl_fname
                    fout.write('better performance in %s\n' % gen_rl_fname)
                    best_rl_fname = gen_rl_fname
                    best_acc = acc
                print temp_paras
                print acc, cors
                print '-----------------------------------------------'
                fout.write('-----------------------------------------------\nparameters:' + json.dumps(temp_paras) + '\nperformance:' +
                           str(acc) + '\n' + json.dumps(cors) + '-----------------------------------------------\n')
            fout.write('best performance: %f\n' % best_acc)
            fout.write('best generate list: %s\n' % best_rl_fname)
        print 'best performance:', best_acc
        print 'best generate list:', best_rl_fname

    def test(self, tool, gen_rl_fname):
        # read generated ranking list
        gen_rl = []
        if tool == 'rank_svm':
            gen_rl = np.genfromtxt(gen_rl_fname, dtype=float, delimiter=',')
        elif tool == 'rank_net':
            data = np.genfromtxt(gen_rl_fname, dtype=str, delimiter='\t')
            print 'data shape:', data.shape
            for r_kv in data:
                gen_rl.append(float(r_kv[2]))
        print '%d ranks in the given file %s' % (len(gen_rl), gen_rl_fname)
        assert len(gen_rl) == len(self.sel_unis), 'length mismatch'
        gen_rank_dict = {}
        for ind, uni in enumerate(self.sel_unis):
            gen_rank_dict[uni] = gen_rl[ind]

        # evaluate
        acc, cors = self.evaluator.evaluate(gen_rank_dict)
        print acc, cors
        return acc, cors

    def train(self, tool, paras):
        mod_out_fname = 'mod'
        for p_kv in paras.iteritems():
            if type(p_kv[1]) == float:
                mod_out_fname = mod_out_fname + '_' + p_kv[0] + '-' + '{:.5f}'.format(p_kv[1])
            else:
                mod_out_fname = mod_out_fname + '_' + p_kv[0] + '-' + str(p_kv[1])
        if tool == 'rank_svm':
            # command = '%s %s %s' % (os.path.join(self.tool_path, 'rank_svm_'), self.train_fname, mod_out_fname)
            command = os.path.join(self.tool_path, 'svm_rank_learn')
            for p_kv in paras.iteritems():
                command = command + ' -' + p_kv[0] + ' ' + str(p_kv[1])
            command = command + ' ' + self.train_fname + ' ' + mod_out_fname
        elif tool == 'rank_net':
            # java -jar ../../tool/trunk/bin/RankLib.jar -train ../../data/prepared/ground_truth/pair_wise_top_university_all_feature.csv -ranker 1 -metric2t NDCG@10 -save model
            command = 'java -jar ' + os.path.join(self.tool_path, 'RankLib.jar') + ' -train ' + self.train_fname + \
                      ' -ranker 1 -metric2t NDCG@5'
            for p_kv in paras.iteritems():
                command = command + ' -' + p_kv[0] + ' ' + str(p_kv[1])
            command = command + ' -save ' + mod_out_fname
        else:
            command = 'no such command'
        print 'command:', command
        os.system(command)
        return mod_out_fname

    def transfer(self, tool, gen_rl_fname):
        # read generated ranking list
        gen_rl = []
        if tool == 'rank_svm':
            gen_rl = np.genfromtxt(gen_rl_fname, dtype=float, delimiter=',')
            print '%d ranks in the given file %s' % (len(gen_rl), gen_rl_fname)
        else:
            data = np.genfromtxt(gen_rl_fname, dtype=str, delimiter='\t')
            print 'data shape:', data.shape
            for r_kv in data:
                gen_rl.append(float(r_kv[2]))
        assert len(gen_rl) == len(self.sel_unis), 'length mismatch'
        gen_rank_dict = {}
        for ind, uni in enumerate(self.sel_unis):
            gen_rank_dict[uni] = gen_rl[ind]
        sor_gen_rl = sorted(gen_rank_dict.items(), key=operator.itemgetter(1), reverse=True)
        np.savetxt(gen_rl_fname + '_sorted.csv', sor_gen_rl, fmt='%s', delimiter=',')

def exp_province():
    titles = ['2015_51_2', '2015_51_6', '2015_50_1', '2015_50_5', '2015_61_1', '2015_61_5', '2015_62_1', '2015_62_5', '2015_63_1',
              '2015_63_5', '2015_64_1', '2015_64_5', '2015_53_1', '2015_53_5', '2015_52_1', '2015_52_5', '2015_21_1', '2015_21_5',
              '2015_22_1', '2015_22_5', '2015_23_1', '2015_23_5', '2015_46_1', '2015_46_5', '2015_44_1', '2015_44_5', '2015_45_1',
              '2015_45_5', '2015_42_1', '2015_42_5', '2015_43_1', '2015_43_5', '2015_41_1', '2015_41_5', '2015_11_11', '2015_11_15',
              '2015_13_1', '2015_13_5', '2015_12_1', '2015_12_5', '2015_15_1', '2015_15_5', '2015_14_1', '2015_14_5', '2015_33_1',
              '2015_33_5', '2015_32_1', '2015_32_5', '2015_31_1', '2015_31_5', '2015_37_1', '2015_37_5', '2015_36_1', '2015_36_5',
              '2015_35_1', '2015_35_5', '2015_34_11', '2015_34_15', '2015_54_1', '2015_54_5', '2015_65_11', '2015_65_15']
    for i, title in enumerate(titles):
        if os.path.exists(os.path.join('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/dir_tun_pro/ranksvm', title)):
            continue
        print 'working on:', title
        os.mkdir(os.path.join('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/dir_tun_pro/ranksvm', title))
        os.chdir(os.path.join('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/dir_tun_pro/ranksvm', title))
        runner = experiment(os.path.join('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/dir_tun_pro/ranksvm', title),
                                # work path
                            '/home/ffl/nus/MM/cur_trans/tool/',  # tool path rank SVM
                            os.path.join('/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/', 'ground_truth', 'gt_pw_province',
                                        'pair_wise_top_university_expand_' + title + '_feature.csv'),
                                # training data
                            '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/candidate_universites_feature.csv',  # testing data
                            os.path.join('/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/', 'ground_truth', 'gt_pw_province',
                                        'pair_wise_first_level_' + title + '.csv'),
                            # ground truth file
                            '/home/ffl/nus/MM/cur_trans/data/prepared/rank_lists/',  # ranking list path
                            ['cuaa_2016', 'wsl_2017', 'rank_2017', 'qs_2016', 'usn_2017'],  # ranking names
                            '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/candidate_universites.csv'  # selected university file
                            )
        runner.rank_svm_par_tun()
        # if i > 4:
        #     break

if __name__ == '__main__':
    runner = experiment(# '/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/with_ent_lin/ranksvm', # work path
                        # '/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/direct_tune/ranknet',  # work path
                        # '/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/only_ent_lin/ranksvm', # work path
                        # '/home/ffl/nus/MM/cur_trans/exp/ensemble',                                              # work path
                        '/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/direct_tune/ensemble', # work path
                        '/home/ffl/nus/MM/cur_trans/tool/',                                                     # tool path rank SVM
                        # '/home/ffl/nus/MM/cur_trans/tool/trunk/bin/',                                           # tool path RankLib
                        '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/ground_truth/pair_wise_top_university_expand_feature.csv',  # training data
                        # '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/ground_truth/pair_wise_top_university_expand_feature_ent-lin.csv',  # training data
                        # '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/ground_truth/pair_wise_top_university_expand_tr_feature.csv', # training data
                        '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/candidate_universites_feature.csv',           # testing data
                        # '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/candidate_universites_feature_ent-lin.csv',  # testing data
                        '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/ground_truth/pair_wise_first_level.csv',      # ground truth file
                        # '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/ground_truth/pair_wise_top_university_expand_dev.csv',  # gt develop file
                        '/home/ffl/nus/MM/cur_trans/data/prepared/rank_lists/',                                 # ranking list path
                        ['cuaa_2016', 'wsl_2017', 'rank_2017', 'qs_2016', 'usn_2017'],                          # ranking names
                        # '/home/ffl/nus/MM/cur_trans/data/prepared/ground_truth/top_university.csv'            # selected university file
                        '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/candidate_universites.csv'                    # selected university file
                        )
    # mod_out_fname = runner.train('rank_svm', {'c': 3, 'e': 0.01})
    # gen_rl_fname = runner.predict('rank_svm', mod_out_fname)
    # runner.test(gen_rl_fname)
    # runner.test('rank_svm', 'gen_c-128_e-0.01000')
    # runner.transfer('rank_svm', 'gen_c-64_e-0.0100_t-0')
    # runner.transfer('rank_net', 'gen_node-50_epoch-10_lr-0.00007')
    # runner.ens_par_tun('/home/ffl/nus/MM/cur_trans/exp/ranksvm/gen_c-64_e-0.0100_t-0_sorted.csv',
    #                    '/home/ffl/nus/MM/cur_trans/exp/ranknet/gen_node-50_epoch-10_lr-0.00007_sorted.csv')
    runner.transfer('rank_net',
                    '/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/direct_tune/ranknet/gen_node-50_epoch-10_lr-0.00005')
    runner.transfer('rank_svm',
                    '/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/direct_tune/ranksvm/gen_c-128_e-0.01000')
    runner.ens_par_tun('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/direct_tune/ranksvm/gen_c-128_e-0.01000_sorted.csv',
                       '/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/direct_tune/ranknet/gen_node-50_epoch-10_lr-0.00005_sorted.csv')
    # runner.rank_svm_par_tun()
    # runner.rank_net_par_tun()


    # exp_province()