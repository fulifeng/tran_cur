import copy
import json
import numpy as np
import operator
import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import evaluator
from util import read_ranking_list


'''
    Functions are similar as experiment, this is for point-wise experiment
'''
class experiment:
    def __init__(self, work_path, tool_path, train_all_fname, test_fname, sel_uni_fname):
        self.work_path = work_path
        self.tool_path = tool_path
        self.train_all_fname = train_all_fname
        self.test_fname = test_fname
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

        # # initialize evaluator
        # self.evaluator = evaluator.evaluator(self.rl_path, self.rl_names, self.gt_fname)

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

    def epsilon_svr_par_tun(self):
        # parameters
        c = [128, 256, 512, 1024, 2048, 4096, 8192]
        g = [0.0001, 0.0005, 0.001, 0.00195, 0.0039, 0.0078, 0.01562, 0.03125, 0.0625]
        p = [0.00005, 0.0001, 0.00039, 0.00078, 0.00156, 0.00312, 0.00625, 0.0125, 0.025]
        e = [0.001]
        paras_com = []
        temp_paras = {'s':3}
        for cv in c:
            temp_paras['c'] = cv
            for ev in e:
                temp_paras['e'] = ev
                # paras_com.append(copy.copy(temp_paras))
                for gv in g:
                    temp_paras['g'] = gv
                    for pv in p:
                        temp_paras['p'] = pv
                        paras_com.append(copy.copy(temp_paras))
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

        best_mse = 10e8
        best_para = ''
        flout = open('epsilon_svr_par_tun.log', 'w')
        for i, paras in enumerate(paras_com):
            mod_out_fname = 'mod'
            for p_kv in paras.iteritems():
                if type(p_kv[1]) == float:
                    mod_out_fname = mod_out_fname + '_' + p_kv[0] + '-' + '{:.5f}'.format(p_kv[1])
                else:
                    mod_out_fname = mod_out_fname + '_' + p_kv[0] + '-' + str(p_kv[1])

            # train model with tr_all
            train_com = os.path.join(self.tool_path, 'svm-train')
            for p_kv in paras.iteritems():
                train_com = train_com + ' -' + p_kv[0] + ' ' + str(p_kv[1])
            if i == 0:
                train_com = train_com + ' ' + self.train_all_fname + ' ' + mod_out_fname
            else:
                train_com = train_com + ' ' + self.train_all_fname + ' ' + mod_out_fname
            print 'train command:', train_com
            flout.write('train command:' + train_com + '\n')
            os.system(train_com)

            # test model
            test_com = os.path.join(self.tool_path, 'svm-predict') + ' ' + self.test_fname + ' ' + mod_out_fname + ' ' + \
                       mod_out_fname.replace('mod', 'out')
            print 'test command:', test_com
            flout.write('test command:' + test_com + '\n')
            # os.system(test_com)
            result = subprocess.check_output(test_com, shell=True)
            print result
            print mod_out_fname
            flout.write(result + '\n' + mod_out_fname + '\n')
            tokens = result.split(' ')
            per = float(tokens[4])
            if per < best_mse:
                best_mse = per
                best_para = copy.copy(mod_out_fname)
                print '!!!better!!!'
                flout.write('!!!better!!!\n')
                # print 'better performance:', best_mse
                # print 'better parameter:', mod_out_fname
            print '-----------------------------------------------'
            flout.write('-----------------------------------------------\n')
        print 'best performance:', best_mse
        flout.write('best performance:' + str(best_mse) + '\n')
        print 'best parameter:', best_para
        flout.write('best parameter:' + best_para + '\n')

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

def exp_province():
    titles = ['2015_51_2', '2015_51_6', '2015_50_1', '2015_50_5', '2015_61_1', '2015_61_5', '2015_62_1', '2015_62_5', '2015_63_1',
              '2015_63_5', '2015_64_1', '2015_64_5', '2015_53_1', '2015_53_5', '2015_52_1', '2015_52_5', '2015_21_1', '2015_21_5',
              '2015_22_1', '2015_22_5', '2015_23_1', '2015_23_5', '2015_46_1', '2015_46_5', '2015_44_1', '2015_44_5', '2015_45_1',
              '2015_45_5', '2015_42_1', '2015_42_5', '2015_43_1', '2015_43_5', '2015_41_1', '2015_41_5', '2015_11_11', '2015_11_15',
              '2015_13_1', '2015_13_5', '2015_12_1', '2015_12_5', '2015_15_1', '2015_15_5', '2015_14_1', '2015_14_5', '2015_33_1',
              '2015_33_5', '2015_32_1', '2015_32_5', '2015_31_1', '2015_31_5', '2015_37_1', '2015_37_5', '2015_36_1', '2015_36_5',
              '2015_35_1', '2015_35_5', '2015_34_11', '2015_34_15', '2015_54_1', '2015_54_5', '2015_65_11', '2015_65_15']
    for i, title in enumerate(titles):
        if os.path.exists(os.path.join('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/pow_ent_lin/epsilon_svr', title)):
            continue
        print 'working on:', title
        os.mkdir(os.path.join('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/pow_ent_lin/epsilon_svr', title))
        os.chdir(os.path.join('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/pow_ent_lin/epsilon_svr', title))
        runner = experiment(os.path.join('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/pow_ent_lin/epsilon_svr', title),
                                # work path
                            '/home/ffl/nus/MM/cur_trans/tool/libsvm-3.22',  # tool path rank SVM
                            os.path.join('/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/pow_ground_truth', 'tr_all_' + title + '.csv'),
                                # all training data
                            os.path.join('/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/pow_ground_truth', 'te_' + title + '.csv'),
                                # testing data
                            '/home/ffl/nus/MM/cur_trans/data/entrance_line_prediction/candidate_universites.csv'  # selected university file
                            )
        runner.epsilon_svr_par_tun()
        # if i > 0:
        #     break

if __name__ == '__main__':
    exp_province()