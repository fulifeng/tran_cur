import copy
import evaluator
import json
import math
import numpy as np
import operator
import os
import scipy.stats as sps
from util import read_ranking_list
from util_expand import initialize_ranks


class generater:
    def __init__(self, work_path, gt_fname, rl_path, rl_names, sel_uni_fname, top_uni_fname):
        self.work_path = work_path
        self.gt_fname = gt_fname
        self.rl_path = rl_path
        self.rl_names = rl_names
        self.top_uni_fname = top_uni_fname
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

        # read ranking lists
        self.rank_lists, self.rank_dicts = initialize_ranks(rl_path, rl_names)

    def gen_fea_cor(self):
        feature_fname_fname = '/home/ffl/nus/MM/cur_trans/data/prepared/ground_truth/feature_fnames_ori.json'
        feature_fnames = {}
        with open(feature_fname_fname, 'r') as fin:
            feature_fnames = json.load(fin)

        in_path = '/home/ffl/nus/MM/complementary/chinese_university_ranking/experiment/ori_feature/'
        out_path = '/home/ffl/nus/MM/cur_trans/data/prepared/ori_feature/'
        fea_mat = []
        fea_names = []
        is_first = True
        for perspective in feature_fnames.itervalues():
            for types in perspective.itervalues():
                for fname in types:
                    print 'transferring feature from:', os.path.join(in_path, fname)
                    data = np.genfromtxt(os.path.join(in_path, fname), delimiter=',', dtype=str)
                    print 'read data with shape:', data.shape
                    if data[0][0] == 'university':
                        for i in range(1, data.shape[1]):
                            fea_names.append(fname.replace('.csv', '').replace('feature_', '') + '_' + data[0][i])
                    else:
                        for i in range(1, data.shape[1]):
                            fea_names.append(fname.replace('.csv', '').replace('feature_', '') + '_f' + str(i))
                    ofname = os.path.join(out_path, fname)
                    print 'reading feature from:', fname
                    data = np.genfromtxt(os.path.join(ofname), delimiter=',', dtype=float)
                    print 'read data with shape:', data.shape
                    if is_first:
                        is_first = False
                        fea_mat = copy.copy(data)
                    else:
                        fea_mat = np.concatenate((fea_mat, data), axis=1)
                    print 'current feature with shape:', fea_mat.shape
                    print '-----------------------------------------------'
        assert len(fea_names) == fea_mat.shape[1], 'length mismatch'

        # calculate correlation
        with open('feaana.table', 'w') as fout:
            for find, fea_name in enumerate(fea_names):
                lines = ['<tr><td>%s</td>' % fea_name]
                for rank_dict in self.rank_dicts:
                    y_pre = []
                    y_tru = []
                    for uind, uname in enumerate(self.sel_unis):
                        if uname in rank_dict.keys() and math.fabs(fea_mat[uind][find] + 1) > 10e-8:
                            y_pre.append(fea_mat[uind][find])
                            y_tru.append(rank_dict[uname])
                    tau, tpvalue = sps.kendalltau(y_pre, y_tru)
                    ftau = '{:.3f}'.format(-1 * tau)
                    ftp = '%.2E' % tpvalue
                    lines.append('<td>%s</td><td>%s</td>' % (ftau, ftp))
                lines.append('</tr>\n')
                fout.writelines(lines)

    def gen_init_rank(self):
        data = np.genfromtxt(self.top_uni_fname, dtype=str, delimiter=',')
        print 'data shape:', data.shape
        with open('inirank.table', 'w') as fout:
            for u_kv in data:
                fout.write('<tr>\n')
                fout.write('<td>%s</td>\n' % (u_kv[0]))
                fout.write('<td>%s</td>\n' % (u_kv[1]))
                for r_dict in self.rank_dicts:
                    if u_kv[0] in r_dict.keys():
                        fout.write('<td>%d</td>\n' % (r_dict[u_kv[0]]))
                    else:
                        fout.write('<td>-</td>\n')
                fout.write('</tr>\n')

    def gen_ranks(self):
        names = ['Rank_SVM'
                 ]
        rfnames = ['/home/ffl/nus/MM/cur_trans/exp/ranksvm/gen_c-64_e-0.0100_t-0_sorted.csv'
                   ]
        rank_lists = []
        for rfn in rfnames:
            rank_lists.append(np.genfromtxt(rfn, delimiter=',', dtype=str))
        for rl in rank_lists:
            assert  len(rl) == len(rank_lists[0]), 'length mismathc'
        with open('ranks.table', 'w') as fout:
            for method in names:
                fout.write('<tr><td colspan="3">%s</td></tr>\n' % method)
            fout.write('<tr>')
            for method in names:
                fout.write('<td>University</td><td>Score</td><td>Rank</td></tr>\n')
            for ind in range(len(rank_lists[0])):
                fout.write('<tr>')
                for rl in rank_lists:
                    fout.write('<td>%s</td><td>%s</td><td>%d</td>' % (rl[ind][0], rl[ind][1], ind + 1))
                fout.write('</tr>\n')
            print 'write finished'

    def gen_pw_gt(self):
        unis = np.genfromtxt(os.path.join('/home/ffl/nus/MM/cur_trans/data/prepared/', 'university-selected_merged_list.csv'),
                             dtype=str, delimiter=',')
        print 'data shape:', unis.shape
        data_dict = read_ranking_list(os.path.join('/home/ffl/nus/MM/cur_trans/data/prepared/', 'university-selected_merged_list.csv'), int)

        # read ground truth pair
        gt_pairs = []
        data = np.genfromtxt(self.gt_fname, delimiter=',', dtype=str)
        print 'ground truth pair read in with shape:', data.shape
        positive_pair = 0
        negative_pair = 0
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

        # read template
        fin = open('/home/ffl/nus/MM/cur_trans/code/tcurweb/flaskr/flaskr/templates/uxxx.html')
        lines = fin.readlines()
        tags_before = []
        tags_after = []
        is_before = True
        for line in lines:
            if is_before:
                tags_before.append(line)
            else:
                tags_after.append(line)
            if line.__contains__('<!---->'):
                is_before = False

        fuout = open('gt_func.py', 'w')
        with open('gt.table', 'w') as fout:
            # currently, no unseen universities in traditional ranking systems
            ind = 0
            for u_kv in unis:
                bef = 0
                beh = 0
                unk = 0
                uni_bef = []
                uni_beh = []
                print u_kv[0]
                for gtp in gt_pairs:
                    if gtp[0] == u_kv[0]:
                        beh += 1
                        uni_beh.append(gtp[1])
                    if gtp[1] == u_kv[0]:
                        bef += 1
                        uni_bef.append(gtp[0])
                unk = len(self.sel_unis) - bef - beh
                # write html table
                fout.write('<tr>\n')
                fout.write('<td><a href="{{url_for(\'u%d\')}}">%s</td>\n' % (ind, u_kv[0]))
                fout.write('<td>%d</td>\n' % (bef))
                fout.write('<td>%d</td>\n' % (beh))
                fout.write('<td>%d</td>\n' % (unk))
                fout.write('</tr>\n')

                # write handle function
                fuout.write('@app.route(\'/u%d/\',methods = [\'GET\'])\ndef u%d():\n    return render_template(\'u%d.html\')\n\n' % (ind, ind, ind))

                # write university page
                fpout = open('/home/ffl/nus/MM/cur_trans/code/tcurweb/flaskr/flaskr/templates/u%d.html' % ind, 'w')
                fpout.writelines(tags_before)
                fpout.writelines(self.gen_pw_gt_sin(ind, uni_bef, uni_beh))
                fpout.writelines(tags_after)
                ind += 1

    def gen_pw_gt_sin(self, ind, uni_before, uni_behind):
        lines = ['<table>']
        # write universities before
        cols = 4
        for i in range(len(uni_before) / cols + 1):
            lines.append('<tr>')
            for j in range(cols):
                if i * cols + j < len(uni_before):
                    lines.append('<td>%s</td>' % uni_before[i * cols + j])
                else:
                    lines.append('<td>-</td>')
            lines.append('</tr>')
        lines.append('</table><h2>Universities Behind</h2><table>')
        for i in range(len(uni_behind) / cols + 1):
            lines.append('<tr>')
            for j in range(cols):
                if i * cols + j < len(uni_behind):
                    lines.append('<td>%s</td>' % uni_behind[i * cols + j])
                else:
                    lines.append('<td>-</td>')
            lines.append('</tr>')
        # write universities behind
        lines.append('</table>')
        return lines

if __name__ == '__main__':
    gen = generater(
        '/home/ffl/nus/MM/cur_trans/exp/gen_htmls/',
        '/home/ffl/nus/MM/cur_trans/data/prepared/ground_truth/pair_wise_top_university_all.csv',
        '/home/ffl/nus/MM/cur_trans/data/prepared/rank_lists/',  # ranking list path
        ['cuaa_2016', 'wsl_2017', 'rank_2017', 'qs_2016', 'usn_2017'],  # ranking names
        '/home/ffl/nus/MM/cur_trans/data/prepared/candidate_universites.csv',  # selected university file
        '/home/ffl/nus/MM/cur_trans/data/prepared/ground_truth/top_university.csv'  # top university file
    )
    # gen.gen_init_rank()
    # gen.gen_pw_gt()
    # gen.gen_fea_cor()
    gen.gen_ranks()
