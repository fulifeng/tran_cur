import numpy as np
import scipy.stats as sps
from util_expand import initialize_ranks


class evaluator:
    def __init__(self, rl_path, rank_names, gt_fname):
        self.rank_names = rank_names
        # read ground truth pair
        self.gt_pairs = []
        data = np.genfromtxt(gt_fname, delimiter=',', dtype=str)
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
            self.gt_pairs.append(gt_pair)

        # read ranking lists
        self.rank_lists, self.rank_dicts = initialize_ranks(rl_path, rank_names)

    '''
        Given a generated ranking list, compare it with ground truth and
        calculate accuracy and Spearman's rank correlation

        gen_rank: generated ranking list

        return: accuracy and Spearman's rank correlation
    '''
    def evaluate(self, gen_rank):
        # calculate accuracy #correctly ranked pairs / #pairs occur in the generated rank as well as in ground truth pairs
        occured = 0
        correct = 0
        for gtp in self.gt_pairs:
            if gtp[0] in gen_rank.keys() and gtp[1] in gen_rank.keys():
                occured += 1
                if gen_rank[gtp[0]] > gen_rank[gtp[1]]:
                    correct += 1
        if occured == 0:
            acc = 0
        else:
            acc = float(correct) / occured
        print '%d correct among %d occured' % (correct, occured)

        # calculate kendall tau and spearman's rank list by list
        cors = {}
        for ind, rank_dict in enumerate(self.rank_dicts):
            y_pre = []
            y_er = []
            for uni_kv in gen_rank.iteritems():
                if uni_kv[0] in rank_dict.keys():
                    # !!!!!!!!!!!!!!!!!
                    y_pre.append( -1 * uni_kv[1])
                    y_er.append(rank_dict[uni_kv[0]])
            print '#universities tested:', len(y_pre)
            tau, tpvalue = sps.kendalltau(y_pre, y_er)
            rho, rpvalue = sps.spearmanr(y_pre, y_er)
            cors[self.rank_names[ind]] = [[tau, tpvalue], [rho, rpvalue]]

        return acc, cors


