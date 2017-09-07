# -*- coding: utf-8 -*-
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from util import handle_locations
# handle_locations('/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/locations')
def write_pro_per_csv():
    locations = {}
    with open('/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/locations.json') as fin:
        locations = json.load(fin)
    # with open('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/ent_lin_gt/dir_tun_pro/ranksvm/per.log') as fin:
    with open('/home/ffl/nus/MM/cur_trans/exp/entrance_line_prediction/pow_ent_lin/epsilon_svr/per.log') as fin:
        performances = fin.readlines()
        max_per = 10e5
        for ind, line in enumerate(performances):
            tokens = line.split(':')
            if not len(tokens) == 3:
                print 'shit'
                exit()
            year_location_batch = tokens[0].strip('./').split('/')[0]
            ylb_tokens = year_location_batch.split('_')
            location, batch = locations[unicode(ylb_tokens[1])], ylb_tokens[2]
            if batch == '1' or batch == '11' or batch == '2':
                batch = '文史'
            else:
                batch = '理工'
            if float(tokens[2]) < max_per:
                print 'better: ', location.encode('utf-8'), batch
                max_per = float(tokens[2])
            if ind % 2 == 0:
                print '%s,%s,%s' % (location.encode('utf-8'), batch, '{:.4f}'.format(float(tokens[2])))
            else:
                print ',%s,%s' % (batch, '{:.4f}'.format(float(tokens[2])))
        print max_per
write_pro_per_csv()