# -*- coding: utf-8 -*-
import argparse
import copy
from decimal import Decimal
import json
import numpy as np
import operator
try:
    set
except NameError:
    from sets import Set as set
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from util import read_kldm_uni_values, write_unicode_matrix, \
    write_matrix_unicode_header, clean_uni_name


'''
    Extract universities' collage entrance lines, only a function
    'extract_uni_line_fir' is used.
'''
class gaokao_feature:
    def __init__(self, uni_line_fname, pro_line_fname, kldm_value_fname):
        self.uni_line_fname = uni_line_fname
        self.pro_line_fname = pro_line_fname
        self.kldm_value_fname = kldm_value_fname
        # read university lines
        self.university_lines = {}
        # keys in the json data are in unicode format, besides, university names are not cleaned,
        # the names in the candidate university list are in utf-8, to handle this encoding mismatch
        # we transfer the data read in first.
        with open(self.uni_line_fname) as uni_file:
            data = json.load(uni_file)
            for year_kv in data.iteritems():
                if year_kv[0] not in self.university_lines.keys():
                    self.university_lines[year_kv[0]] = {}
                for uni_kv in year_kv[1].iteritems():
                    self.university_lines[year_kv[0]][clean_uni_name(uni_kv[0].encode('utf-8'))] = uni_kv[1]
        print 'univerisity lines years:', self.university_lines.keys(), '#years:', len(self.university_lines)
        if len(self.university_lines) < 1:
            self.init_flag = False
        # read province lines
        self.province_lines = {}
        with open(self.pro_line_fname) as pro_file:
            self.province_lines = json.load(pro_file)
        print 'province lines years:', self.province_lines.keys(), '#years:', len(self.university_lines)
        if len(self.province_lines) < 1:
            self.init_flag = False
        # selected years to extract features
        self.years = ['2013'.encode('utf-8'), '2014'.encode('utf-8'),
                      '2015'.encode('utf-8')]
        # self.years = ['2013'.encode('utf-8'), '2014'.encode('utf-8')]
        # read kldm values
        self.kldm_values = read_kldm_uni_values(self.kldm_value_fname)
        print '#years with kldm values:', len(self.kldm_values)
        # initialize the union university name set
        self.uni_names = set()
        self.init_flag = self.init_union_uni_name()
        print '#union university names:', len(self.uni_names)

    # here the province lines are not subtracted.
    def extract_aver_uni_line(self, normalize=True):
        # get columns
        columns = 1
        # for provinces in self.kldm_values.itervalues():
        max_year = -1
        for year in self.years:
            temp_columns = 1
            provinces = self.kldm_values[year]
            for kldms in provinces.itervalues():
                temp_columns += len(kldms)
            if temp_columns > columns:
                columns = temp_columns
                max_year = year
        print '#columns:', columns
        # lines_table = np.ones((len(self.uni_names) + 1, columns), dtype=str)
        # print 'lines table\'s shape:', lines_table.shape
        lines_table = [['university']]
        # generate titles
        j = 1
        provinces = self.kldm_values[max_year]
        for ssdm_kv in provinces.iteritems():
            for kldm in ssdm_kv[1]:
                # print self.gen_title(year, ssdm_kv[0], kldm)
                lines_table[0].append(ssdm_kv[0] + '_' + kldm)
                j += 1
        if not (j == columns and columns == len(lines_table[0])):
            print 'unexpected j:', j, 'and columns:', columns
            return
        if normalize:
            max_pro_lines = copy.copy(self.province_lines)
            for year_kv in max_pro_lines.iteritems():
                for ssdm_kv in year_kv[1].iteritems():
                    for kldm_kv in ssdm_kv[1].iteritems():
                        for level_kv in kldm_kv[1].iteritems():
                            max_pro_lines[year_kv[0]][ssdm_kv[0]][
                                kldm_kv[0]][level_kv[0]] = max(level_kv[1])
        else:
            max_pro_lines = None
        no_lines = 0
        with_lines = 0
        max_uni_line = -1
        # level_str = '2'.encode('utf-8')
        normalize_batch = '2'.encode('utf-8')
        for i, name in enumerate(self.uni_names):
            lines_table.append([unicode(name.decode('utf-8'))])
            j = 1
            # provinces = self.kldm_values[year]
            for ssdm_kv in provinces.iteritems():
                for kldm in ssdm_kv[1]:
                    uni_lines = []
                    for year in self.years:
                        if name not in self.university_lines[year].keys():
                            # print 'no line data of university:', name, 'in year:', year
                            no_lines += 1
                        elif ssdm_kv[0] in self.university_lines[year][name].\
                                keys() and kldm in self.university_lines[
                                    year][name][ssdm_kv[0]].keys():
                            lines = self.university_lines[year][name][
                                ssdm_kv[0]][kldm]
                            lines_sel_bat = []
                            for batch in lines.keys():
                                batch_type = self._get_bat_typ(batch)
                                if batch_type == 0 or batch_type == 2:
                                    lines_sel_bat.append(lines[batch])
                            if len(lines_sel_bat) == 0:
                                continue
                            else:
                                if max(lines_sel_bat) == 999:
                                    print 'shit'
                                    exit()
                                average_university_line_batches = \
                                    sum(lines_sel_bat) / len(lines_sel_bat)
                                # subtract province line
                                if normalize:
                                    average_university_line_batches -= \
                                        max_pro_lines[year][ssdm_kv[0]][
                                            kldm][normalize_batch]
                                uni_lines.append(
                                    average_university_line_batches)
                                if average_university_line_batches < -1000:
                                    print max_pro_lines[year][ssdm_kv[0]][
                                            kldm][normalize_batch]
                                    exit()
                            # update maximum line for the final report
                            max_line = max(lines_sel_bat)
                            with_lines += 1
                            if max_line > max_uni_line:
                                max_uni_line = max_line
                        else:
                            no_lines += 1
                    if len(uni_lines) == 0:
                        if normalize:
                            lines_table[i + 1].append('-10000')
                        else:
                            lines_table[i + 1].append('-1')
                    else:
                        lines_table[i + 1].append(str(float(sum(uni_lines)) /
                                                      len(uni_lines)))
                    j += 1
            if not (j == columns and columns == len(lines_table[i + 1])):
                print 'unexpected j:', j, ', columns:', columns, ', and list length:', len(lines_table[i + 1]), 'while writing university:', name
                return
        print 'lines table\'s shape:', len(lines_table), len(lines_table[0])
        print '#with lines:', with_lines, '#no lines:', no_lines
        print 'maximum university line:', max_uni_line
        if normalize:
            ofname = self.uni_line_fname.replace('.json', '_' + '_'.join(
                self.years) + '_norm_aver.csv')
        else:
            ofname = self.uni_line_fname.replace('.json', '_' + '_'.join(
                self.years) + '_aver.csv')
        print 'output fname:', ofname
        # write_matrix_unicode_header(lines_table, ofname)
        if normalize:
            # further normalize each column of lines_table to 0-1
            lines_table_scaled = np.ones((len(lines_table),
                                          len(lines_table[0])), dtype=float)
            lines_table_scaled *= -10000
            for i in range(1, len(lines_table)):
                for j in range(1, len(lines_table[0])):
                    lines_table_scaled[i][j] = float(lines_table[i][j])
            for j in range(1, len(lines_table[0])):
                column_max_line = np.max(lines_table_scaled[:, j])
                print column_max_line
                for i in range(len(lines_table)):
                    if abs(lines_table_scaled[i][j] + 1e4) > 1e-8:
                        lines_table_scaled[i][j] = lines_table_scaled[i][j] \
                                                   / column_max_line
                        lines_table[i][j] = '{:.6f}'.format(
                            lines_table_scaled[i][j])
        write_unicode_matrix(lines_table, ofname)

        # write out average university lines across provinces
        uni_ranks = {}
        for i in range(1, len(lines_table)):
            lines = []
            for j in range(1, len(lines_table[i])):
                # if j % 2 == 0:
                #     continue
                if normalize:
                    if not lines_table[i][j] == '-10000':
                        lines.append(float(lines_table[i][j]))
                else:
                    if not lines_table[i][j] == '-1':
                        lines.append((lines_table[i][j]))
            if len(lines) < 32:
                print '#missing entries:', columns - 1 - len(lines), \
                lines_table[i][0]
            if len(lines) == 0:
                uni_ranks[lines_table[i][0]] = 12345678.12345678
            else:
                uni_ranks[lines_table[i][0]] = sum(lines) / len(lines)
        print uni_ranks['首都医科大学'.decode('utf-8')]
        sorted_uni_aver = sorted(uni_ranks.items(), key=operator.itemgetter(1),
                                 reverse=True)
        if normalize:
            ofname = self.uni_line_fname.replace('.json',
                                                 '_norm_aver_aver.csv')
        else:
            ofname = self.uni_line_fname.replace('.json', '_aver_aver.csv')
        print 'output fname:', ofname
        write_matrix_unicode_header(sorted_uni_aver, ofname)

    # it just convert the university_lines.json in JSON format to csv format,
    # if multiple batches exist at the same time, they will be concatenated by
    # '-'
    def extract_pure_uni_line(self):
        # get columns
        columns = 1
        # for provinces in self.kldm_values.itervalues():
        for year in self.years:
            provinces = self.kldm_values[year]
            for kldms in provinces.itervalues():
                columns += len(kldms)
        # print '#columns:', columns
        # lines_table = np.ones((len(self.uni_names) + 1, columns), dtype=str)
        # print 'lines table\'s shape:', lines_table.shape
        lines_table = [['name'.encode('utf-8')]]
        # generate titles
        j = 1
        for year in self.years:
            provinces = self.kldm_values[year]
            for ssdm_kv in provinces.iteritems():
                for kldm in ssdm_kv[1]:
                    # print self.gen_title(year, ssdm_kv[0], kldm)
                    lines_table[0].append(self.gen_title(year, ssdm_kv[0], kldm))
                    j += 1
        if not (j == columns and columns == len(lines_table[0])):
            print 'unexpected j:', j, 'and columns:', columns
            return
        no_lines = 0
        with_lines = 0
        for i, name in enumerate(self.uni_names):
            lines_table.append([name])
            j = 1
            for year in self.years:
                provinces = self.kldm_values[year]
                for ssdm_kv in provinces.iteritems():
                    for kldm in ssdm_kv[1]:
                        if name not in self.university_lines[year].keys():
                            print 'no line data of university:', name, 'in year:', year
                            lines_table[i + 1].append('-1')
                            j += 1
                            no_lines += 1
                        elif ssdm_kv[0] in self.university_lines[year][name].keys() and kldm in self.university_lines[year][name][ssdm_kv[0]].keys():
                            lines = self.university_lines[year][name][ssdm_kv[0]][kldm]
                            lines_table[i + 1].append(self.gen_line(lines))
                            j += 1
                            with_lines += 1
                        else:
                            lines_table[i + 1].append('-1')
                            j += 1
                            no_lines += 1
            if not (j == columns and columns == len(lines_table[i])):
                print 'unexpected j:', j, ', columns:', columns, ', and list length:', len(lines_table[i]), 'while writing university:', name
                return
        print 'lines table\'s shape:', len(lines_table), len(lines_table[0])
        print '#with lines:', with_lines, '#no lines:', no_lines
        ofname = self.uni_line_fname.replace('json', 'csv')
        print 'output fname:', ofname
        write_unicode_matrix(lines_table, ofname)
        # odata = np.array(lines_table, dtype=np.unicode)
        # np.savetxt(self.uni_line_fname.replace('json', 'csv'), odata, delimiter=',', fmt='%s')

    # extract university lines, first batch, second batch, and average of
    # first and second batches if they occur at the same time.
    def extract_uni_line_fir(self, year, with_head=True):
        # get columns
        if with_head:
            columns = 1
        else:
            columns = 0
        provinces = self.kldm_values[year]
        for kldms in provinces.itervalues():
            columns += len(kldms)
        print '#columns:', columns
        if with_head:
            lines_table = [['name'.encode('utf-8')]]
            # generate titles
            j = 1
            provinces = self.kldm_values[year]
            for ssdm_kv in provinces.iteritems():
                for kldm in ssdm_kv[1]:
                    # print self.gen_title(year, ssdm_kv[0], kldm)
                    lines_table[0].append(self.gen_title(year, ssdm_kv[0],
                                                         kldm))
                    j += 1
            if not (j == columns and columns == len(lines_table[0])):
                print 'unexpected j:', j, 'and columns:', columns
                return
        else:
            lines_table = [[]]
        no_lines = 0
        with_lines = 0
        for i, name in enumerate(self.uni_names):
            if with_head:
                lines_table.append([unicode(name.decode('utf-8'))])
                j = 1
            else:
                lines_table.append([])
                j = 0
            # for year in self.years:
            provinces = self.kldm_values[year]
            for ssdm_kv in provinces.iteritems():
                for kldm in ssdm_kv[1]:
                    line_value = -1
                    # if name not in self.university_lines[year].keys():
                    #     lines_table[i + 1].append('-1')
                    #     # j += 1
                    #     no_lines += 1
                    # elif ssdm_kv[0] in self.university_lines[year][name].keys() and kldm in self.university_lines[year][name][ssdm_kv[0]].keys():
                    if name in self.university_lines[year].keys() and \
                                    ssdm_kv[0] in self.university_lines[year][name].keys() and \
                                    kldm in self.university_lines[year][name][ssdm_kv[0]].keys():
                        lines = self.university_lines[year][name][ssdm_kv[0]][kldm]
                        bat_cous = len(lines)
                        if bat_cous == 1:
                            bat_type = self._get_bat_typ(lines.keys()[0])
                            if bat_type == 0 or bat_type == 2:
                                line_value = lines.values()[0]
                                # lines_table[i + 1].append(str(lines.values()[0]))
                                # with_lines += 1
                            # else:
                            #     lines_table[i + 1].append('-1')
                            #     no_lines += 1
                        else:
                            # !!!!!!   MENTION that there are 5 lines with number 999 in BENZHUANKETIQIANPI, BENKETIQIANPI, BENKEDISANPI!!!!!!
                            bat_types = []
                            for batch in lines.keys():
                                bat_types.append(self._get_bat_typ(batch))
                            line_scores_sum = 0
                            line_scores_count = 0
                            for k in range(bat_cous):
                                if bat_types[k] == 0 or bat_types[k] == 2:
                                    line_scores_count += 1
                                    line_scores_sum += int(lines[lines.keys()[k]])
                            # if line_scores_count == 0:
                            #     lines_table[i + 1].append('-1')
                            # else:
                            #     lines_table[i + 1].append(str(line_scores_sum / line_scores_count))
                            if not line_scores_count == 0:
                                line_value = line_scores_sum / line_scores_count
                        # j += 1
                    # else:
                    #     # lines_table[i + 1].append('-1')
                    #     line_value = -1
                    #     # j += 1
                    #     no_lines += 1
                    if with_head:
                        lines_table[i + 1].append(str(line_value))
                    else:
                        lines_table[i].append(line_value)
                    j += 1
            if not (j == columns and columns == len(lines_table[i])):
                print 'unexpected j:', j, ', columns:', columns, ', and list length:', len(lines_table[i]), 'while writing university:', name
                return
        print 'lines table\'s shape:', len(lines_table), len(lines_table[0])
        # print '#with lines:', with_lines, '#no lines:', no_lines
        if with_head:
            ofname = self.uni_line_fname.replace('.json', '_' + year + '.csv')
            write_unicode_matrix(lines_table, ofname)
        else:
            ofname = self.uni_line_fname.replace('.json', '_' + year + '_nohead.csv')
            # data = np.array(lines_table, dtype=int)
            print len(lines_table), len(lines_table[0])
            data = np.zeros((len(lines_table), len(lines_table[0])), dtype=int)
            print data.shape
            for i, lines in enumerate(lines_table):
                for j, line in enumerate(lines):
                    data[i, j] = line
            # data = np.array([np.array(xi, dtype=int) for xi in lines_table])
            # print data.shape
            np.savetxt(ofname, np.array(data), delimiter=',', fmt='%d')
        print 'output fname:', ofname
        #

    def extract_uni_line_rank(self):
        # get columns
        columns = 1
        # for provinces in self.kldm_values.itervalues():
        max_year = -1
        for year in self.years:
            temp_columns = 1
            provinces = self.kldm_values[year]
            for kldms in provinces.itervalues():
                temp_columns += len(kldms)
            if temp_columns > columns:
                columns = temp_columns
                max_year = year
        print '#columns:', columns
        # lines_table = np.ones((len(self.uni_names) + 1, columns), dtype=str)
        # print 'lines table\'s shape:', lines_table.shape
        lines_table = [['name']]
        # generate titles
        j = 1
        provinces = self.kldm_values[max_year]
        for ssdm_kv in provinces.iteritems():
            for kldm in ssdm_kv[1]:
                # print self.gen_title(year, ssdm_kv[0], kldm)
                lines_table[0].append(ssdm_kv[0] + '_' + kldm)
                j += 1
        if not (j == columns and columns == len(lines_table[0])):
            print 'unexpected j:', j, 'and columns:', columns
            return
        max_pro_lines = copy.copy(self.province_lines)
        for year_kv in max_pro_lines.iteritems():
            for ssdm_kv in year_kv[1].iteritems():
                for kldm_kv in ssdm_kv[1].iteritems():
                    for level_kv in kldm_kv[1].iteritems():
                        max_pro_lines[year_kv[0]][ssdm_kv[0]][kldm_kv[0]][level_kv[0]] = max(level_kv[1])
        no_lines = 0
        with_lines = 0
        max_uni_line = -1
        level_str = '2'.encode('utf-8')
        for i, name in enumerate(self.uni_names):
            lines_table.append([name])
            j = 1
            # provinces = self.kldm_values[year]
            for ssdm_kv in provinces.iteritems():
                for kldm in ssdm_kv[1]:
                    uni_lines = []
                    for year in self.years:
                        if name not in self.university_lines[year].keys():
                            print 'no line data of university:', name, 'in year:', year
                            no_lines += 1
                        elif ssdm_kv[0] in self.university_lines[year][name].keys() and kldm in self.university_lines[year][name][ssdm_kv[0]]. \
                                keys():
                            max_line = max(self.university_lines[year][name][ssdm_kv[0]][kldm])
                            if not max_line == 999:
                                uni_lines.append(max(self.university_lines[year][name][ssdm_kv[0]][kldm]) -
                                                 max_pro_lines[year][ssdm_kv[0]][kldm][level_str])
                            else:
                                temp_lines = self.university_lines[year][name][ssdm_kv[0]][kldm]
                                max_line = -1
                                for num in temp_lines:
                                    if num > max_line and (not num == 999):
                                        max_line = num
                                uni_lines.append(max_line - max_pro_lines[year][ssdm_kv[0]][kldm][level_str])
                            with_lines += 1
                            if max_line > max_uni_line:
                                max_uni_line = max_line
                        else:
                            no_lines += 1
                    if len(uni_lines) == 0:
                        lines_table[i + 1].append(-10000)
                    else:
                        lines_table[i + 1].append(float(sum(uni_lines)) / len(uni_lines))
                    j += 1
            if not (j == columns and columns == len(lines_table[i + 1])):
                print 'unexpected j:', j, ', columns:', columns, ', and list length:', len(
                    lines_table[i + 1]), 'while writing university:', name
                return
        print 'lines table\'s shape:', len(lines_table), len(lines_table[0])
        print '#with lines:', with_lines, '#no lines:', no_lines
        uni_ranks = {}
        missing_list = []
        for i in range(1, len(lines_table)):
            lines = []
            for j in range(1, len(lines_table[i])):
                if not lines_table[i][j] == -10000:
                    lines.append(lines_table[i][j])
            if len(lines) < 32:
                print '#missing entries:', columns - 1 - len(lines), lines_table[i][0]
            missing_list.append([lines_table[i][0], str(columns - 1 - len(lines))])
            uni_ranks[lines_table[i][0]] = sum(lines) / len(lines)
        print uni_ranks['首都医科大学'.decode('utf-8')]
        mfname = self.uni_line_fname.replace('.json', '_aver_missing.csv')
        write_unicode_matrix(missing_list, mfname)
        sorted_uni_aver = sorted(uni_ranks.items(), key=operator.itemgetter(1), reverse=True)
        ofname = self.uni_line_fname.replace('.json', '_aver_aver.csv')
        print 'output fname:', ofname
        write_matrix_unicode_header(sorted_uni_aver, ofname)
        rfname = self.uni_line_fname.replace('.json', '_aver_rank.csv')
        # transform the sore to rank, it means higher score higher rank (small rank number)
        sorted_uni_rank = []
        for i in range(len(sorted_uni_aver)):
            sorted_uni_rank.append([sorted_uni_aver[i][0], sorted_uni_aver[0][1] - sorted_uni_aver[i][1]])
        write_matrix_unicode_header(sorted_uni_rank, rfname)
        # write_unicode_matrix(lines_table, ofname)

    def init_union_uni_name(self):
        # check whether we have collected the data of the selected year
        missing_data = False
        for year in self.years:
            if year not in self.university_lines.keys():
                print 'we don\'t have university line data of year:', year
                missing_data = True
            if year not in self.province_lines.keys():
                print 'we don\'t have province line data of year:', year
                missing_data = True
        for year in self.years:
            lines = self.university_lines[year]
            for uni_name in lines.iterkeys():
                self.uni_names.add(uni_name)
        return not missing_data

    def gen_title(self, year, ssdm, kldm):
        # return (year * 100 + ssdm) * 10 + kldm
        # return str(year) + '_' + str(ssdm) + '_' + str(kldm)
        return year + '_' + ssdm + '_' + kldm

    def gen_line(self, lines):
        line_str = str(lines[0])
        for i in range(1, len(lines)):
            line_str = line_str + '-' + str(lines[i])
        return line_str

    def observe_dat_mis(self, candidate_fname):
        # read candidate universities
        data = np.genfromtxt(candidate_fname, dtype=str, delimiter=',')
        print '#candidate universities:', len(data)
        can_unis = set(data)

        # overall
        no_lines = 0
        with_lines = 0
        # liberal arts
        no_lines_lib = 0
        with_lines_lib = 0
        # science
        no_lines_sci = 0
        with_lines_sci = 0
        years = copy.copy(self.years)
        years = [years[2]]
        print '-'.join(years)
        for name in can_unis:
            for year in years:
                provinces = self.kldm_values[year]
                for ssdm_kv in provinces.iteritems():
                    for kl_ind, kldm in enumerate(ssdm_kv[1]):
                        if name not in self.university_lines[year].keys():
                            # print 'no line data of university:', name, 'in year:', year
                            no_lines += 1
                            if kl_ind == 0:
                                no_lines_lib += 1
                            elif kl_ind == 1:
                                no_lines_sci += 1
                            else:
                                print 'shit'
                                exit()
                        elif ssdm_kv[0] in self.university_lines[year][name].keys() and kldm in self.university_lines[year][name][ssdm_kv[0]].keys():
                            lines = self.university_lines[year][name][ssdm_kv[0]][kldm]
                            # lines_table[i + 1].append(self.gen_line(lines))
                            with_lines += 1
                            if kl_ind == 0:
                                with_lines_lib += 1
                            elif kl_ind == 1:
                                with_lines_sci += 1
                            else:
                                print 'shit'
                                exit()
                        else:
                            no_lines += 1
                            if kl_ind == 0:
                                no_lines_lib += 1
                            elif kl_ind == 1:
                                no_lines_sci += 1
                            else:
                                print 'shit'
                                exit()
        print '#with lines:', with_lines, '#no lines:', no_lines, 'ratio:', no_lines / float(with_lines + no_lines)
        print '#liberal arts with lines: %d, #without lines: %d, ratio: %f' % (with_lines_lib, no_lines_lib,
                                                                               no_lines_lib / float(with_lines_lib + no_lines_lib))
        print '#science with lines: %d, #without lines: %d, ratio: %f' % (with_lines_sci, no_lines_sci,
                                                                          no_lines_sci / float(with_lines_sci + no_lines_sci))

    # observe batch counts distribution
    def observe_mul_bat_cou(self, candidate_fname):
        # read candidate universities
        data = np.genfromtxt(candidate_fname, dtype=str, delimiter=',')
        print '#candidate universities:', len(data)
        can_unis = set(data)

        # overall
        no_lines = 0
        with_lines = 0
        lib_bat_cous = [0] * 10
        sci_bat_cous = [0] * 10
        years = copy.copy(self.years)
        years = [years[2]]
        print '-'.join(years)
        for name in can_unis:
            for year in years:
                provinces = self.kldm_values[year]
                for ssdm_kv in provinces.iteritems():
                    for kl_ind, kldm in enumerate(ssdm_kv[1]):
                        if name not in self.university_lines[year].keys():
                            # print 'no line data of university:', name, 'in year:', year
                            no_lines += 1
                        elif ssdm_kv[0] in self.university_lines[year][name].keys() and kldm in self.university_lines[year][name][ssdm_kv[0]].keys():
                            lines = self.university_lines[year][name][ssdm_kv[0]][kldm]
                            # lines_table[i + 1].append(self.gen_line(lines))
                            with_lines += 1
                            if kl_ind == 0:
                                lib_bat_cous[len(lines)] += 1
                            elif kl_ind == 1:
                                sci_bat_cous[len(lines)] += 1
                            else:
                                print 'shit'
                                exit()
                        else:
                            no_lines += 1
        print '#with lines:', with_lines, '#no lines:', no_lines, 'ratio:', no_lines / float(with_lines + no_lines)
        print 'liberal batch counts:'
        print ' | '.join(map(str, lib_bat_cous))
        print 'science batch counts'
        print ' | '.join(map(str, sci_bat_cous))

    # observe batch cooccurrence among pairs (一批, 本科提前批, 二批, others)
    def observe_mul_bat_coo(self, candidate_fname):
        # read candidate universities
        data = np.genfromtxt(candidate_fname, dtype=str, delimiter=',')
        print '#candidate universities:', len(data)
        can_unis = set(data)

        # overall
        coo_mat = np.zeros([4, 4], dtype=int)
        years = copy.copy(self.years)
        years = [years[2]]
        print '-'.join(years)
        for name in can_unis:
            for year in years:
                provinces = self.kldm_values[year]
                for ssdm_kv in provinces.iteritems():
                    for kl_ind, kldm in enumerate(ssdm_kv[1]):
                        if name not in self.university_lines[year].keys():
                            pass
                        elif ssdm_kv[0] in self.university_lines[year][name].keys() and kldm in self.university_lines[year][name][ssdm_kv[0]].keys():
                            lines = self.university_lines[year][name][ssdm_kv[0]][kldm]
                            # lines_table[i + 1].append(self.gen_line(lines))
                            if len(lines) == 1:
                                bat_type = self._get_bat_typ(lines.keys()[0])
                                coo_mat[bat_type][bat_type] += 1
                            else:
                                bat_types = []
                                bat_cous = len(lines)
                                for batch in lines.keys():
                                    bat_types.append(self._get_bat_typ(batch))
                                bat_types.sort()
                                for i in range(bat_cous):
                                    for j in range(i + 1, bat_cous):
                                        coo_mat[bat_types[i]][bat_types[j]] += 1
        print 'batch cooccurrence counts:'
        for line in coo_mat:
            print ' | '.join(map(str, line))

    # 0, 1, 2, 3 denotes (一批, 本科提前批, 二批, others), respectively
    def _get_bat_typ(self, batch_name):
        if u'本科' in batch_name and u'一批' in batch_name:
            return 0
        if u'本科' in batch_name and u'提前' in batch_name:
            return 1
        if u'本科' in batch_name and u'二批' in batch_name:
            return 2
        # if u'本科' in batch_name and u'一批' in batch_name:
        return 3

if __name__ == '__main__':
    desc = "extract gaokao features from the university_lines.json and province_lines.json"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-pf', help='path to the json file of province_lines.json')
    # parser.add_argument('-uf', help='path to the json file of univeristy_lines.json')
    parser.add_argument('-uf', help='path to the json file of univeristy_lines_batch.json')
    parser.add_argument('-kf', help='path to the json file of kldm_values.json')
    parser.add_argument('-cf', help='path to the list of candidate file')
    parser.add_argument('-t', help='type of extractions: 0|1|2|3, 0 denotes do nothing, 1 denotes extract_pure_uni_line, '
                                   '2 denotes extract_aver_uni_line, 3 denotes extract_uni_line_rank')
    args = parser.parse_args()

    if args.pf is None:
        args.pf = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/province_lines.json'
    if args.uf is None:
        # args.uf = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/university_lines.json'
        args.uf = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/university_lines_batch.json'
    if args.kf is None:
        args.kf = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/kldm_values.json'
    if args.cf is None:
        args.cf = '/home/ffl/nus/MM/cur_trans/data/prepared/candidate_universites.csv'
    if args.t is None:
        args.t = 'line_fir'

    gf = gaokao_feature(args.uf, args.pf, args.kf)
    args.t = '2'
    if args.t == '1':
        gf.extract_pure_uni_line()
    elif args.t == '2':
        gf.extract_aver_uni_line()
    elif args.t == '3':
        gf.extract_uni_line_rank()
    elif args.t == 'mul_bat_cou':
        gf.observe_mul_bat_cou(args.cf)
    elif args.t == 'mul_bat_coo':
        gf.observe_mul_bat_coo(args.cf)
    elif args.t == 'line_fir':
        gf.extract_uni_line_fir('2015'.encode('utf-8'), with_head=False)
    else:
        print 'do nothing'
    # gf.extract_feature()