# -*- coding: utf-8 -*-
import argparse
from bs4 import BeautifulSoup
import json
import os
import re
import urlparse
from util import read_kldm_key_values, clean_uni_name
from util_expand import get_fnames

class yiben_selector:
    def __init__(self, pages_path, kldm_values_fname):
        self.pages_path = pages_path
        self.kldm_values_fname = kldm_values_fname
        self.fnames = get_fnames(self.pages_path, '.html')
        self.student_type = read_kldm_key_values(self.kldm_values_fname)
        self.response_no_result = ''
        self.university_lines = {}
        self.level_strs = ['第一批'.decode('utf-8'), '第二批'.decode('utf-8'), '第三批'.decode('utf-8')]
        self.kldm_strs = ['文科'.decode('utf-8'), '理科'.decode('utf-8')]
        self.para_kldm_strs = ['文史'.decode('utf-8'), '理工'.decode('utf-8')]
        self.year_sel = 2015
        self.pici = set()

    def extract_score(self):
        index = 0
        no_results = 0
        for fname in self.fnames:
            # # TEST
            # print 'working on:', fname
            parsed = urlparse.urlparse('http://gaokao.do?' + fname)
            ssdm = int(urlparse.parse_qs(parsed.query)['ssdm'][0])
            year = int(urlparse.parse_qs(parsed.query)['year'][0])
            kldm = int(urlparse.parse_qs(parsed.query)['kldm'][0])
            yxmc = urlparse.parse_qs(parsed.query)['yxmc'][0]
            # # TEST
            # print year, yxmc
            # parse single page
            soup = BeautifulSoup(open(self.pages_path + fname), 'lxml')
            # extract university line
            uni_results = self.extract_university_line(soup, fname)
            if uni_results == False:
                print '!!!parse error occurs while parsing university lines!!!'
                break
            elif uni_results == True:
                no_results += 1
            elif len(uni_results) > 0:
                if year not in self.university_lines.keys():
                    self.university_lines[year] = {}
                if yxmc not in self.university_lines[year].keys():
                    self.university_lines[year][yxmc] = {}
                if ssdm not in self.university_lines[year][yxmc].keys():
                    self.university_lines[year][yxmc][ssdm] = {}
                self.university_lines[year][yxmc][ssdm][kldm] = uni_results
                # # TEST
                # print self.has_yiben(self.university_lines[year][yxmc])
                # pass
            else:
                print '!!!unexpected return university line results!!!'
                break
            index += 1
            if index % 500 == 0:
                print '#files finished:', index
            # # TEST
            # if index > 50:
            #     break
        print '#file parsed:', index
        print '#file without result:', no_results
        for kv_year in self.university_lines.iteritems():
            print 'year:', kv_year[0], '#universities:', len(kv_year[1])
        print '#picis overall:', len(self.pici)
        for pici in self.pici:
            print pici

        '''traverse university lines to check whether there is any "本科一批" in the lines of every university:
            we consider two conditions seperately:
            1. has "本科一批" in the selected year
            2. always has "本科一批" in previous years
        '''
        uni_yiben_year_sel = set()      # universities have "本科一批" in the selected year
        # for kv_year in self.university_lines.iteritems():
        for yxmc_kv in self.university_lines[self.year_sel].iteritems():
            has_yiben = self.has_yiben(yxmc_kv[1])
            if has_yiben:
                uni_yiben_year_sel.add(yxmc_kv[0])
        uni_yiben = set()       # universities always have "本科一批" in previous years
        for uni_name in uni_yiben_year_sel:
            always_yiben = True
            for year in self.university_lines.iterkeys():
                if not year == self.year_sel:
                    if not (uni_name in self.university_lines[year].keys() and self.has_yiben(self.university_lines[year][uni_name])):
                        always_yiben = False
            if always_yiben:
                uni_yiben.add(uni_name)

        print '#universities have yiben:', len(uni_yiben_year_sel)
        print '#universities always have yiben:', len(uni_yiben)
        with open('university_yiben.csv', 'w') as uni_yiben_out:
            for uni in uni_yiben_year_sel:
                uni_yiben_out.write('%s\n' % (clean_uni_name(uni)))

        with open('university_yiben_always.csv', 'w') as uni_yiben_out:
            for uni in uni_yiben:
                uni_yiben_out.write('%s\n' % (clean_uni_name(uni)))
        uni_dif = uni_yiben_year_sel - uni_yiben
        for uni in uni_dif:
            print uni


    '''return: results = {year: {'文科': {1: [scores], 2: [scores], 3: [scores]}, '理科': {1: [scores]...}}}
                1, 2, and 3 denotes '第一批', '第二批', and '第三批' '''
    # def extract_province_line(self, soup, fname):
    #     tags = soup.find_all(id='rightHeight')
    #     if len(tags) == 1:
    #         right_tables = tags[0].find_all('table')
    #         right_divs = tags[0].find_all('div')
    #         if len(right_tables) == 4 and len(right_divs) == 4:
    #             results = {}
    #             for i in range(3):
    #                 # extract year
    #                 # year = int(right_divs[i].get_text().strip().replace('年', ''))
    #                 year_nums = re.findall('\d+', right_divs[i].get_text())
    #                 if not len(year_nums) == 1:
    #                     print 'unexpected year text:', right_divs[i].get_text()
    #                     return False
    #                 year = int(year_nums[0])
    #                 if not (year == 2013 or year == 2014 or year == 2015):
    #                     print 'unexpected year:', year, '\n\n', fname
    #                     return False
    #                 results[year] = {}
    #                 # extract table contents
    #                 table_rows = right_tables[i].find_all('tr')
    #                 if len(table_rows) == 3:
    #                     # traverse rows
    #                     for j in range(3):
    #                         table_columns = table_rows[j].find_all('td')
    #                         if len(table_columns) == 4:
    #                             # traverse columns
    #                             cell_strs = []
    #                             for k in range(4):
    #                                 cell_strs.append(table_columns[k].get_text().strip())
    #                             if j == 0:
    #                                 # if not (cell_strs[1] == '第一批' and cell_strs[2] == '第二批' and cell_strs[3] == '第三批'):
    #                                 if not cell_strs[1] == self.level_strs[0] and cell_strs[2] == self.level_strs[1] and cell_strs[3] == self.level_strs[2]:
    #                                     print 'unexpected title:', '\n\n', table_rows[j], '\n\n', fname
    #                                     return False
    #                             if j == 1:
    #                                 if not cell_strs[0] == self.kldm_strs[0]:
    #                                     print 'unexpected wen ke row:', '\n\n', table_rows[j], '\n\n', fname
    #                                     return False
    #                             if j == 2:
    #                                 if not cell_strs[0] == self.kldm_strs[1]:
    #                                     print 'unexpected li ke row:', '\n\n', table_rows[j], '\n\n', fname
    #                                     return False
    #                             if j == 1 or j == 2:
    #                                 results[year][self.para_kldm_strs[j - 1]] = {}
    #                                 for k in range(1, 4):
    #                                     cell_lines = map(int, re.findall('\d+', cell_strs[k]))
    #                                     if len(cell_lines) > 0:
    #                                         results[year][self.para_kldm_strs[j - 1]][k] = cell_lines
    #                         else:
    #                             print 'unexpected row:', '\n\n', table_rows[j], '\n\n', fname
    #                             return False
    #                 else:
    #                     print 'unexpected table:', '\n\n', right_tables[i], '\n\n', fname
    #                     return False
    #             return results
    #         else:
    #             print 'unexpected rightHeight div with #tables:', len(right_tables), '\n\n', right_tables, '\n\n', fname
    #             print 'unexpected rightHeight div with #divs:', len(right_divs), '\n\n', right_divs, '\n\n', fname
    #             return False
    #     else:
    #         print 'more than one tag with id=rightHeight', '\n\n', tags, '\n\n', fname
    #         return False

    def extract_university_line(self, soup, fname):
        '''return: {'本科一批': university_line(598), '本科二批': university_line(508)}'''
        tags = soup.find_all(id='leftHeight')
        if len(tags) == 1:
            left_divs = tags[0].contents
            if len(left_divs) == 9:
                content_div = left_divs[5]
                result_table = 0
                for child in content_div.children:
                    if child.name == 'table':
                        result_table = child
                # no result scores
                if result_table == 0:
                    response = content_div.get_text().strip()
                    if self.response_no_result == '':
                        self.response_no_result = response
                        print response
                    elif not self.response_no_result == response:
                        print 'unexpected response of no result:', response
                        return False
                    return True
                # table_rows = content_div.find_all('tr')
                # if len(table_rows) == 0:
                #     print content_div.get_text().strip()
                else:
                    table_rows = result_table.find_all('tr')
                    results = {}
                    for i in range(2, len(table_rows)):
                        tds = table_rows[i].find_all('td')
                        if len(tds) == 7:
                            pici = tds[2].get_text().strip()
                            # # TEST
                            # print pici
                            if pici not in self.pici:
                                self.pici.add(pici)
                            score = int(tds[5].get_text().strip())
                            results[pici] = score
                        else:
                            print 'unexpected table row in file:', fname, '\n\n', table_rows[i]
                            return False
                    if len(results) == 0:
                        print 'no result score in table:', '\n\n', result_table, '\n\n', fname
                        return False
                    return results
            else:
                print 'unexpected leftHeight div with #divs:', len(left_divs), '\n\n', left_divs, '\n\n', fname
                return False
        else:
            print 'more than one tag with id=leftHeight', '\n\n', tags, '\n\n', fname
            return False

    def has_yiben(self, uni_lines):
        # self.university_lines[year][yxmc][ssdm][kldm] = uni_results
        for ssdm_kv in uni_lines.iteritems():
            for kldm_kv in ssdm_kv[1].iteritems():
                for pici in kldm_kv[1].iterkeys():
                    if u'本科' in pici and u'一批' in pici:
                        return True
        return False


if __name__ == '__main__':
    desc = "extract scores from the yang guang gao kao pages"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path to the page files')
    parser.add_argument('-kv', help='kldm value fname')
    args = parser.parse_args()

    if args.p is None:
        args.p = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/pages/'
    if args.kv is None:
        args.kv = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/kldm_values.json'

    ybs = yiben_selector(args.p, args.kv)
    ybs.extract_score()
