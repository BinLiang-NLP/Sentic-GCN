#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------
# @File Name: handle_xml.py
# @Version: 
# @Author: BinLiang
# @Mail: 18b951033@stu.hit.edu.cn
# @For: 
# @Created Time: Tue 16 Jun 2020 03:34:21 PM CST
# ------------------

from xml.dom.minidom import parse
import nltk

def load_data():
    polarity_dic = {"positive": '1', "neutral": '0', "negative": '-1'}
    w_fp = open('./con_datasets/msma_test.raw','w')
    domTree = parse("./orig_datasets/msma_test.xml")
    rootNode = domTree.documentElement
    all_sentences = rootNode.getElementsByTagName("sentence")
    for sentence in all_sentences:
        text = sentence.getElementsByTagName("text")[0].childNodes[0].data
        text = nltk.word_tokenize(text)
        text = ' '.join(text).lower().strip()
        aspectTerms = sentence.getElementsByTagName("aspectTerms")[0]
        all_terms = aspectTerms.getElementsByTagName("aspectTerm")
        aspect_list = []
        polarity_list = []
        position_list = []
        for item in all_terms:
            aspect = item.getAttribute("term")
            aspect = nltk.word_tokenize(aspect)
            aspect = ' '.join(aspect).lower().strip()
            polarity = item.getAttribute("polarity")
            text_left, _, _ = text.partition(aspect)
            position = len(text_left.split())
            aspect_list.append(aspect)
            polarity_list.append(polarity_dic[polarity])
            position_list.append(str(position))
        line = '||'.join(aspect_list) + '\t' + '||'.join(polarity_list) + '\t' + '||'.join(position_list) + '\t' + str(text) + '\n'
        w_fp.write(line)
    w_fp.close()

if __name__ == '__main__':
    load_data()
