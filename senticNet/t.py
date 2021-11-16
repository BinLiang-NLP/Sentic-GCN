#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------
# @File Name: t.py
# @Version: 
# @Author: BinLiang
# @Mail: 18b951033@stu.hit.edu.cn
# @For: 
# @Created Time: Fri 05 Jun 2020 06:36:31 PM CST
# ------------------

path = './senticnet_word.txt'
w_path = './senticnet_word.sort'
fp = open(path, 'r')
w_fp = open(w_path, 'w')
word_dic = {}
for line in fp:
    line = line.strip()
    word, score = line.split('\t')
    word_dic[word] = float(score)
fp.close()
word_dic = sorted(word_dic.items(), key=lambda a: -abs(a[1]))
for word, score in word_dic:
    w_fp.write(word+'\t'+str(score)+'\n')
w_fp.close()
