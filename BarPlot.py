#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: BarPlot.py
# Created Date: Monday November 11th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 11th November 2019 10:17:03 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8), dpi=80)

movie_name = ['雷神3：诸神黄昏','正义联盟','东方快车谋杀案']
first_day = [10587.6, 10062.5, 1275.7]
first_weekend = [36224.9, 34479.6, 11830]

# 先得到movie_name长度, 再得到下标组成列表
x = range(len(movie_name))

plt.bar(x, first_day, width=0.2)
# 向右移动0.2, 柱状条宽度为0.2
plt.bar([i + 0.2 for i in x], first_weekend, width=0.2)

# 底部汉字移动到两个柱状条中间(本来汉字是在左边蓝色柱状条下面, 向右移动0.1)
plt.xticks([i + 0.1 for i in x], movie_name)
plt.show()