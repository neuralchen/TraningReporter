#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Plot.py
# Created Date: Monday November 11th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 16th November 2019 1:32:23 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')

# Matplotlib forms basis for visualization in Python
import matplotlib.pyplot as plt

# We will use the Seaborn library
import seaborn as sns
sns.set(style='whitegrid',rc={
                            'axes.edgecolor':'0.0',
                            'xtick.bottom': True, 
                            'ytick.left': True,
                            'grid.linestyle': 'dotted',
                            },
                            font="Times New Roman",
                            font_scale=1.6
                            )

# Graphics in retina format are more sharp and legible
# InlineBackend.figure_format = 'retina' 

import pandas as pd

sns.axes_style()
df = pd.read_csv('./data.csv')
df.head()

fig = sns.catplot(data=df,
                    kind='bar',
                    x='Name',
                    y='Score',
                    hue='Arch',
                    legend=False, 
                    height=5,
                    aspect=2)
sns.set(font="times new roman",font_scale=1)

fig.set(ylabel='Attribute Generation Accuracy', xlabel='', ylim=(0,1))
fig.set_xticklabels(rotation=60)
sns.despine(left=False, top = False, right=False)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

legend_x = 0.5
legend_y = 1.
plt.legend(loc='lower center', bbox_to_anchor=(legend_x, legend_y), ncol = 5, frameon=False,prop=font1,labelspacing=0.3)
fig.savefig('diffframework.pdf')
