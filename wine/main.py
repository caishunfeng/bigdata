import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# fixed acidity  非挥发性酸
# volatile acidity  挥发性酸度
# citric acid  柠檬酸
# residual sugar  残糖
# chlorides  氯化物
# free sulfur dioxide  游离二氧化硫
# total sulfur dioxide  总二氧化硫
# density    密度
# pH  酸碱度
# sulphates 硫酸盐
# alcohol  酒精
# quality  质量

# 颜色
color = sns.color_palette()
# 数据print精度
pd.set_option('precision', 3)

red_tf = pd.read_csv('./winequality-red.csv', sep=';')
white_tf = pd.read_csv('./winequality-white.csv', sep=';')

# print(red_tf.head())
# print(red_tf.tail())
# print(red_tf.info())

# print(white_tf.head())
# print(white_tf.tail())
# print(white_tf.info())
# print(red_tf['quality'].unique())
# print(white_tf['quality'].unique())

# plt.scatter(
#     red_tf['fixed acidity'],
#     red_tf['quality'],
#     label='fixed acidity:quality',
#     color='r')
# plt.show()

# plt.style.use('ggplot')
# colnm = red_tf.columns.tolist()
# fig = plt.figure(figsize=(10, 6))

# for i in range(12):
#     plt.subplot(2, 6, i + 1)
#     sns.boxplot(red_tf[colnm[i]], orient="v", width=0.5, color=color[0])
#     plt.ylabel(colnm[i], fontsize=12)

# plt.tight_layout()
# plt.show()
# print('\nFigure 1: Univariate Boxplots')

# colnm = red_tf.columns.tolist()
# plt.figure(figsize=(10, 8))

# for i in range(12):
#     plt.subplot(4, 3, i + 1)
#     red_tf[colnm[i]].hist(bins=100, color=color[0])
#     plt.xlabel(colnm[i], fontsize=12)
#     plt.ylabel('Frequency')
# plt.tight_layout()
# print('\nFigure 2: Univariate Histograms')
# plt.show()

# acidityFeat = [
#     'fixed acidity', 'volatile acidity', 'citric acid', 'free sulfur dioxide',
#     'total sulfur dioxide', 'sulphates'
# ]

# plt.figure(figsize=(10, 4))

# for i in range(6):
#     ax = plt.subplot(2, 3, i + 1)
#     v = np.log10(
#         np.clip(red_tf[acidityFeat[i]].values, a_min=0.001, a_max=None))
#     plt.hist(v, bins=50, color=color[0])
#     plt.xlabel('log(' + acidityFeat[i] + ')', fontsize=12)
#     plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6, 3))

# bins = 10**(np.linspace(-2, 2))
# plt.hist(
#     red_tf['fixed acidity'], bins=bins, edgecolor='k', label='Fixed Acidity')
# plt.hist(
#     red_tf['volatile acidity'],
#     bins=bins,
#     edgecolor='k',
#     label='Volatile Acidity')
# plt.hist(
#     red_tf['citric acid'],
#     bins=bins,
#     edgecolor='k',
#     alpha=0.8,
#     label='Citric Acid')
# plt.xscale('log')
# plt.xlabel('Acid Concentration (g/dm^3)')
# plt.ylabel('Frequency')
# plt.title('Histogram of Acid Concentration')
# plt.legend()
# plt.tight_layout()
# plt.show()

# style
sns.set_style('ticks')
sns.set_context("notebook", font_scale=1.4)

plt.figure(figsize=(6, 5))
cm = plt.cm.get_cmap('RdBu')
sc = plt.scatter(
    red_tf['fixed acidity'],
    red_tf['citric acid'],
    c=red_tf['pH'],
    vmin=2.6,
    vmax=4,
    s=15,
    cmap=cm)
bar = plt.colorbar(sc)
bar.set_label('pH', rotation=0)
plt.xlabel('fixed acidity')
plt.ylabel('citric acid')
plt.xlim(4, 18)
plt.ylim(0, 1)
plt.show()
# print('Figure 12: pH with Fixed Acidity and Citric Acid')