import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# plt.plot(np.random.rand(10))
# plt.show()

# df = pd.DataFrame(np.random.rand(10, 2), columns=['A', 'B'])
# fig = df.plot(figsize=(8, 4))  # figsize 创建图表窗口，设置窗口大小

# plt.title('Figure Name')
# plt.xlabel('xxx')
# plt.ylabel('yyy')

# plt.legend(loc='upper right')  # 显示图例，loc表示位置

# plt.xlim([0, 12])  #x轴边界
# plt.xlim([0, 1.5])

# plt.xticks(range(10))  #设置x轴刻度
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

# fig.set_xticklabels("%.1f" % i for i in range(10))  #x轴刻度标签
# fig.set_yticklabels(
#     "%.2f" % i for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])  #y轴刻度标签

# plt.show()

#  独立设置
# s = pd.Series(np.random.randn(100).cumsum())
# s.plot(linestyle='--', marker='.', color="r", grid=True)
# plt.show()

iris_tf = pd.read_csv("./iris.csv")
X = iris_tf[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
             'Petal.Width']].values
y = iris_tf['Species'].values

# print(iris_tf.info())
# print(iris_tf.describe())
# print(iris_tf['Species'].value_counts())
# print(iris_tf.head())

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(211)

label_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

colors = ['blue', 'red', 'green']
markers = ['s', 'o', '^']

for lab, c, m in zip(range(3), colors, markers):
    ax.scatter(
        X[y == lab, 2],
        X[y == lab, 3],
        c=c,  # color
        marker=m,  # marker symbol
        s=40,  # markersize
        alpha=0.4,  # transparency
        label=label_dict[lab])

ax.set_xlabel('petal height (cm)')
ax.set_ylabel('petal width (cm)')
plt.legend(loc='upper left')
plt.grid()

ax = fig.add_subplot(212)
for lab, c, m in zip(range(3), colors, markers):
    ax.scatter(
        X[y == lab, 0],
        X[y == lab, 1],
        c=c,  # color
        marker=m,  # marker symbol
        s=40,  # markersize
        alpha=0.4,  # transparency
        label=label_dict[lab])

ax.set_xlabel('sepal height (cm)')
ax.set_ylabel('setal width (cm)')
plt.legend(loc='upper left')
plt.grid()

plt.show()
