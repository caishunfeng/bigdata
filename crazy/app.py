import pandas as pd
from matplotlib import pyplot as plt

# positive_crazy_feature = pd.read_csv('./dataset/positive_sample.csv', sep=',')
# app = ''
# for index,row in positive_crazy_feature.iterrows():
#     if len(row['labelid'].split('.'))>=2:
#         # print(row['labelid'],",",row['cnt'])
#         app+=row['labelid']
#         app+=","
#         app+=str(row['cnt'])
#         app+="\n"

# with open("./dataset/app.csv","w") as f:
#         f.write(app)

app = pd.read_csv('./dataset/crazy_app.csv')
# print(app)

plt.bar(app['app'],app['cnt'],label="BMW", color='b', width=2)
plt.legend()
plt.xlabel('包名')
plt.ylabel('注册数')
plt.title('crazy app包名 分布')
plt.show()