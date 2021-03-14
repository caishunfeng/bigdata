import pandas as pd
from matplotlib import pyplot as plt

# weiboMap = dict()
# weibo_feature = pd.read_csv('./dataset/crazy_to_weibo_feature.csv')
# txt = ''
# for index,row in weibo_feature.iterrows():
#     if index==0:
#         continue
#     strs = row['weibo_feature'].split('|')
#     if len(strs) > 0:
#         for s in strs:
#             if s in weiboMap:
#                 weiboMap[s] = weiboMap[s]+1
#             else:
#                 weiboMap[s] = 1
        
# for k,v in weiboMap.items():
#     txt += k + ',' + str(v) + '\n'

# with open("./dataset/weibo_feature_count.csv","w") as f:
#         f.write(txt)

# print(weiboMap)


weibo_feature_count = pd.read_csv('./dataset/weibo_feature_count.csv')
# print(weibo_feature_count['feature'])

weibo_feature_count['feature'] = weibo_feature_count['feature'].apply(str)

plt.bar(weibo_feature_count['feature'],weibo_feature_count['cnt'],label="BMW",color='b',width=0.5)
plt.legend()
plt.xlabel('weibo_feature')
plt.ylabel('count')
plt.title('crazy weibo feature')
plt.show()