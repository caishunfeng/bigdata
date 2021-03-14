import pandas as pd
from matplotlib import pyplot as plt


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#设置图例并且设置图例的字体及大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 5,
}

# crazy_features = pd.read_csv('./dataset/crazy_register_feature_new.csv')
# countryFeatures = crazy_features.groupby(['country'],as_index=False)['deviceid'].count()
# # 去除中国，基本99%都是中国的
# countryFeatures = countryFeatures.drop(0)

genderMap = {-1:'未知',0:'女',1:'男'}

crazy_features = pd.read_csv('./dataset/crazy_register_feature_new.csv')
crazy_features['gender'] = crazy_features['gender'].map(genderMap)
locationFeatures = crazy_features.groupby(['model'],as_index=False)['deviceid'].count()
# locationFeatures = locationFeatures[(locationFeatures['deviceid']<=50) & (locationFeatures['deviceid']>25)]
# locationFeatures = locationFeatures[locationFeatures['deviceid']<=25]

# region_null = pd.isnull(crazy_features['region'])
# crazy_features = crazy_features[region_null == False]
# crazy_features = crazy_features[crazy_features['cnt']>100]

plt.figure(figsize=(15, 8))
b = plt.bar(locationFeatures['model'], locationFeatures['deviceid'],label="BMW", color='b', width=.5)
plt.legend(handles=[b],prop=font1)
plt.xlabel('性别')
plt.ylabel('数量')
plt.title('疯狂原始人正样本机型分布')
plt.xticks(rotation=90)
plt.show()