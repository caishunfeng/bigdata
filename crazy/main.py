# import numpy as np
import pandas as pd

# 热云人群整理
# reyun = pd.read_csv('./dataset/reyun.csv',sep=',')
# reyun = reyun[pd.isna(reyun['userid'])==False]
# reyun.to_csv('./dataset/reyun2.csv',index=False)

# 舜飞标签转换
# labels = pd.read_csv('./dataset/label.csv',sep=',')
# labelMap = dict()
# for index,row in labels.iterrows():
#     labelMap[row['labelid']] = row['topic']

# # 人云人群标签
# reyuns = pd.read_csv('./dataset/reyun.csv',sep=',')
# reyunMap =  dict()
# for index,row in reyuns.iterrows():
#     reyunMap[row['userid']] = row['game']

# # 城市省份转换
# regions = pd.read_csv('./dataset/china_city.csv',sep=',')
# regionMap = dict()
# for index,row in regions.iterrows():
#     regionMap['region_'+str(row['dim_location.region'])] = row['dim_location.region_name_cn']
#     regionMap['city_'+str(row['dim_location.city'])] = row['dim_location.city_name_cn']

# # 地域修正
# def region_fix(region):
#     if region.find('city_')>=0 or region.find('region_')>=0:
#         s = int(region.split('_')[1])
#         if s in regionMap:
#             return regionMap[int(s)]
#         else:
#             return region
#     else :
#         return region

# # 正样本处理
# positive_crazy_feature = pd.read_csv('./dataset/positive_sample.csv', sep=',')
# # positive_crazy_feature = positive_crazy_feature[positive_crazy_feature['cnt']>100]

# # 舜飞标签转换
# # positive_crazy_feature['label'] = positive_crazy_feature['labelid'].map(labelMap)
# # 热云人群转换
# # positive_crazy_feature['labelid'] = positive_crazy_feature['labelid'].map(lambda x: reyunMap[x] if x in reyunMap else x)
# positive_crazy_feature['reyun_label'] = positive_crazy_feature['labelid'].map(reyunMap)

# # 地域转换
# positive_crazy_feature['region'] = positive_crazy_feature['labelid'].map(regionMap)
# positive_crazy_feature.to_csv('./dataset/positive_crazy_feature_new.csv',index=False)



# ------------------------------crazy_register_feature 分析---------------------------------------------------------
features = pd.read_csv('./dataset/crazy_register_feature.csv')
locations = pd.read_csv('./dataset/location.csv')

# print(features['country'].unique())
# print(features['region'].unique())
# print(features['city'].unique())

features = features.drop(['brand','ostype'],axis=1)
features = features.drop(['juxiao_feature','sina_feature','reyun_feature','app_action','weibo_mau','weibo_uid','weibo_feature'],axis=1)

genderMap = {'-':-1,'female':0,'male':1}
features['gender'] = features['gender'].map(genderMap)

features['age'] = features['age_low'] + '_' + features['age_high']
features = features.drop(['age_low','age_high'],axis=1)

locations['country'] = locations['country'].apply(str)
locations['region'] = locations['region'].apply(str)
locations['city'] = locations['city'].apply(str)

countryMap = dict()
regionMap = dict()
cityMap = dict()
for index,row in locations.iterrows():
    if row['country'] not in countryMap:
        countryMap[row['country']] = row['country_name_cn']
    if row['region'] not in regionMap:
        regionMap[row['region']] = row['region_name_cn']
    if row['city'] not in cityMap:
        cityMap[row['city']] = row['city_name_cn']

features['country'] = features['country'].map(countryMap)
features['region'] = features['region'].map(regionMap)
features['city'] = features['city'].map(cityMap)
    
features.to_csv('./dataset/crazy_register_feature_new.csv')

# gender 性别处理
# age_low age_high年龄处理
# model 机型处理
# brand 设备品牌处理 直接删除，全都是apple
# ostype 操作系统类型，直接删除，全都是apple
# 地域处理 国家、省份、城市
# 人群特征匹配 暂时不做 juxiao_feature,sina_feature,reyun_feature,app_action,weibo_mau,weibo_uid,weibo_feature
