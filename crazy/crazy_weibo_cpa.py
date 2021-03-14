import pandas as pd

imp_prices = pd.read_csv('./dataset/crazy_weibo_imp_price.csv')
register_cnt = pd.read_csv('./dataset/crazy_weibo_register_cnt.csv')

impMap = dict()

for index,row in imp_prices.iterrows():
   impMap[row['weibolabel']] = row['feature_price']

def featureMap(feature):
    return float(impMap[feature])/1000/100


register_cnt['price'] = register_cnt['weibolabel'].map(featureMap)

register_cnt.to_csv('./dataset/crazy_weibo_cpa.csv')