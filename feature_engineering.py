# -*- conding: utf-8 -*-

'''	author
	@AIbert
	hantaoer@foxmail.com
'''
'''
	feature_engineering part selects best single features
	then combines them to cross_features
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cross_validation
from xgboost.sklearn import XGBClassifier as XGBC
from sklearn.metrics import roc_auc_score as AUC

data_a = pd.read_csv(r'E:\DataGo\Kesci\credit\A_train.csv')
data_b = pd.read_csv(r'E:\DataGo\Kesci\credit\B_train_new.csv')
test = pd.read_csv(r'E:\DataGo\Kesci\credit\B_test_new.csv')

# train_data visualization
fea = np.sum(data_a.isnull(),axis=0)
feb = np.sum(data_b.isnull(),axis=0)
plt.subplot(211).plot(fea.values)
plt.subplot(212).plot(feb.values)
plt.show()
# sort_values
plt.subplot(211).plot(np.sort(fea))
plt.subplot(212).plot(np.sort(feb))
plt.show()
# test_data visualization
fet = np.sum(test.isnull(),axis=0)
plt.plot(fet.values)
plt.plot(np.sort(fet))
plt.show()

# select 330 features with less null
a = fea[fea==fea[330]]
b = feb[feb==feb[330]]
t = fet[fet==fet[330]]

# save common features in a,b,t
c = a.index.difference(a.index.difference(b.index))
c = c.difference(c.difference(t.index))
c = pd.DataFrame(c)
c = c.append(['ProductInfo_89'])
c = c[c[0]!='no']
c.to_csv(r'E:\DataGo\Kesci\credit\index.csv',index=False)

# train these features select those importance > 0
clf1 = XGBC(max_depth=8,seed=999,n_estimators=100)
train = data_b.sample(3000,random_state=999)
v = data_b[~data_b['no'].isin(train['no'])]
clf1.fit(train[train.columns.difference(['flag','no'])],train['flag'])
p = clf1.predict_proba(v[v.columns.difference(['flag','no'])])[:,1]
print AUC(v['flag'],p)
# print clf1.feature_importances_
a  = pd.Series(clf1.feature_importances_)
ind = v.columns.difference(['flag','no'])
a.index=ind
select_a = a[a>0]

# log smoothing
data_a['ProductInfo_89'] = (np.log(data_a['ProductInfo_89'].fillna(-1)+1.1)).astype(np.int)
data_b['ProductInfo_89'] = (np.log(data_b['ProductInfo_89'].fillna(-1)+1.1)).astype(np.int)
test['ProductInfo_89'] = (np.log(test['ProductInfo_89'].fillna(-1)+1.1)).astype(np.int)


# features grouped by name
cross = select_a[select_a>0]
cross.shape
cu=[];cp=[];cw=[]
for i in cross.index:
    if(i[0]=='P'):
        cp.append(i)
    if(i[0]=='U'):
        cu.append(i)
    if(i[0]=='W'):
        cw.append(i)
# print len(cu),len(cp),len(cw)

# cross-features using web_info and product_info
for i in cw:
    for j in cp:
        data_a[i+j] = data_a[i]*data_a[j]
        data_b[i+j] = data_b[i]*data_b[j]
        test[i+j] = test[i]*test[j]


# save new data_set
print data_a.shape,data_b.shape,test.shape
data_a.to_csv(r'E:\Kesci\credit\A_train.csv',index=False)
data_b.to_csv(r'E:\Kesci\credit\B_train_new.csv',index=False)
test.to_csv(r'E:\Kesci\credit\B_test_new.csv',index=False)
