#!/usr/bin/env python
# coding: utf-8

# In[329]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[330]:


train_csv = pd.read_csv('train.csv')


# In[331]:


final_csv = pd.read_csv('test.csv')


# In[332]:


train_csv.columns


# In[333]:


def show_null_count(csv):
    idx = csv.isnull().sum()
    idx = idx[idx>0]
    idx.sort_values(inplace=True)
    idx.plot.bar()


# In[334]:


def get_corr(col, csv):
    corr = csv.corr()[col]
    idx_gt0 = corr[corr>0].sort_values(ascending=False).index.tolist()
    return corr[idx_gt0]


# In[335]:


show_null_count(train_csv)


# In[336]:


sns.heatmap(train_csv.corr(), vmax=.8, square=True)


# In[337]:


print(get_corr('additional_fare', train_csv))


# In[338]:


from datetime import datetime
def find_time(csv):
    t2_data=csv['drop_time']
    t1_data=csv['pickup_time']
    data=[]
    for i in range(0,len(t2_data)):
        t2=t2_data[i].split(' ')[1]
        t1=t1_data[i].split(' ')[1]
        FMT='%H:%M'
        tdelta = (datetime.strptime(t2, FMT) - datetime.strptime(t1, FMT)).total_seconds()
        data.append(tdelta)
    csv['time']=data
find_time(train_csv)


# In[339]:


train_csv['label']=train_csv['label'].replace('incorrect',0)


# In[340]:


train_csv['label']=train_csv['label'].replace('correct',1)


# In[341]:


print(train_csv.head(n=10))


# In[342]:


train_csv.dropna(axis = 0, how ='any',inplace=True)
train_csv.reset_index(drop=True,inplace=True)
train_csv.isnull().sum()


# In[343]:


print(train_csv.head(n=5))


# In[344]:


train_csv['additional_fare'].unique()


# In[345]:


train_csv.isnull().sum()


# In[346]:


train_csv['additional_fare_bin'] = pd.cut(train_csv['additional_fare'], bins=[0,6,11,31,61,int(max(train_csv['additional_fare'] ))+1], labels=['v_l','l','m','h','v_h'])


# In[347]:


min(train_csv['additional_fare'] )


# In[348]:


# train_csv['pick_lat_bin'] = pd.cut(train_csv['pick_lat'], bins=[0,6.99999,9], labels=[0,1])
# train_csv['drop_lat_bin'] = pd.cut(train_csv['drop_lat'], bins=[0,6.99999,9], labels=[0,1])

# train_csv['pick_lon_bin'] = pd.cut(train_csv['pick_lon'], bins=[0,79.99999,90], labels=[0,1])
# train_csv['drop_lon_bin'] = pd.cut(train_csv['drop_lon'], bins=[0,79.99999,90], labels=[0,1])


# In[349]:


print(train_csv.head(n=5))


# In[350]:


train_csv.isnull().sum()


# In[351]:


def find_duration(csv):
    time=csv['time']
    duration=csv['duration']
    data1=[]
    for i in range(0,len(time)):
        val1=int(time[i])/100
        val2=int(duration[i])/100
        if(int(val1)==int(val2)):
            data1.append(1)
        else:
            data1.append(0)

    csv['check_duration']=data1
find_duration(train_csv)


# In[352]:


print(train_csv.head(n=5))


# In[353]:


def find_distance(csv):
    lat1=csv['pick_lat']
    lat2=csv['drop_lat']
    lon1=csv['pick_lon']
    lon2=csv['drop_lon']
    distance=[]
    for i in range(0,len(lat1)):
        s=((lat1[i]-lat2[i])**2+(lon1[i]-lon2[i])**2)**0.5
        distance.append(int(s*1000))
    csv['distance']=distance

find_distance(train_csv)
        


# In[354]:


print(train_csv['distance'].head(n=5))


# In[355]:


train_csv.drop(['tripid','additional_fare','pickup_time','drop_time','pick_lat','pick_lon','drop_lat','drop_lon','duration','time','meter_waiting_till_pickup'],axis=1,inplace=True)

   


# In[356]:


print(train_csv.head(n=5))


# In[357]:


print(get_corr('meter_waiting', train_csv))


# In[358]:


def waiting(csv):
    time=csv["meter_waiting"]
    fare=csv['meter_waiting_fare']
    waiting=[]
    for i in range(0,len(time)):
        if(int(time[i]/200)==int(fare[i]/15)):
           waiting.append(1)
        else:
           waiting.append(0)     
    csv['waiting']=waiting        
    
waiting(train_csv)


# In[359]:


train_csv.drop(['meter_waiting',"meter_waiting_fare"],axis=1,inplace=True)
print(train_csv.head(n=5))
   


# In[362]:


max(train_csv['distance'])


# In[361]:


train_csv['fare_bin'] = pd.cut(train_csv['fare'], bins=[0,100,500,1000,2000,int(max(train_csv['fare']))+1], labels=['v_l','l','m','h','v_h'])


# In[363]:


train_csv['distance_bin'] = pd.cut(train_csv['distance'], bins=[0,100,500,1000,2000,int(max(train_csv['fare']))+1], labels=['v_l','l','m','h','v_h'])


# In[364]:


train_csv.drop(['fare',"distance"],axis=1,inplace=True)
print(train_csv.head(n=5))


# In[366]:


train_df = pd.get_dummies(train_csv,prefix=None, prefix_sep='_', dummy_na=False, columns=['additional_fare_bin',"fare_bin","distance_bin"], sparse=False, drop_first=False, dtype=None)


# In[367]:


train_df.head(n=5)


# In[368]:


sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()


# In[369]:


def throttling(arr, thres):
    #res = arr.copy()
    res = np.zeros(len(arr))
    res[arr >= thres] = int(1)
    res[arr < thres] = int(0)
    return res


# In[370]:


from sklearn.model_selection import train_test_split


# In[372]:


x_train,x_test,y_train,y_test = train_test_split(train_df.drop('label', axis=1),
                                                 train_df['label'],
                                                 test_size=0.2,
                                                 random_state=123)


# In[373]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[374]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)


# In[375]:


print('The accuracy of the Logistic Regression is',round(accuracy_score(y_pred_lr,y_test)*100,2))


# In[376]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# In[377]:


def baselineNN(dims):
    model = Sequential()
    model.add(Dense(10, input_dim=dims, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[378]:


def use_keras_nn_model(x, y, xx, yy, epochs):
    model = baselineNN(x.shape[1])
    model.fit(x.as_matrix(), y.as_matrix(), epochs=epochs)
    y_pred = model.predict(xx.as_matrix()).reshape(xx.shape[0],)
    return y_pred, model


# In[379]:


y_pred_nn, model_nn = use_keras_nn_model(x_train, y_train, x_test, y_test, 100)


# In[380]:


#print('The accuracy of the Neural Network is',round(accuracy_score(y_pred_nn_thres,y_test)*100,2))
print('The accuracy of the Neural Network is',round(accuracy_score(throttling(y_pred_nn, 0.6), y_test)*100,2))


# In[381]:


import xgboost as xgb
from xgboost import plot_importance


# In[382]:


params = {
    'objective': 'binary:logistic',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

num_round = 10


# In[383]:


dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
watchlist = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(params, dtrain, num_round, watchlist)
y_pred_xgb = bst.predict(dtest)


# In[384]:


print('The accuracy of the Neural Network is',round(accuracy_score(throttling(y_pred_xgb, 0.6),y_test)*100,2))


# In[385]:


plot_importance(bst)


# In[386]:


final_csv.columns


# In[387]:


find_time(final_csv)


# In[388]:


final_csv.dropna(axis = 0, how ='any',inplace=True)
final_csv.reset_index(drop=True,inplace=True)
final_csv.isnull().sum()


# In[389]:


print(final_csv.head(n=5))


# In[390]:


final_csv.isnull().sum()


# In[392]:


final_csv['additional_fare_bin'] = pd.cut(final_csv['additional_fare'], bins=[0,6,11,31,61,int(max(final_csv['additional_fare'] ))+1], labels=['v_l','l','m','h','v_h'])


# In[393]:


find_duration(final_csv)


# In[394]:


find_distance(final_csv)


# In[395]:


final_csv.drop(['additional_fare','pickup_time','drop_time','pick_lat','pick_lon','drop_lat','drop_lon','duration','time','meter_waiting_till_pickup'],axis=1,inplace=True)

   


# In[396]:


waiting(final_csv)


# In[397]:


final_csv.drop(['meter_waiting',"meter_waiting_fare"],axis=1,inplace=True)
print(final_csv.head(n=5))


# In[398]:


final_csv['fare_bin'] = pd.cut(final_csv['fare'], bins=[0,100,500,1000,2000,int(max(final_csv['fare']))+1], labels=['v_l','l','m','h','v_h'])


# In[399]:


final_csv['distance_bin'] = pd.cut(final_csv['distance'], bins=[0,100,500,1000,2000,int(max(final_csv['fare']))+1], labels=['v_l','l','m','h','v_h'])


# In[400]:


final_csv.drop(['fare',"distance"],axis=1,inplace=True)
print(final_csv.head(n=5))


# In[401]:


final_df = pd.get_dummies(final_csv,prefix=None, prefix_sep='_', dummy_na=False, columns=['additional_fare_bin',"fare_bin","distance_bin"], sparse=False, drop_first=False, dtype=None)


# In[403]:


final_df.drop(['tripid'],axis=1,inplace=True)
final_df.head(n=5)


# In[404]:


y_final_prob = model_nn.predict(final_df.as_matrix()).reshape(final_df.shape[0],)


# In[405]:


y_final = throttling(y_final_prob, .6)


# In[406]:


submission = pd.concat([final_csv['tripid'], pd.DataFrame(y_final)], axis=1)
submission.columns = ['tripid', 'prediction']


# In[408]:


submission.to_csv('submission.csv', encoding='utf-8', index = False)


# In[ ]:




