import pandas as pd
import numpy as np
import MultiColumnLabelEncoder
import pickle

df=pd.read_csv('Dataset/garments_worker_productivity.csv')

df=df.drop(columns=['wip'])

#converting date column to datetime format
df['date']=pd.to_datetime(df['date'])

#adding new column of month
df['month']=df['date'].dt.month

#now no need of date column so remove it
df=df.drop(columns=['date'])

#as finishing split in two (maybe due to spaces) hence merge them
df['department']=df['department'].apply(lambda x:'finishing' if x.replace(" ","")=='finishing' else 'sweing')

mcle=MultiColumnLabelEncoder.MultiColumnLabelEncoder()

df=mcle.fit_transform(df)

x=df.drop(['actual_productivity'],axis=1)
y=df['actual_productivity']

X=x.to_numpy()

#splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

import xgboost as xgb

model_xgb=xgb.XGBRegressor(n_estimators=200,max_depth=5,learning_rate=0.05)

model_xgb.fit(x_train,y_train)

xg_prediction=model_xgb.predict(x_test)

#printing results
from sklearn import metrics
from sklearn.metrics import r2_score

print('MSE:-',metrics.mean_squared_error(y_test,xg_prediction))
print('MAE:-',metrics.mean_absolute_error(y_test,xg_prediction))
print('r2_score:',r2_score(y_test,xg_prediction)*100)


#save trained model
pickle.dump(model_xgb,open('gwp.pkl','wb'))