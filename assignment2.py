#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
testData=pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
testData['meal'] = testData['meal'].fillna(0)

y=data['meal']
x=data.drop(['meal', 'id', 'DateTime'], axis=1)

yt=testData['meal']
xt=testData.drop(['meal', 'id', 'DateTime'], axis=1)


model=XGBClassifier(
    n_estimators=500,
    max_depth=30,
    learning_rate=0.50,
    objective='binary:logistic'
)

modelFit = model.fit(x,y)

predictions=modelFit.predict(xt)

predictions=predictions.flatten()

pred=predictions.astype(int).tolist()

print(f"Accuracy score: {accuracy_score(yt,pred)*100:.2f}%")

