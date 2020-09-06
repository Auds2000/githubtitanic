#!/usr/bin/env python
# coding: utf-8

# In[1]:


## BASIC TITANIC MODEL FOR TESTING PURPOSES ONLY 

import pandas as pd
from sklearn.model_selection import train_test_split 
import pickle 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 

# load data
titanic_train = 'titanictrain.csv' 
titanic_test = 'titanictest.csv'   

dft0 = pd.read_csv(titanic_train)
dft = pd.read_csv(titanic_test)


def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return "No title in name"
    
titles = sorted(set([x for x in dft0.Name.map(lambda x: get_title(x))])) 


def shorter_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ['Jonkheer', 'Don', 'the Countess', 'Dona', 'Lady', 'Sir']:
        return 'Royalty'
    elif title == 'Mme':
        return 'Mrs'
    elif title in ['Mlle', 'Miss']:
        return 'Miss'
    else:
        return title
    
# new column for titles
dft0['Title'] = dft0['Name'].map(lambda x: get_title(x))
# 
dft0['Title'] = dft0.apply(shorter_titles, axis=1)   


# convert categorical values into numbers 
dft0.Sex.replace(('male', 'female'), (0,1), inplace=True)
dft0.Embarked.replace(('S', 'C', 'Q'), (0,1,2), inplace=True)
dft0.Title.replace(('Mr', 'Miss', 'Mrs','Master','Dr', 'Rev', 'Officer', 'Royalty'), (0,1,2,3,4,5,6,7), inplace=True)
 

# fill missing values
dft0['Age'].fillna(dft0['Age'].median(), inplace=True)
dft0['Fare'].fillna(dft0['Fare'].median(), inplace=True)
dft0['Embarked'].fillna(dft0['Embarked'].median(), inplace=True)

# removing the name column 
# del dft0['Name']
dft0.drop('Name', axis=1, inplace=True)
dft0.drop('Ticket', axis=1, inplace=True)
dft0.drop('Cabin', axis=1, inplace=True)




predictors = dft0.drop(['Survived', 'PassengerId', 'Title'], axis=1)
target = dft0['Survived']

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size=0.22, random_state=0)  



# random forest
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)

filename = 'titanic_model.sav'
pickle.dump(randomforest, open(filename, "wb"))

