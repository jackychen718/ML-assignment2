import numpy as np
from statistics import mode
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler



def split_train_test_breast_cancer():
    handle = open('breast-cancer-wisconsin.data', 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[i for i in r.split(',')] for r in rows if r])
    out=out[:,1:]
    for col in range(9):
        out[:,col][out[:,col]=='?']=mode(out[:,col])
    np.random.shuffle(out)
    test_data=out[:199,:]
    train_data=out[199:,:]
    train_features=np.array(train_data[:,:-1],dtype=int)
    train_labels=np.array(train_data[:,-1],dtype=int)
    train_labels[train_labels==4]=1
    train_labels[train_labels==2]=0
    test_features=np.array(test_data[:,:-1],dtype=int)
    test_labels=np.array(test_data[:,-1],dtype=int)
    test_labels[test_labels==4]=1
    test_labels[test_labels==2]=0
    return (train_features,train_labels,test_features,test_labels)
    
def split_train_test_spam():
    handle = open('spambase.data', 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[i for i in r.split(',')] for r in rows if r])
    np.random.shuffle(out)
    test_data=out[:601,:]
    train_data=out[601:,:]
    train_features=np.array(train_data[:,:-1],dtype=float)
    train_labels=np.array(train_data[:,-1],dtype=int)
    test_features=np.array(test_data[:,:-1],dtype=float)
    test_labels=np.array(test_data[:,-1],dtype=int)
    scaler=StandardScaler()
    scaler.fit(train_features)
    train_features_spam_norm=scaler.transform(train_features)
    test_features_spam_norm=scaler.transform(test_features)
    return (train_features_spam_norm,train_labels,test_features_spam_norm,test_labels)