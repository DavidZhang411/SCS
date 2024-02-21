import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def breast_cancer():
    dataset_path = os.path.abspath("data/wdbc.data")
    df = pd.read_csv(dataset_path,header=None)
    Word = 1
    df[Word] = df[Word].map({"B": 1, "M": -1})
    X_df = df.drop([0,Word], axis=1)
    Y_df = df[Word]

    X = X_df.to_numpy()
    X = normalize(X, axis=1, norm='l1')
    #n,m = X.shape # for generality
    #X0 = np.ones((n,1))
    #X = np.hstack((X,X0))
    Y = Y_df.to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test

def heart_failure():
    dataset_path = os.path.abspath("data/heart_failure_clinical_records_dataset.csv")
    df = pd.read_csv(dataset_path)  

    Word = "DEATH_EVENT"
    df[Word] = df[Word].map({1: 1, 0: -1})
    X_df = df.drop([Word], axis=1)
    Y_df = df[Word]

    X = X_df.to_numpy()
    X = normalize(X, axis=1, norm='l1')
    #n,m = X.shape # for generality
    #X0 = np.ones((n,1))
    #X = np.hstack((X,X0))
    Y = Y_df.to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def wine_quality():
    dataset_path = os.path.abspath("data/winequality-red.csv")
    df = pd.read_csv(dataset_path, sep=';')  
    Word = 'quality'
    df = df[df[Word].isin({4,7})]
    df[Word] = df[Word].map({7: 1, 4: -1})
    X_df = df.drop([Word], axis=1)
    Y_df = df[Word]

    X = X_df.to_numpy()
    X = normalize(X, norm='l2')
    #n,m = X.shape # for generality
    #X0 = np.ones((n,1))
    #X = np.hstack((X,X0))
    Y = Y_df.to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test