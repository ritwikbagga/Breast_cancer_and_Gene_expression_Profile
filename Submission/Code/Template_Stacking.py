# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
# feel free to import any sklearn model here
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing




def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 419
    # of testing samples: 150
    ------
    """
    df = pd.read_csv("../../Data/breast_cancer_data/data.csv")

    cols = df.columns
    X = df[cols[2:-1]].to_numpy()
    y = df[cols[1]].to_numpy()
    y = (y=='M').astype(np.int) * 2 - 1
    sc = preprocessing.StandardScaler()
    train_X = X[:-150]
    #train_X = sc.fit_transform(train_X)

    train_y = y[:-150]

    test_X = X[-150:]
    #test_X = sc.fit_transform(test_X)
    test_y = y[-150:]


    return train_X, train_y, test_X, test_y


def main():

    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data()

    # Stacking models:
    # Create your stacked model using StackingClassifier
    level0 = list()
    level0.append(('rf', RandomForestClassifier(n_estimators=150, max_depth=5)))
    level0.append(('svm', SVC(C=1, kernel='rbf')))
    dtc = DecisionTreeClassifier(max_depth=3)
    level0.append(('ADA', AdaBoostClassifier(n_estimators=100, base_estimator=dtc, learning_rate=0.1)))
    level0.append(('lr', LogisticRegression( solver='liblinear')))
    level0.append(('bayes', GaussianNB()))


    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10)
    # fit the model on the training data
    model.fit(train_X, train_y)
    # Get and print f1-score on test data
    y_pred = model.predict(test_X)
    F1_score = metrics.f1_score(y_pred, test_y , average= 'weighted')
    print("ANS 3.1 - F1_score of model with stacking different models is: "+ str(F1_score))
if __name__ == '__main__':
    main()
