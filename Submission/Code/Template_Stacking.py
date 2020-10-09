# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
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



def get_models():
    models = dict()
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    models['svm'] = SVC(kernel = 'linear')
    models['bayes'] = GaussianNB()
    models['rf'] = RandomForestClassifier(n_estimators=150, max_depth=10)
    return models


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

    train_X = X[:-150]
    train_y = y[:-150]

    test_X = X[-150:]
    test_y = y[-150:]

    return train_X, train_y, test_X, test_y

def model_weights(X, y):
    models = get_models()
    def evaluate_model(model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        return scores
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    plt.figure("distribution of accuracy scores for each algorithm")
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()
def main():

    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data()

    #get models and evaluate
    model_weights(train_X, train_y)

    # Stacking models:
    # Create your stacked model using StackingClassifier
    level0 = list()
    level0.append(('rf', RandomForestClassifier(n_estimators=150, max_depth=10)))
    level0.append(('svm', SVC(C=1, kernel='linear')))
    level0.append(('lr', LogisticRegression()))
    level0.append(('bayes', GaussianNB()))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('cart', DecisionTreeClassifier()))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    # fit the model on the training data
    model.fit(train_X, train_y)
    # Get and print f1-score on test data
    y_pred = model.predict(test_X)
    F1_score = metrics.f1_score(y_pred, test_y , average = 'weighted')
    print(F1_score)
if __name__ == '__main__':
    main()
