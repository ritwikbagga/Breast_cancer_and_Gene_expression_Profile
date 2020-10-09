# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import math

def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 63
    # of testing samples: 20
    ------
    """
    train_X = np.genfromtxt("../../Data/gene_data/gene_train_x.csv", delimiter= ",")
    train_y = np.genfromtxt("../../Data/gene_data/gene_train_y.csv", delimiter= ",")
    test_X = np.genfromtxt("../../Data/gene_data/gene_test_x.csv", delimiter= ",")
    test_y = np.genfromtxt("../../Data/gene_data/gene_test_y.csv", delimiter= ",")

    return train_X, train_y, test_X, test_y



def main():
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data()

    Number_S, Dim = train_X.shape
    x_arr=[]
    err_m1 =[]  # m = sqrt(n_features)
    err_m2 = [] # m = n_features
    err_m3 =[]  # m = n_features/10
    m1 = int(math.sqrt(Dim))
    m2 = Dim
    m3 = int(Dim/3)
    N = 150 # Each part will be tried with 1 to 150 estimators
    print("running for model Random Forest Classifer")
    for i in range(1, N+1):
        x_arr.append(i)
    # Train RF with m = sqrt(n_features) recording the errors (errors will be of size 150)

        p1 = int(math.sqrt(Dim))
        clf1 = RandomForestClassifier(n_estimators= i, max_features= p1)
        clf1.fit(train_X, train_y)
        y_pred = clf1.predict(test_X)
        accuracy = metrics.accuracy_score(test_y, y_pred)
        err_m1.append(1-accuracy)

    # Train RF with m = n_features recording the errors (errors will be of size 150)

        p2 = Dim
        clf2 = RandomForestClassifier(n_estimators=i, max_features=p2)
        clf2.fit(train_X, train_y)
        y_pred = clf2.predict(test_X)
        accuracy = metrics.accuracy_score(test_y, y_pred)
        err_m2.append(1-accuracy)

    # Train RF with m = n_features/10 recording the errors (errors will be of size 150)

        p3 = int(Dim/3)
        clf3 = RandomForestClassifier(n_estimators=i, max_features=p3)
        clf3.fit(train_X, train_y)
        y_pred = clf3.predict(test_X)
        accuracy = metrics.accuracy_score(test_y, y_pred)
        err_m3.append(1-accuracy)

    # plot the Random Forest results
    plt.figure("Ensemble Random Forest Classifier")
    plt.plot(x_arr,err_m1 , c='r', label=f"max_features = {m1} | √(p)")
    plt.plot(x_arr, err_m2, c='g', label=f"max_features = {m2} | p")
    plt.plot(x_arr, err_m3, c='b', label=f"max_features = {m3} | p/3")
    plt.ylabel("Test Classification Error")
    plt.xlabel("# of Trees (n_estimators)")
    plt.legend(loc=1)
    plt.show()
    #plt.savefig('../Figures/ensemble_randomforest_Q2.4_plot.png')

    breakpoint()


    

    
    # Train AdaBoost with max_depth = 1 recording the errors (errors will be of size 150)

    # Train AdaBoost with max_depth = 3 recording the errors (errors will be of size 150)

    # Train AdaBoost with max_depth = 5 recording the errors (errors will be of size 150)

    # plot the adaboost results

if __name__ == '__main__':
    main()
