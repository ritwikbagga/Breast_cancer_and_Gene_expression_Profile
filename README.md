# 1. Breast Cancer Detection using SVM 

The Breast Cancer Wisconsin (Diagnostic) Dataset contains 30 features computed from a digitized
image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei
present in the image. The dataset contains 569 samples which are split into 419 training samples, and
150 testing samples. The task is to classify the diagnosis of the breast tissues to be malignant (label 1) or benign (label 0).

Results:
We trained a SVM classifier with different kernels (RBF, Linear, Polynomial) and got an F1 score of 0.9 (train data) and 0.8 (Test data) with Linear kernel. 


# 2. Breast Cancer Detection by Stacking different models

After stacking different models the F1 score jumped to 0.966 (Test data).

From the nature of the data we chose:

In base model: RandomForestClassifier(), DecisionTreeClassifier(), AdaBoostClassifier(), LogisticRegression(), GaussianNB() 
because the diverse range models make different assumptions about the prediction class.

In the final layer: LogisticRegression() 
because it provides a smooth interpretation of the prediction made by the base models. 






