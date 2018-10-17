import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


from sklearn.model_selection import GridSearchCV
from  sklearn.tree  import  DecisionTreeClassifier
from  sklearn.ensemble  import  RandomForestClassifier , VotingClassifier
from  sklearn.linear_model  import  LogisticRegression
from  sklearn.metrics  import  accuracy_score , roc_curve , auc , f1_score, confusion_matrix, classification_report
from  sklearn.preprocessing  import  LabelEncoder , MinMaxScaler
from  sklearn  import svm #SVC , LinearSVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
#draw roc cuver
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.metrics import (geometric_mean_score, make_index_balanced_accuracy)
from imblearn.metrics import classification_report_imbalanced

def svm(X_tr, Y_tr, X_te, Y_te):
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)
    #parameters =  [{'kernel': ['rbf'], 'gamma': [1e-3],
     #                'C': [1]}]
                    #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = LinearSVC(random_state=0)
    #svc = svm.SVC()
    #clf = GridSearchCV(svc, parameters,cv= 5)
    clf.fit(X_tr, Y_tr)
    y_pred = clf.predict(X_te)
    fpr_vot , tpr_vot , _ = roc_curve(Y_te , y_pred , pos_label =1,  drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot , tpr_vot)
    cmat = classification_report_imbalanced(Y_te, y_pred)
    #print (cmat.diagonal()/cmat.sum(axis=1))
    print (cmat)
    print('The geometric mean is {}'.format(geometric_mean_score(Y_te,y_pred)))
    print('The auc is {}'.format(roc_auc_vot))
    print('The f1 is {}'.format(f1_score(Y_te, y_pred, average='weighted')))
    return clf, fpr_vot, tpr_vot, roc_auc_vot

def randomforest(X_tr, Y_tr, X_te, Y_te):
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)
    rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=40, oob_score = True)

    param_grid = {
    'n_estimators': [40, 100]}


    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid)
    CV_rfc.fit(X_tr, Y_tr)
    #print CV_rfc.best_params_
    #clf = RandomForestClassifier(n_estimators=150, random_state =42)
    #clf.fit(X_tr, Y_tr)
    y_pred = CV_rfc.predict(X_te)
    fpr_vot , tpr_vot , _ = roc_curve(Y_te , y_pred , pos_label =1,  drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot , tpr_vot)
    cmat = classification_report_imbalanced(Y_te, y_pred)
    #print (cmat.diagonal()/cmat.sum(axis=1))
    print (cmat)
    print('The geometric mean is {}'.format(geometric_mean_score(Y_te,y_pred)))
    print('The auc is {}'.format(roc_auc_vot))
    print('The f1 is {}'.format(f1_score(Y_te, y_pred, average='weighted')))
    return CV_rfc, fpr_vot, tpr_vot, roc_auc_vot
def decisiontree(X_tr, Y_tr, X_te, Y_te):
     if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)
     param_grid = {'max_depth': np.arange(3, 6)}

     tree = GridSearchCV(DecisionTreeClassifier(), param_grid)

     tree.fit(X_tr, Y_tr)
     print (tree.best_params_)
     #clf = DecisionTreeClassifier(random_state =150)
     #clf = clf.fit(X_tr, Y_tr)
     y_pred = tree.predict(X_te)
     fpr_vot , tpr_vot , _ = roc_curve(Y_te , y_pred , pos_label =1,  drop_intermediate=False)
     roc_auc_vot = auc(fpr_vot , tpr_vot)
     cmat = classification_report_imbalanced(Y_te, y_pred)
     #print (cmat.diagonal()/cmat.sum(axis=1))
     print (cmat)
     print('The geometric mean is {}'.format(geometric_mean_score(Y_te,y_pred)))
     print('The auc is {}'.format(roc_auc_vot))
     print('The f1 is {}'.format(f1_score(Y_te, y_pred, average='weighted')))
    
     return tree, fpr_vot, tpr_vot, roc_auc_vot
