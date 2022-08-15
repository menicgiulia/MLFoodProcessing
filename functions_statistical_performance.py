import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from scipy import interp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import operator

#AUC ROC Curve Scoring Function for Multi-class Classification
#"macro"
#"weighted"
# None
def multiclass_roc_auc_score(y_test, y_probs, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_probs, average=average)

def multiclass_average_precision_score(y_test, y_probs, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return average_precision_score(y_test, y_probs, average=average)

def multiclass_roc_curve(y_test, y_probs):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    fpr = dict()
    tpr = dict()
    for i in range(y_probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_probs[:, i])
        
    return (fpr, tpr)

def multiclass_average_precision_curve(y_test, y_probs):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    precision = dict()
    recall = dict()
    for i in range(y_probs.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_probs[:, i])
    
    return (precision, recall)

def Accuracykfold(X,y,splits):
    Xs = np.copy(X)
    ys=np.copy(y)
    numfolds=5;
    numlabels=4;
    performancesAccuracy=np.empty([numfolds, 1]);
    
    
    index=0
    for train, test in splits:
        # print("%s %s" % (train, test))
        clf = RandomForestClassifier(n_estimators = 200, max_features='sqrt', max_depth=420,random_state=0)
        #{'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 420}
        clf.fit(Xs[train,:], ys[train])

        # Predicting the Test set results
        y_pred = clf.predict(Xs[test,:])
        performancesAccuracy[index]=accuracy_score(ys[test], y_pred)
        index+=1

    return performancesAccuracy

# returns performances and splits/models used in the cross-validation
def AUCAUPkfold(X,y,smoteflag,verbose=True):
    numfolds=5;
    numlabels=4;
    cv = StratifiedKFold(n_splits=numfolds, shuffle=True)
    Xs = np.copy(X)
    ys=np.copy(y)
    
    if smoteflag==True:
        smote=SMOTE()
        Xs, ys= smote.fit_sample(Xs, ys)
        
    performancesAUC=np.empty([numfolds, numlabels]);
    performancesAUP=np.empty([numfolds, numlabels]);
    splits=[];
    model_per_fold=[]
    index=0
    for train, test in cv.split(Xs, ys):
        # print("%s %s" % (train, test))
        clf = RandomForestClassifier(n_estimators = 200, max_features='sqrt', max_depth=420, random_state=0)
        #clf = RandomForestClassifier(n_estimators = 1800, max_features='sqrt', max_depth=260)
        #{'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 420}
        
        splits.append([train, test])
        #print(Xs[train,:].shape)
        #print(ys[train].shape)
        clf.fit(Xs[train,:], ys[train])

        # Predicting the Test set results
        y_pred = clf.predict(Xs[test,:])
        y_probs= clf.predict_proba(Xs[test,:])
        performancesAUC[index,:]=np.array(multiclass_roc_auc_score(ys[test], y_probs, average=None))
        performancesAUP[index,:]=np.array(multiclass_average_precision_score(ys[test], y_probs, average=None))
        index+=1
        model_per_fold.append(clf)
        #if verbose==True:
        #    print(multiclass_roc_auc_score(ys[test], y_probs, average=None))
    
    if verbose==True:
        print("AUC: average over the folds")
        print(performancesAUC.mean(axis=0))
        print("AUC: std over the folds")
        print(performancesAUC.std(axis=0))
        
    if verbose==True:
        print("AUP: average over the folds")
        print(performancesAUP.mean(axis=0))
        print("AUP: std over the folds")
        print(performancesAUP.std(axis=0))
        
        
    return (performancesAUC, performancesAUP, splits, model_per_fold) 


def ROCkfold(X,y,splits,verbose=True):
    Xs = np.copy(X)
    ys=np.copy(y)
    numfolds=5;
    numlabels=4;       
    
    mean_fpr = np.linspace(0, 1, 500)
    performancesAUC=np.empty([numfolds, numlabels]);
    performancesROC=[np.empty([numfolds,len(mean_fpr)]) for ind in range(numlabels)];
    
    index=0
    for train, test in splits:
        # print("%s %s" % (train, test))
        clf = RandomForestClassifier(n_estimators = 200, max_features='sqrt', max_depth=420,random_state=0)
        #{'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 420}
        clf.fit(Xs[train,:], ys[train])

        # Predicting the Test set results
        y_pred = clf.predict(Xs[test,:])
        y_probs= clf.predict_proba(Xs[test,:])
        # ROC curve for the classes--> returns two dictionaries
        fprclasses, tprclasses=multiclass_roc_curve(ys[test], y_probs)
        
        for c in range(numlabels):
            interp_tpr = interp(mean_fpr, fprclasses[c], tprclasses[c])
            interp_tpr[0] = 0.0
            performancesROC[c][index, :]=interp_tpr
            
        
        performancesAUC[index,:]=np.array(multiclass_roc_auc_score(ys[test], y_probs, average=None))
        index+=1
        #if verbose==True:
        #    print(multiclass_roc_auc_score(ys[test], y_probs, average=None))
            
            
    mean_tpr =[np.empty([1,len(mean_fpr)]) for ind in range(numlabels)];
    std_tpr =[np.empty([1,len(mean_fpr)]) for ind in range(numlabels)];
    tprs_upper =[np.empty([1,len(mean_fpr)]) for ind in range(numlabels)];
    tprs_lower =[np.empty([1,len(mean_fpr)]) for ind in range(numlabels)];
    
    for c in range(numlabels):
        mean_tpr[c]=np.mean(performancesROC[c], axis=0)
        mean_tpr[c][-1] = 1.0
        std_tpr[c]=np.std(performancesROC[c], axis=0)
        tprs_upper[c]=np.minimum(mean_tpr[c] + std_tpr[c], 1)
        tprs_lower[c]=np.maximum(mean_tpr[c] - std_tpr[c], 0)   
    
    if verbose==True:
        print("AUC: average over the folds")
        print(performancesAUC.mean(axis=0))
        print("AUC: std over the folds")
        print(performancesAUC.std(axis=0))
        
    return performancesAUC, performancesROC, mean_fpr, mean_tpr, std_tpr, tprs_upper, tprs_lower



def ROCplot(mean_fpr, mean_tpr, std_tpr, tprs_upper, tprs_lower, performancesAUC, labelc):
    mean_auc=performancesAUC.mean(axis=0)
    std_auc=performancesAUC.std(axis=0)
                                
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC "+"Class "+ str(labelc+1))
    ax.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


# labeld--> label for all the dbs

def ROCMultiPlot(mean_fprL, mean_tprL, std_tprL, tprs_upperL, tprs_lowerL, performancesAUCL, labelc, labeld, colord):
    mean_auc=[p.mean(axis=0) for p in  performancesAUCL]
    std_auc=[p.std(axis=0) for p in  performancesAUCL]

    fig, ax = plt.subplots(figsize=(5, 5))
    #ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    for d in range(len(labeld)):
        ax.plot(mean_fprL[d], mean_tprL[d], color=colord[d],label=labeld[d]+ r': AUC = %0.4f $\pm$ %0.4f' % (mean_auc[d], std_auc[d]), lw=2, alpha=.5)
        ax.fill_between(mean_fprL[d], tprs_lowerL[d], tprs_upperL[d], color=colord[d], alpha=.1)

    ax.set(xlim=[0, 1], ylim=[0, 1], title="ROC "+"Class "+ str(labelc+1))        
    ax.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def ROCMultiPlotCallable(mean_fprL, mean_tprL, std_tprL, tprs_upperL, tprs_lowerL, performancesAUCL, labelc, labeld, colord,ax):
    mean_auc=[p.mean(axis=0) for p in  performancesAUCL]
    std_auc=[p.std(axis=0) for p in  performancesAUCL]
    #ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    for d in range(len(labeld)):
        ax.plot(mean_fprL[d], mean_tprL[d], color=colord[d],label=labeld[d]+ r': AUC = %0.4f $\pm$ %0.4f' % (mean_auc[d], std_auc[d]), lw=2, alpha=.5)
        ax.fill_between(mean_fprL[d], tprs_lowerL[d], tprs_upperL[d], color=colord[d], alpha=.1)
     
    ax.legend(loc="lower right")


def PrecisionRecallCurvekfold(X,y,splits,verbose=True):
    Xs = np.copy(X)
    ys=np.copy(y)
    numfolds=5;
    numlabels=4;

    mean_recall = np.linspace(1, 0, 500)
    performancesAUP=np.empty([numfolds, numlabels]);
    performancesPrecisionRecall=[np.empty([numfolds,len(mean_recall)]) for ind in range(numlabels)];
    
    
    index=0
    for train, test in splits:
        # print("%s %s" % (train, test))
        clf = RandomForestClassifier(n_estimators = 200, max_features='sqrt', max_depth=420,random_state=0)
        #clf = RandomForestClassifier(n_estimators = 1800, max_features='sqrt', max_depth=260)
        #{'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 420}
        clf.fit(Xs[train,:], ys[train])

        # Predicting the Test set results
        y_pred = clf.predict(Xs[test,:])
        y_probs= clf.predict_proba(Xs[test,:])
        # Precision-Recall curve for the classes--> returns two dictionaries
        precisionclasses, recallclasses=multiclass_average_precision_curve(ys[test], y_probs)
       
        
        for c in range(numlabels):
            interp_precision = interp(mean_recall, recallclasses[c][::-1], precisionclasses[c][::-1])
            performancesPrecisionRecall[c][index, :]=interp_precision
            
        
        performancesAUP[index,:]=np.array(multiclass_average_precision_score(ys[test], y_probs, average=None))
        index+=1
        #if verbose==True:
        #    print(multiclass_average_precision_score(ys[test], y_probs, average=None))
            
            
    mean_precision =[np.empty([1,len(mean_recall)]) for ind in range(numlabels)];
    std_precision =[np.empty([1,len(mean_recall)]) for ind in range(numlabels)];
    precision_upper =[np.empty([1,len(mean_recall)]) for ind in range(numlabels)];
    precision_lower =[np.empty([1,len(mean_recall)]) for ind in range(numlabels)];
    
    for c in range(numlabels):
        mean_precision[c]=np.mean(performancesPrecisionRecall[c], axis=0)
        std_precision[c]=np.std(performancesPrecisionRecall[c], axis=0)
        precision_upper[c]=np.minimum(mean_precision[c] + std_precision[c], 1)
        precision_lower[c]=np.maximum(mean_precision[c] - std_precision[c], 0)   
    
    if verbose==True:
        print("AUP: average over the folds")
        print(performancesAUP.mean(axis=0))
        print("AUP: std over the folds")
        print(performancesAUP.std(axis=0))
        
    return performancesAUP, performancesPrecisionRecall, mean_recall, mean_precision, std_precision, precision_upper, precision_lower


def PrecisionRecallplot(mean_recall, mean_precision, std_precision, precision_upper, precision_lower, performancesAUP, labelc):
    mean_aup=performancesAUP.mean(axis=0)
    std_aup=performancesAUP.std(axis=0)
                                
    fig, ax = plt.subplots()
    ax.plot(mean_recall, mean_precision, color='b',label=r'Mean Precision Recall (AUP = %0.2f $\pm$ %0.2f)' % (mean_aup, std_aup), lw=2, alpha=.8)
    ax.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Precision-Recall Curve "+ "Class "+ str(labelc +1))
    ax.legend(loc="lower left")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


# labeld--> label for all the dbs
def PrecisionRecallMultiPlot(mean_recallL, mean_precisionL, std_precisionL, precision_upperL, precision_lowerL, performancesAUPL, labelc, labeld, colord):
    mean_aup=[p.mean(axis=0) for p in  performancesAUPL]
    std_aup=[p.std(axis=0) for p in  performancesAUPL]
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    for d in range(len(labeld)):
        ax.plot(mean_recallL[d], mean_precisionL[d], color=colord[d],label=labeld[d]+ r': AUP = %0.4f $\pm$ %0.4f' % (mean_aup[d], std_aup[d]), lw=2, alpha=.5)
        ax.fill_between(mean_recallL[d], precision_lowerL[d], precision_upperL[d], color=colord[d], alpha=.1)



    ax.set(xlim=[0, 1], ylim=[0, 1], title="Precision-Recall Curve "+ "Class "+ str(labelc +1)) 
    ax.legend(loc="lower left")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


def PrecisionRecallMultiPlotCallable(mean_recallL, mean_precisionL, std_precisionL, precision_upperL, precision_lowerL, performancesAUPL, labelc, labeld, colord,ax):
    mean_aup=[p.mean(axis=0) for p in  performancesAUPL]
    std_aup=[p.std(axis=0) for p in  performancesAUPL]
    
    for d in range(len(labeld)):
        ax.plot(mean_recallL[d], mean_precisionL[d], color=colord[d],label=labeld[d]+ r': AUP = %0.4f $\pm$ %0.4f' % (mean_aup[d], std_aup[d]), lw=2, alpha=.5)
        ax.fill_between(mean_recallL[d], precision_lowerL[d], precision_upperL[d], color=colord[d], alpha=.1)

    ax.legend(loc="lower left")


