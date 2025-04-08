from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from KNN import KNN
from KNN import LDA
from KNN import RandomForest
import matplotlib.pyplot as plt
from plotter import BoxPlot, OptimismPlot

from helper import Evaluation, doubleCV, CVErrorVSHyperparam

def main():
    # Load the data
    df = pd.read_csv('data/Numbers.txt', delimiter=' ')
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    outerCV = KFold(n_splits=10, shuffle=True, random_state=42)
    foldDataTraining = []
    foldDataTest = []
    #Plot data
    allTestErrors = {}
    optimism = {}

    #Get the folded training and test data
    for train_idx, test_idx in outerCV.split(X, y):
        X_train = X.iloc[train_idx].to_numpy()
        y_train = y.iloc[train_idx].to_numpy().reshape(-1, 1)

        X_test = X.iloc[test_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy().reshape(-1, 1)

        train_combined = np.hstack((y_train, X_train))
        test_combined = np.hstack((y_test, X_test))

        foldDataTraining.append(train_combined)
        foldDataTest.append(test_combined)
    
    """Evaluation without tuning"""

    #KNN 
    smallKClassifier = KNN(n_neighbors=5)
    largeKClassifier = KNN(n_neighbors=20)
    smallKtrainingErrorsNoTuning,smallKtestErrorsNoTuning,_,_ = Evaluation(smallKClassifier,foldDataTraining,foldDataTest)
    largeKtrainingErrorsNoTuning,largeKtestErrorsNoTuning,_,_ = Evaluation(largeKClassifier,foldDataTraining,foldDataTest)
    allTestErrors["Small k (no tuning)"] = smallKtestErrorsNoTuning
    allTestErrors["Large k (no tuning)"] = largeKtestErrorsNoTuning

    KNNOptimismSmallK = np.array(smallKtestErrorsNoTuning) - np.array(smallKtrainingErrorsNoTuning)
    KNNOptimismLargeK = np.array(largeKtestErrorsNoTuning) - np.array(largeKtrainingErrorsNoTuning)

    optimism["Small k (no tuning)"] = KNNOptimismSmallK
    optimism["Large k (no tuning)"] = KNNOptimismLargeK

    #LDA 
    ldaClassifier = LDA() 
    ldaTrainingErrors, ldaTestErrors, _, _ = Evaluation(ldaClassifier, foldDataTraining, foldDataTest)
    allTestErrors["LDA (no tuning)"] = ldaTestErrors

    LDAOptimism = np.array(ldaTestErrors) - np.array(ldaTrainingErrors)

    optimism["LDA (no tuning)"] = LDAOptimism

    #RandomForest
    rfClassifier = RandomForest()
    rfTrainingErrors, rfTestErrors, _, _ = Evaluation(rfClassifier, foldDataTraining, foldDataTest)
    allTestErrors["Random Forest (no tuning)"] = rfTestErrors

    rfOptimism = np.array(rfTestErrors) - np.array(rfTrainingErrors)

    optimism["Random Forest (no tuning)"] = rfOptimism

    """Evaluation with tuning"""

    #KNN
    
    smallKparamGrid = {'n_neighbors': list(range(1,11))} #k = 1,2,...10 --> Flexible
    largeKparamGrid = {'n_neighbors': list(range(50, 151, 10))} #k = 50,60..150 --> Rigid

    _,smallKtrainingErrors,smallKtestErrors = doubleCV(foldDataTraining, foldDataTest, KNeighborsClassifier(), smallKparamGrid)
    _,largeKtrainingErrors,largeKtestErrors = doubleCV(foldDataTraining, foldDataTest, KNeighborsClassifier(), largeKparamGrid)

    KNNOptimismSmallK_Tuned = np.array(smallKtestErrors) - np.array(smallKtrainingErrors)
    KNNOptimismLargeK_Tuned = np.array(largeKtestErrors) - np.array(largeKtrainingErrors)

    allTestErrors["Small k (tuned)"] = smallKtestErrors
    allTestErrors["Large k (tuned)"] = largeKtestErrors
    optimism["Small k (tuned)"] = KNNOptimismSmallK_Tuned
    optimism["Large k (tuned)"] = KNNOptimismLargeK_Tuned

    print("KNN done tuning")

    #LDA
    ldaParamGrid = [
    {'solver': ['svd']}, 
    {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto']}
    ]
    _, ldaTrainingErrors, ldaTestErrors = doubleCV(foldDataTraining, foldDataTest, LinearDiscriminantAnalysis(), ldaParamGrid)
    allTestErrors["LDA (tuned)"] = ldaTestErrors

    LDAOptimism_Tuned = np.array(ldaTestErrors) - np.array(ldaTrainingErrors)
    optimism["LDA (tuned)"] = LDAOptimism_Tuned

    print("LDA Done tuning")

    #Random Forest
    rfParamGrid = {
    'n_estimators': [20,50, 100],          # number of trees
    'max_depth': [None, 20,50],         # control overfitting
    'max_features': ['sqrt'],  # feature selection per split
    'min_samples_split': [2, 5,10]          # min samples for splitting
    }
    _, rfTrainingErrors, rfTestErrors = doubleCV(foldDataTraining, foldDataTest, RandomForestClassifier(), rfParamGrid)
    allTestErrors["Random Forest (tuned)"] = rfTestErrors

    rfOptimism_Tuned = np.array(rfTestErrors) - np.array(rfTrainingErrors)
    optimism["Random Forest (tuned)"] = rfOptimism_Tuned

    print("randomForest Done tuning")

    
    """Evaluate range of hyperparameter values to get plot"""
    #rangeOfk = np.arange(1,10)
    #testErrorsForDifferentK = CVErrorVSHyperparam(KNN, 'n_neighbors', rangeOfk, foldDataTraining, foldDataTest)

    #plt.plot(rangeOfk, testErrorsForDifferentK, marker='o')
    #plt.xlabel("k")
    #plt.ylabel("CV Error")
    #plt.title("Unbiased CV Error vs k")
    #plt.grid(True)
    #plt.show()

    "Get plots"
    BoxPlot(allTestErrors)
    OptimismPlot(optimism)

main()