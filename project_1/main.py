from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from KNN import KNN
import matplotlib.pyplot as plt

def doubleCV(foldDataTraining, foldDataTest, model, paramGrid):
    numberOfFolds = len(foldDataTraining)
    #For each fold, perform inner CV
    outerScores = [] #Accuracy, essentially same as 1-testerror... (can remove)
    trainingErrors = []
    testErrors = []

    for k in range(numberOfFolds):

        train = foldDataTraining[k]
        test = foldDataTest[k]

        xTrain = train[:, 1:]
        yTrain = train[:, 0]
        xTest = test[:, 1:]
        yTest = test[:, 0]

        innerCV = KFold(n_splits=3, shuffle = True, random_state=42)

        gridSearch = GridSearchCV(
            estimator= model,
            param_grid=paramGrid,
            cv = innerCV,
            scoring='accuracy'
        )

        gridSearch.fit(xTrain,yTrain)

        bestModel = gridSearch.best_estimator_
        yPred = bestModel.predict(xTest)

        outerScores.append(accuracy_score(yTest, yPred))
    
        #Training Error
        yPredTraining = bestModel.predict(xTrain)
        trainingErrorForFold = np.mean(np.where(yPredTraining != yTrain,1,0))
        trainingErrors.append(trainingErrorForFold)

        #Test Error i.e cross validation error after tuning
        testErrorForFold = np.mean(np.where(yPred != yTest,1,0))
        testErrors.append(testErrorForFold)
    
    
    return outerScores,trainingErrors,testErrors



def main():
    # Load the data
    df = pd.read_csv('data/Numbers.txt', delimiter=' ')

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    outerCV = KFold(n_splits=5, shuffle=True, random_state=42)
    foldDataTraining = []
    foldDataTest = []

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
    largeKClassifier = KNN(n_neighbors=100)

    smallKtrainingErrorsNoTuning,smallKtestErrorsNoTuning,_,_ = smallKClassifier.Evaluation(foldDataTraining,foldDataTest)
    largeKtrainingErrorsNoTuning,largeKtestErrorsNoTuning,_,_ = largeKClassifier.Evaluation(foldDataTraining,foldDataTest)

    #Optimism (note: per fold)
    KNNOptimismSmallK = np.array(smallKtrainingErrorsNoTuning) - np.array(smallKtestErrorsNoTuning)
    KNNOptimismLargeK = np.array(largeKtrainingErrorsNoTuning) - np.array(largeKtestErrorsNoTuning)


    """Evaluation with tuning"""

    #KNN
    smallKparamGrid = {'n_neighbors': list(range(1,11))} #k = 1,2,...10
    largeKparamGrid = {'n_neighbors': list(range(50, 151, 10))} #k = 50,60..150 
    smallKouterScores,smallKtrainingErrors,smallKtestErrors = doubleCV(foldDataTraining, foldDataTest, KNeighborsClassifier(), smallKparamGrid)
    largeKouterScores,largeKtrainingErrors,largeKtestErrors = doubleCV(foldDataTraining, foldDataTest, KNeighborsClassifier(), largeKparamGrid)

    """Box plot comparison"""
    means = [
    np.mean(smallKtestErrorsNoTuning),
    np.mean(smallKtestErrors),
    np.mean(largeKtestErrorsNoTuning),
    np.mean(largeKtestErrors)
    ]
    stds = [
    np.std(smallKtestErrorsNoTuning),
    np.std(smallKtestErrors),
    np.std(largeKtestErrorsNoTuning),
    np.std(largeKtestErrors)
    ]

    labels = [
    "Small k (KNN) (no tuning)",
    "Small k (KNN) (tuned)",
    "Large k (KNN) (no tuning)",
    "Large k (KNN) (tuned)"
    ]
    x = np.arange(len(labels))

    plt.figure()
    plt.errorbar(x, means, yerr=stds,fmt='o', elinewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("Mean Test Error")
    plt.title("Test Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

main()