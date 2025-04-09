from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from KNN import KNN
from KNN import LDA
from KNN import RandomForest
import matplotlib.pyplot as plt
from plotter import BoxPlot, OptimismPlot, seaborn_boxplot
from misslabel import misslabel_data_simple, misslabel_data_specified
from helper import Evaluation, doubleCV, CVErrorVSHyperparam

def main():
    # Load the data
    df = pd.read_csv('data/Numbers.txt', delimiter=' ')
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    
    # train test split
    X_t, X_tst, y_t, y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)

    # y_t = misslabel_data_simple(y_t,0.4)
    

    outerCV = KFold(n_splits=10, shuffle=True)
    foldDataTraining = []
    foldDataTest = []
    #Plot data
    allTestErrors = {}
    optimism = {}
    bestModels = {}
    finalTestErrors = {}

    #Get the folded training and test data
    for train_idx, test_idx in outerCV.split(X_t, y_t):
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
    kNNClassifier = KNN(n_neighbors=5)
    kNNtrainingErrorsNoTuning, kNNtestErrorsNoTuning ,_,_ = Evaluation(kNNClassifier,foldDataTraining,foldDataTest)
    allTestErrors["KNN (no tuning)"] = kNNtestErrorsNoTuning

    kNNOptimism = np.array(kNNtestErrorsNoTuning) - np.array(kNNtrainingErrorsNoTuning)
    optimism["KNN (no tuning)"] = kNNOptimism

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
    
    kNNparamGrid = {'n_neighbors': list(range(1,15))} #k = 1,2,...10 --> Flexible

    kNNtrainingErrors, kNNtestErrors, kNNBestModel, df_kNNTuningResults = doubleCV(foldDataTraining, foldDataTest, KNeighborsClassifier(), kNNparamGrid)

    df_kNNTuningResults["Test error"] = 1 - df_kNNTuningResults["mean_test_score"]
    df_kNNTuningResults = df_kNNTuningResults.rename(columns={"param_n_neighbors": "Number of neighbors"})
    df_kNNTuningResults.to_csv("data/kNNTuningResults_task1.csv")
    kNNOptimism_Tuned = np.array(kNNtestErrors) - np.array(kNNtrainingErrors)
    
    allTestErrors["KNN (tuned)"] = kNNtestErrors
    optimism["KNN (tuned)"] = kNNOptimism_Tuned

    print("KNN done tuning")

    #LDA
    ldaParamGrid = [
    {'solver': ['svd']}, 
    {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto']}
    ]
    ldaTrainingErrors, ldaTestErrors, ldaBestModels, df_ldaTuningResults = doubleCV(foldDataTraining, foldDataTest, LinearDiscriminantAnalysis(), ldaParamGrid)
    
    df_ldaTuningResults["Test error"] = 1 - df_ldaTuningResults["mean_test_score"]
    # df_ldaTuningResults = df_ldaTuningResults.rename(columns={"param_n_neighbors": "Number of neighbors"})
    df_ldaTuningResults.to_csv("data/ldaTuningResults_task1.csv")

    allTestErrors["LDA (tuned)"] = ldaTestErrors

    LDAOptimism_Tuned = np.array(ldaTestErrors) - np.array(ldaTrainingErrors)
    optimism["LDA (tuned)"] = LDAOptimism_Tuned
    bestModels["LDA (tuned)"] = ldaBestModels

    print("LDA Done tuning")

    # #Random Forest
    rfParamGrid = {
    'n_estimators': [20,50, 100],          # number of trees
    'max_depth': [None, 20,50],         # control overfitting
    'max_features': ['sqrt'],  # feature selection per split
    'min_samples_split': [2, 5,10]          # min samples for splitting
    }
    rfTrainingErrors, rfTestErrors, rfBestModels, df_rfTuningResults = doubleCV(foldDataTraining, foldDataTest, RandomForestClassifier(), rfParamGrid)
    
    df_rfTuningResults["Test error"] = 1 - df_rfTuningResults["mean_test_score"]
    # df_rfTuningResults = df_rfTuningResults.rename(columns={"param_n_neighbors": "Number of neighbors"})
    df_rfTuningResults.to_csv("data/rfTuningResults_task1.csv")
    
    allTestErrors["Random Forest (tuned)"] = rfTestErrors

    rfOptimism_Tuned = np.array(rfTestErrors) - np.array(rfTrainingErrors)
    optimism["Random Forest (tuned)"] = rfOptimism_Tuned
    bestModels["Random Forest (tuned)"] = rfBestModels

    print("randomForest Done tuning")



    ### Converting from dictionaries to pandas dataframe:
    records = []

    for classifier_name in allTestErrors:
        test_errors = allTestErrors[classifier_name]
        optimism_values = optimism.get(classifier_name, [None] * len(test_errors))

        tuned_flag = 'Yes' if "(tuned)" in classifier_name else 'No'

        name = classifier_name.replace(" (no tuning)", "").replace(" (tuned)", "")

        for test_err, opt in zip(test_errors, optimism_values):
            records.append({
                "Classifier": name,
                "Test error": test_err,
                "Optimism": opt,
                "Tuned": tuned_flag
            })

    # Create final DataFrame
    data = pd.DataFrame(records)
    data.to_csv("data/task1_data.csv")
    
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

    seaborn_boxplot(df_kNNTuningResults,"Number of neighbors","Test error") # Plot parameter performance for knn when tuning
   
    # print(data)
    seaborn_boxplot(data,'Classifier','Test error','Tuned')
    # BoxPlot(allTestErrors)
    # OptimismPlot(optimism)
    
    # calculate the final test error for each model
    for name, model in bestModels.items():
        y_pred = model.predict(X_tst)
        finalTestError = np.mean(np.where(y_pred != y_tst, 1, 0))
        finalTestErrors[name] = finalTestError
    
    


main()