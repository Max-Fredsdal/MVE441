from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import numpy as np


class RandomForest:
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='auto', bootstrap=True, random_state=None):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)
    
    def score(self, X, y):
        return self.classifier.score(X, y)
    
    def Error(self, foldDataTraining, foldDataTest):
        training_errors = []
        test_errors = []
        for k in range(len(foldDataTraining[0, 0, :])):
            xTrain = foldDataTraining[:, 1:, k]
            yTrain = foldDataTraining[:, 0, k]

            self.classifier.fit(xTrain, yTrain)
            yPred = self.classifier.predict(xTrain)

            training_errors.append(np.mean(np.where(yPred != yTrain, 1, 0)))

        for k in range(len(foldDataTest[0, 0, :])):
            xTest = foldDataTest[:, 1:, k]
            yTest = foldDataTest[:, 0, k]

            yPred = self.classifier.predict(xTest)

            test_errors.append(np.mean(np.where(yPred != yTest, 1, 0)))

        return np.mean(training_errors), np.mean(test_errors)
    
    def cross_val_score(self, X, y, cv=5):
        kf = KFold(n_splits=cv)
        scores = cross_val_score(self.classifier, X, y, cv=kf)
        return scores
    
    def confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)
        