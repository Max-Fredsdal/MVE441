from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

class KNN:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean'):
        self.classifier = KNeighborsClassifier(
            n_neighbors= n_neighbors,
            weights= weights,
            algorithm= algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric
        )

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)


class RandomForest:
    def __init__(self, n_estimators=25, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=None):
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
    

class LDA:
    def __init__(self, solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=1e-4):
        """
        Wrapper for sklearn's LinearDiscriminantAnalysis.
        """
        self.classifier = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage,
            priors=priors,
            n_components=n_components,
            store_covariance=store_covariance,
            tol=tol
        )

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)