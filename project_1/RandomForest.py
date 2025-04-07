from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
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
        if not hasattr(self.classifier, "estimators_"):
            raise NotFittedError("Model must be fitted before predicting.")
        return self.classifier.predict(X)
    
    def score(self, X, y):
        if not hasattr(self.classifier, "estimators_"):
            raise NotFittedError("Model must be fitted before scoring.")
        return self.classifier.score(X, y)
    
    def cross_val_score(self, X, y, cv=5):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.classifier, X, y, cv=kf)
        return np.mean(scores), np.std(scores)
    
    def nested_cross_val(self, X, y, param_grid, outer_splits=5, inner_splits=3):
        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
        outer_scores = []

        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner CV for tuning
            inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=self.classifier,
                                       param_grid=param_grid,
                                       cv=inner_cv,
                                       scoring='accuracy')
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            outer_scores.append(accuracy_score(y_test, y_pred))

        return np.mean(outer_scores), np.std(outer_scores)
        