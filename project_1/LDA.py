from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np


class LDA:
    def __init__(self, solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=1e-4):
        """
        Wrapper for sklearn's LinearDiscriminantAnalysis.
        """
        self.lda = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage,
            priors=priors,
            n_components=n_components,
            store_covariance=store_covariance,
            tol=tol
        )

    def fit(self, X, y):
        self.lda.fit(X, y)

    def predict(self, X):
        if not hasattr(self.lda, "coef_"):
            raise NotFittedError("Model must be fitted before predicting.")
        return self.lda.predict(X)

    def score(self, X, y):
        if not hasattr(self.lda, "coef_"):
            raise NotFittedError("Model must be fitted before scoring.")
        return self.lda.score(X, y)

    def nested_cross_val(X, y, model, param_grid, outer_splits=5, inner_splits=3):
        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
        outer_scores = []

        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner CV for tuning
            inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=param_grid,
                                       cv=inner_cv,
                                       scoring='accuracy')
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            outer_scores.append(accuracy_score(y_test, y_pred))

        return np.mean(outer_scores), np.std(outer_scores)