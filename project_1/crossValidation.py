import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# Vet ej om vi får använda en inbyggd KFold algorithm

filepath = 'data/Numbers.txt'
destination = 'data/cFold'

def k_fold(path, nFolds, saveToPath, classifier):
    df = pd.read_csv(path, delimiter=' ')

    kf = KFold(n_splits = nFolds, shuffle = True, random_state=42)
    i = 0
    for train, test in kf.split(df):
        df.iloc[train,:].to_csv(f"{saveToPath}_train_{i}.csv")
        df.iloc[test,:].to_csv(f"{saveToPath}_test_{i}.csv")
        i += 1
        classifier.fit()

def create_foldData():
