import numpy as np
import pandas as pd



def misslabel_data_simple(data, p):
    """
    data:   The label vector of the dataset 
    p:      The proportion of the data length that will
            be misslabeled, rounded down.'
    Return: the misslabeled data set and the corresponding indices that were changed.
    """
    data = np.array(data)
    lenData = np.shape(data)[0]
    nChanges = int(lenData*p)
    labels = np.unique(data)
    newData = data.copy()

    idx = np.random.choice([i for i in range(lenData)], size=nChanges, replace=False)
    for i in idx:
        newData[i] = np.random.choice(labels[labels != data[i]])

    if np.shape(newData) == np.shape(data):
        return newData
    else:
        raise Exception("Modified data shape is inconsistent with original shape!")


def misslabel_data_specified(data,target, labels, p=0.1, pOut = []):
    """
    data:   Data set with shape (nDatapoints, nFeatures) where 
            the first column is the label of the data.
    target: The label(s) to target with the misslabeling.
    labels: The available outcomes of a label change. 
            Example; misslabel variable y1 as one of 
            the following [y2, y3, y4].
    p:      The proportion of the data length that will
            be misslabeled, rounded down.
    pOut:   "Optional" specified proportions of misslabeling 
            outcomes. If left empty the misslabeling will be
            choosen uniformly from the provided labels.
    Return: the misslabeled data set
    """
    
    data = np.array(data)
    target_idx = np.where(data == target)[0]
    nChanges = round(len(target_idx) * p)
    idx = np.random.choice(target_idx, size=nChanges, replace=False)
    if pOut == []:
        misslabel = np.random.choice(labels, size=nChanges, replace=True)
    else:
        misslabel = np.random.choice(labels, size=nChanges, replace=True, p=pOut)

    

    newData = data.copy()
    newData[idx] = misslabel

    return newData
