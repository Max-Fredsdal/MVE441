import numpy as np
import pandas as pd



def misslabel_data_simple(data, p):
    """
    data:   Data set with shape (nDatapoints, nFeatures) where 
            the first column is the label of the data. 
    p:      The proportion of the data length that will
            be misslabeled, rounded down.'
    Return: the misslabeled data set and the corresponding indices that were changed.
    """
    lenData = np.shape(data)[0]
    nChanges = lenData*p
    labels = np.unique(data[:,0])
    newData = data

    idx = np.random.choice([i for i in range(lenData)], size=nChanges, replace=False)
    for i in idx:
        newData[i,0] = np.random.choice(labels[labels != data[i,0]])

    return newData, idx


def misslabel_data_specified(data, labels, p=0.1, pOut = []):
    """
    data:   Data set with shape (nDatapoints, nFeatures) where 
            the first column is the label of the data.
    labels: The available outcomes of a label change. 
            Example; misslabel variable y1 as one of 
            the following [y2, y3, y4].
    p:      The proportion of the data length that will
            be misslabeled, rounded down.
    pOut:   "Optional" specified proportions of misslabeling 
            outcomes. If left empty the misslabeling will be
            choosen uniformly from the provided labels.
    Return: the misslabeled data set and the corresponding indices that were changed.
    """
    if len(pOut) != len(labels):
        raise Exception("The length of the available misslabelings does " \
                        "not mach the length of proportions of misslabelings")

    lenData = np.shape(data)[0]
    nChanges = lenData*p
    idx = np.random.choice([i for i in range(lenData)], size=nChanges, replace=False)
    if pOut == []:
        misslabel = np.random.choice(labels, size=nChanges, replace=True)
    else:
        misslabel = np.random.choice(labels, size=nChanges, replace=True, p=pOut)

    

    newData = data
    newData[idx,0] = misslabel

    return newData, idx
