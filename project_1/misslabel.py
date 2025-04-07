import numpy as np
import pandas as pd


def misslabel_data(data, labels, p=0.1, pOut = []):
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

    

    new_data = data
    new_data[idx,0] = misslabel

    return new_data, idx