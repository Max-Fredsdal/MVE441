import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
FORMAT INPUT DATAFRAME:
df({'X':, 'Y':, 'Z':})

ex:
labels = ['classifier','error','tuned']

classifier: which type of classifier as a string ex: "Random Forest"
tuned:      Boolean if the model is tuned or not
error:      The classification error

here the classifier will be displayed on the x axis, and the classification error
will be displayed on the y axis. the z will be the hue wether the model is tuned or not,
the hue is optional
"""

df = sns.load_dataset("titanic")

def seaborn_boxplot(df: pd.DataFrame, xCol: str, yCol: str, hueCol = None) -> int:
    
    sns.boxplot(data = df, x=xCol, y=yCol, hue=hueCol)
    plt.show()

    return 0


def BoxPlot(testErrors):


    labels = list(testErrors.keys())          
    errors = list(testErrors.values())

    plt.figure()
    plt.boxplot(errors, labels=labels)
    plt.ylabel("Test Error", fontsize=14)
    plt.title("Test Error per Model (with/without tuning)", fontsize=16)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()


def OptimismPlot(optimism):
    labels = list(optimism.keys())          
    optimismAll = list(optimism.values())

    plt.figure()
    plt.boxplot(optimismAll, labels=labels)
    plt.ylabel("Mean optimism ", fontsize=14)
    plt.title("Optimism for each model", fontsize=16)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()


# seaborn_boxplot(df,'class','age')
