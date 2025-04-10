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
def seaborn_boxplot(df: pd.DataFrame, xCol: str, yCol: str, title = "", hueCol = None, hueLabelMap = None, ylim=None) -> int:
    
    ax = sns.boxplot(data = df, x=xCol, y=yCol, hue=hueCol)
    
    if ylim:
        plt.ylim(ylim)
    plt.gca().set_axisbelow(True)
    plt.grid(axis='y', alpha=0.3 , linestyle="--")
    plt.title(title)

    if hueCol and hueLabelMap:
        handles, labels = ax.get_legend_handles_labels()
        custom_labels = [hueLabelMap.get(label, label) for label in labels]
        ax.legend(handles=handles, labels=custom_labels, title=hueCol)

    plt.show()

    return 0

def seaborn_confusion(cm, labels, title):
    # plt.figure(figsize=(6, 5))  # Adjust the size if needed
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    plt.show()

    return 0
