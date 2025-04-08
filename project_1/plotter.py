import matplotlib.pyplot as plt
import numpy as np

def BoxPlot(testErrors):
    labels = []
    means = []
    stds = []
    colors = []
    for name,errors in testErrors.items():
        string = f"{name}"
        mean = np.mean(errors)
        std = np.std(errors)

        means.append(mean)
        stds.append(std)
        labels.append(string)


    x = np.arange(len(labels))

    plt.figure()
    plt.boxplot(x, notch=True, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))
    plt.xticks(x, labels)
    plt.ylabel("Mean Test Error", fontsize=14)
    plt.title("Test Error per Model (with/without tuning)", fontsize=16)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()


def OptimismPlot(optimism):
    labels = []
    means = []
    stds = []
    for name,optimismForClassif in optimism.items():
        string = f"{name}"
        mean = np.mean(optimismForClassif)
        std = np.std(optimismForClassif)

        means.append(mean)
        stds.append(std)
        labels.append(string)
    
    x = np.arange(len(labels))

    plt.figure()
    plt.boxplot(x, notch=True, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))
    plt.xticks(x, labels)
    plt.ylabel("Mean optimism ", fontsize=14)
    plt.title("Optimism for each model", fontsize=16)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()




