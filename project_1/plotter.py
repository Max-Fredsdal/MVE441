import matplotlib.pyplot as plt
import numpy as np

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




