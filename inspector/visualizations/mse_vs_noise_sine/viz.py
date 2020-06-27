import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_style('whitegrid')
import os

directory = "relTaskDifficultyReptile"

file1 = os.path.join(directory,"sine.csv")
file2 = os.path.join(directory,"square.csv")
file3 = os.path.join(directory,"sawtooth.csv")
data1 = np.array(np.genfromtxt(file1, delimiter=','))
data2 = np.array(np.genfromtxt(file2, delimiter=','))
data3 = np.array(np.genfromtxt(file3, delimiter=','))

def mse_vs_sgdstep(data1, data2, data3, filename):
    """
    data is format
    [[y1(x=-5), ....,  y1(x=5)],
     [y2(x=-5), ....,  y2(x=5)],
     [y3(x=-5), ....,  y3(x=5)]]
     """
    df1 = pd.DataFrame(data1).melt(var_name="SGD steps", value_name="MSE on test data")
    df2 = pd.DataFrame(data2).melt(var_name="SGD steps", value_name="MSE on test data")
    df3 = pd.DataFrame(data3).melt(var_name="SGD steps", value_name="MSE on test data")
    plt.figure()
    ax = sns.lineplot(x="SGD steps", y="MSE on test data", data=df1, ci=95, label="sine")
    ax = sns.lineplot(x="SGD steps", y="MSE on test data", data=df3, ci=95, label="sawtooth")
    ax = sns.lineplot(x="SGD steps", y="MSE on test data", data=df2, ci=95, label="square")
    ax.set_ylim([0, 200])
    plt.title("Relative Task Difficulty for Reptile")
    plt.xlabel("meta-test SGD steps")
    plt.ylabel("MSE on mini-test (in % of task amplitude)")
    plt.savefig(filename)



mse_vs_sgdstep(data1,data2, data3, "reltaskdifReptile.png")