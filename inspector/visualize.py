import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style('whitegrid')

def mse_vs_sgdstep(data, filename):
    """
    data is format
    [[y1(x=-5), ....,  y1(x=5)],
     [y2(x=-5), ....,  y2(x=5)],
     [y3(x=-5), ....,  y3(x=5)]]
     """
    df = pd.DataFrame(data).melt(var_name="SGD steps", value_name="MSE on test data")
    plt.figure()
    ax = sns.lineplot(x="SGD steps", y="MSE on test data", data=df, ci=95)
    plt.title("Meta-Test Performance")
    plt.xlabel("SGD steps")
    plt.ylabel("MSE on mini-test (in % of task amplitude)")
    plt.savefig(filename)