import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style('whitegrid')

def mse_vs_sgdstep(data, filename):
    """
    data is format
    [[[y1(x=-5), ....,  y1(x=5)],
     [y2(x=-5), ....,  y2(x=5)],
     [y3(x=-5), ....,  y3(x=5)]]]
     """
    plt.figure()

    df = pd.DataFrame(data).melt(var_name='SGD steps', value_name='MSE on test data')
    df['SGD steps'].loc[df['SGD steps']==3] = 10
    df['SGD steps'].loc[df['SGD steps']==2] = 5
    df['SGD steps'].loc[df['SGD steps']==1] = 2
    df['SGD steps'].loc[df['SGD steps']==0] = 1
    ax = sns.pointplot(x='SGD steps', y='MSE on test data', data=df, ci=95, join=False)
    ax.set_ylim([60, 80]) # sin, sawtooth
    # ax.set_ylim([0, 300]) # square
    plt.title('Meta-Test Performance against SGD Steps taken')
    plt.xlabel('SGD steps')
    plt.ylabel('MSE on mini-test (in % of task amplitude)')
    plt.savefig(filename)

def mse_vs_k(data, filename):
    plt.figure()

    df = pd.DataFrame(data).mel==Int(var_name='K Values', value_name='MSE on test data')
    df['K_values'].loc[df['K_values']==3] = 10
    df['K_values'].loc[df['K_values']==2] = 5
    df['K_values'].loc[df['K_values']==1] = 2
    df['K_values'].loc[df['K_values']==0] = 1
    ax = sns.pointplot(x='K Values', y='MSE on test data', data=df, ci=95, join=False)
    ax.set_ylim([60, 80]) # sin, sawtooth
    # ax.set_ylim([0, 300]) # square
    plt.title('Meta-Test Performance against K Values used')
    plt.xlabel('K Values')
    plt.ylabel('MSE on mini-test (in % of task amplitude)')
    plt.savefig(filename)

def mse_vs_noise(data, filename):
    pass

if __name__ == '__main__':

    mode = 'MSE_v_K' # MSE_v_SGD, MSE_v_K, MSE_v_NOISE (currentlyeach alg, one plot per functional form)
    func = 'sine' # sine, sawtooth, square
    alg = 'maml' # maml, reptile, insect

    os.chdir(mode + '/' + func)
    first = True
    for file in sorted(os.listdir()): # iterate through data files
        if file != '.DS_Store':
            data = pd.io.parsers.read_csv(file, delimiter=',')
            if first:
                first = False
                SGD_10 = data.values[:,10]
            else:
                SGD_10 = np.column_stack((SGD_10, data.values[:,10]))
    print(SGD_10.shape)

    if mode == 'MSE_v_SGD': # Generate MSE vs SGD for each algorithm, for each functional form
        mse_vs_sgdstep(SGD_10, 'MSE vs SGD Step')
    elif mode == 'MSE_v_K': # Generate MSE vs K for each algorithm, for each functional form
        mse_vs_k(SGD_10, 'MSE vs K')
    elif mode == 'MSE_v_NOISE': # Generate MSE vs K for each noise level, for each algorithm, for each functional form
        mse_vs_noise(SGD_10, 'MSE vs Noise')

    os.chdir('../..')

