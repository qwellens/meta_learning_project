import matplotlib.pyplot as plt
import numpy as np

def output_and_truth_points(input, actual, predicted, grad_steps, function):
    plt.figure(1)
    plt.plot(input.flatten(), predicted[grad_steps-1].flatten(), 'xr', label="predicted")
    plt.plot(input.flatten(), actual.flatten(), 'og', label="ground truth")
    x = np.arange(-5, 5, 0.01)
    plt.plot(x, function(x), label="underlying function")
    plt.title("Model predictions and ground truth")
    plt.legend()

def loss_vs_grad_steps(losses):
    plt.figure(2)
    plt.plot(losses)
    plt.title("MSE of test points after x gradient updates on same train points ")
    plt.xlabel("Number of gradient updates on the same train points")
    plt.ylabel("MSE oftest points")

def output_and_truth_function(x, actual, predicted, grad_steps):
    plt.figure(3)
    plt.plot(x.flatten(), actual.flatten(), 'g',  label="ground truth")
    plt.plot(x.flatten(), predicted[grad_steps-1].flatten(), 'r',  label="predicted")
    plt.title("Model predictions and ground truth")
    plt.legend()
