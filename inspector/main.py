from network import Net
from insect import Insect
from maml import MAML
from reptile import Reptile
from data import DataGenerator
from evaluate import evaluate
from visualize import mse_vs_sgdstep
from scipy.stats import sem, t
from scipy import mean

import os
import numpy as np
import torch

META_LEARNER = "reptile"    #choose between: "reptile", "maml", or "insect"
IDENTIFIER = "a"            #give it a letter name (see spreadsheet)

#meta-train
FUNCTION_TRAIN = "sine"     #choose between: "sine", "square", or "sawtooth"
K_TRAIN = 2                #how many training samples per task
SGD_STEPS_TRAIN = 10        #how many sgd steps to take to get the fast/specialized/inner model weights
NOISE_PERCENT_TRAIN = 0     #noise applied on mini_train_set of tasks used during meta-training
ITERATIONS_TRAIN = 100000    #number of iterations (i.e. tasks because batch size is 1) to use
OUTER_LR_TRAIN = 0.001      #outer loop learning rate (Adam)
INNER_LR_TRAIN = 0.01       #inner loop learning rate (SGD)
AVERAGER_SIZE_TRAIN = 50    #relevant for insect only, basically a batch size

#meta-test
FUNCTION_TEST = "sine"     #choose between: "sine", "square", or "sawtooth"
K_TEST = 2                #how many training samples per task
SGD_STEPS_TEST = 10        #how many sgd steps to take to get the fast/specialized/inner model weights
NOISE_PERCENT_TEST = 0     #noise applied on mini_train_set of tasks used during meta-training
RUNS_TEST = 600            #number of tasks to test on
INNER_LR_TEST = 0.01       #inner loop learning rate (SGD)

JOB_NAME = IDENTIFIER + "_alg-" + str(META_LEARNER) + "_fun-" + str(FUNCTION_TRAIN) + "_k-" + str(K_TRAIN) + "_sgd-" + str(SGD_STEPS_TRAIN) + \
           "_noise-" + str(NOISE_PERCENT_TRAIN) + "_iterations-" + str(ITERATIONS_TRAIN) + "_olr-" + str(OUTER_LR_TRAIN) + \
           "_ilr-" + str(INNER_LR_TRAIN) + "_avg-" + str(AVERAGER_SIZE_TRAIN)

JOB_NAME = os.path.join("results", JOB_NAME)

TEST_JOB_NAME = "test" + "_fun-" + str(FUNCTION_TEST) + "_k-" + str(K_TEST) + "_sgd-" + str(SGD_STEPS_TEST) + \
                "_noise-" + str(NOISE_PERCENT_TEST) + "_runs-" + str(RUNS_TEST) + "_ilr-" + str(INNER_LR_TEST)
TEST_JOB_NAME = os.path.join(JOB_NAME, TEST_JOB_NAME)

REUSE_TRAINED_MODEL = False

def get_ci(data):
    confidence = 0.95

    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    start = m - h
    end = m + h

    return start, m, end

def test(model):
    if os.path.exists(TEST_JOB_NAME):
        raise AssertionError("Test job name already exists")
    else:
        os.mkdir(TEST_JOB_NAME)
        f = open(os.path.join(TEST_JOB_NAME, "test_params.txt"), 'w')
        f.write("META_LEARNER " + str(META_LEARNER) + '\n')
        f.write("FUNCTION_TEST " + str(FUNCTION_TEST) + '\n')
        f.write("K_TEST " + str(K_TEST) + '\n')
        f.write("SGD_STEPS_TEST " + str(SGD_STEPS_TEST) + '\n')
        f.write("NOISE_PERCENT_TEST " + str(NOISE_PERCENT_TEST) + '\n')
        f.write("RUNS_TEST " + str(RUNS_TEST) + '\n')
        f.write("INNER_LR_TEST " + str(INNER_LR_TEST) + '\n')
        f.close()

    eval_data = evaluate(model, FUNCTION_TEST, K_TEST, NOISE_PERCENT_TEST, SGD_STEPS_TEST, INNER_LR_TEST, RUNS_TEST)
    # export numpy arrays to csv
    np.savetxt(os.path.join(TEST_JOB_NAME, "eval_results.csv"), eval_data, delimiter=",")
    mse_vs_sgdstep(eval_data, os.path.join(TEST_JOB_NAME, "mse_vs_sgd.png"))

    lo_ci, mean, hi_ci = get_ci(eval_data[:, 10])
    f = open(os.path.join(TEST_JOB_NAME, "test_results.txt"), 'w')
    f.write("MEAN MSE (after 10 sgd) " + str(mean) + '\n')
    f.write("95 CI LOW" + str(lo_ci) + '\n')
    f.write("95 CI HIGH" + str(hi_ci) + '\n')
    f.close()


def main():
    if os.path.exists(JOB_NAME):
        raise AssertionError("Job name already exists")
    else:
        os.mkdir(JOB_NAME)
        f = open(os.path.join(JOB_NAME, "train_params.txt"), 'w')
        f.write("META_LEARNER " + str(META_LEARNER) + '\n')
        f.write("FUNCTION " + str(FUNCTION_TRAIN) + '\n')
        f.write("K_TRAIN " + str(K_TRAIN) + '\n')
        f.write("SGD_STEPS_TRAIN " + str(SGD_STEPS_TRAIN) + '\n')
        f.write("NOISE_PERCENT_TRAIN " + str(NOISE_PERCENT_TRAIN) + '\n')
        f.write("ITERATIONS_TRAIN " + str(ITERATIONS_TRAIN) + '\n')
        f.write("OUTER_LR_TRAIN " + str(OUTER_LR_TRAIN) + '\n')
        f.write("INNER_LR_TRAIN " + str(INNER_LR_TRAIN) + '\n')
        f.write("AVERAGER_SIZE_TRAIN " + str(AVERAGER_SIZE_TRAIN) + '\n')
        f.close()

    model = Net()
    if META_LEARNER == "reptile":
        learning_alg = Reptile(lr_inner=INNER_LR_TRAIN, lr_outer=OUTER_LR_TRAIN, sgd_steps_inner=SGD_STEPS_TRAIN)
    elif META_LEARNER == "maml":
        learning_alg = MAML(lr_inner=INNER_LR_TRAIN, lr_outer=OUTER_LR_TRAIN, sgd_steps_inner=SGD_STEPS_TRAIN)
    else:
        learning_alg = Insect(lr_inner=INNER_LR_TRAIN, lr_outer=OUTER_LR_TRAIN, sgd_steps_inner=SGD_STEPS_TRAIN, averager=AVERAGER_SIZE_TRAIN)
    meta_train_data = DataGenerator(function=FUNCTION_TRAIN, size=ITERATIONS_TRAIN, K=K_TRAIN, noise_percent = NOISE_PERCENT_TRAIN)
    learning_alg.train(model, meta_train_data)

    torch.save(model, os.path.join(JOB_NAME, "trained_model.pth"))
    test(model)


if REUSE_TRAINED_MODEL:
    model = torch.load(os.path.join(JOB_NAME, "trained_model.pth"))
    test(model)
else:
    main()

