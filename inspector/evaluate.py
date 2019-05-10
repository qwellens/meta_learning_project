import numpy as np
import torch
import torch.nn.functional as F
import copy
from data import DataGenerator

def inner_train(model, task, optimizer):
    optimizer.zero_grad()
    x, y = task.mini_train_set()
    predicted = model(x)
    loss = F.mse_loss(predicted, y)
    loss.backward()
    optimizer.step()

def evaluate(base_model, function, K, noise_percent, SGD_steps, inner_lr, runs=100):
    all_losses = []
    test_tasks = DataGenerator(function, runs, K, noise_percent).shuffled_set()

    for test_task in test_tasks:
        xeval, yeval = test_task.eval_set(size=100)

        model = copy.deepcopy(base_model)
        optim = torch.optim.SGD(model.parameters(), lr=inner_lr)
        losses = []
        
        predicted = model(xeval)
        losses.append(F.mse_loss(predicted, yeval).item())

        for i in range(SGD_steps):
            inner_train(model, test_task, optim)
            predicted = model(xeval)
            losses.append(F.mse_loss(predicted, yeval).item())

        norm_losses = np.array(losses) / test_task.amp * 100
        all_losses.append(norm_losses)

    return np.array(all_losses)
