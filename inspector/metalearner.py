import torch
import torch.nn.functional as F

class MetaLearner():
    def __init__(self, higher_order=False, lr_inner=0.01, lr_outer=0.001, sgd_steps_inner=10):
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.sgd_steps_inner = sgd_steps_inner
        self.higher_order = higher_order

    def inner_train(self, model, task, optimizer):
        optimizer.zero_grad()
        x, y = task.mini_train_set()
        predicted = model(x)
        loss = F.mse_loss(predicted, y)
        loss.backward()
        optimizer.step()

    def init_grad(self, model):
        for param in model.parameters():
            param.grad = torch.zeros_like(param)
