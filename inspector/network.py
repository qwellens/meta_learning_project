import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 40)
        self.hidden2 = nn.Linear(40, 40)
        self.out = nn.Linear(40, 1)

    def set_attr(self, name, param):
        # BIGGEST HACK EVER..
        if name == "hidden1.weight":
            self.hidden1.weight = param
        if name == "hidden1.bias":
            self.hidden1.bias = param
        if name == "hidden2.weight":
            self.hidden2.weight = param
        if name == "hidden2.bias":
            self.hidden2.bias = param
        if name == "out.weight":
            self.out.weight = param
        if name == "out.bias":
            self.out.bias = param

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)