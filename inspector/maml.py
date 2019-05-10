import torch
import torch.nn.functional as F
from metalearner import MetaLearner
from network import Net

class MAML(MetaLearner):
    def __init__(self, lr_inner=0.01, lr_outer=0.001, sgd_steps_inner=1):
        super().__init__(True, lr_inner, lr_outer, sgd_steps_inner)

    def inner_train(self, model, task):
        x, y = task.mini_train_set()
        predicted = model(x)
        loss = F.mse_loss(predicted, y)
        loss.backward(create_graph=True, retain_graph=True)

    def train(self, model, train_data):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr_outer)
        self.init_grad(model)

        for i, task in enumerate(train_data.shuffled_set()):
            inner_model = Net()

            for name, param in model.named_parameters():
                inner_model.set_attr(name, param)

            for _ in range(self.sgd_steps_inner):
                self.inner_train(inner_model, task)
                for name, param in inner_model.named_parameters():
                    inner_model.set_attr(name, torch.nn.Parameter(param - self.lr_inner * param.grad))

            x, y = task.mini_test_set()
            predicted = inner_model(x)
            loss = F.mse_loss(predicted, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

