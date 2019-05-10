import torch
import torch.nn.functional as F
from metalearner import MetaLearner
import copy

class Insect(MetaLearner):
    def __init__(self, lr_inner=0.01, lr_outer=0.001, sgd_steps_inner=10, averager=50):
        super().__init__(False, lr_inner, lr_outer, sgd_steps_inner)
        self.average_weights = {}
        self.averager_size = averager
        self.test_tasks = []

    def init_averager(self, model):
        for name, param in model.named_parameters():
            self.average_weights[name] = torch.zeros_like(param)

    def load_averaged_weights(self, model):
        for name, param in model.named_parameters():
            param.data = self.average_weights[name]

    def train(self, model, train_data):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr_outer)
        self.init_grad(model)
        self.init_averager(model)

        for i, task in enumerate(train_data.shuffled_set()):
            self.test_tasks.append(task)
            inner_model = copy.deepcopy(model)
            inner_optim = torch.optim.SGD(inner_model.parameters(), lr=self.lr_inner)

            for _ in range(self.sgd_steps_inner):
                self.inner_train(inner_model, task, inner_optim)

            for name, param in inner_model.named_parameters():
                self.average_weights[name] = self.average_weights[name] + param / self.averager_size

            if i % self.averager_size == 0 and i > 0:
                self.load_averaged_weights(model)

                for task in self.test_tasks:
                    x, y = task.mini_test_set()
                    predicted = model(x)
                    loss = F.mse_loss(predicted, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                self.test_tasks = []
                self.init_averager(model)

            if i % 5000 == 0:
                print("iteration:", i)