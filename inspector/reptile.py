import torch
import copy
from metalearner import MetaLearner

class Reptile(MetaLearner):
    def __init__(self, lr_inner=0.01, lr_outer=0.001, sgd_steps_inner=10):
        super().__init__(False, lr_inner, lr_outer, sgd_steps_inner)

    def compute_store_gradients(self, target, current):
        current_weights = dict(current.named_parameters())
        target_weights = dict(target.named_parameters())
        gradients = {
        name: (current_weights[name].data - target_weights[name].data) / (self.sgd_steps_inner * self.lr_inner) for name
        in target_weights}

        for name in current_weights:
            current_weights[name].grad.data = gradients[name]

    def train(self, model, train_data):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr_outer)
        self.init_grad(model)

        for i, task in enumerate(train_data.shuffled_set()):
            optimizer.zero_grad()

            inner_model = copy.deepcopy(model)
            inner_optim = torch.optim.SGD(inner_model.parameters(), lr=self.lr_inner)

            for _ in range(self.sgd_steps_inner):
                self.inner_train(inner_model, task, inner_optim)

            self.compute_store_gradients(inner_model, model)
            optimizer.step()

            if i % 5000 == 0:
                print("iteration:", i)