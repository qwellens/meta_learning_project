import numpy as np
import torch
import random
import scipy.signal as sp

class Wave:
    def __init__(self, K, noise_percent):
        # K as in K-shot learning
        self.amp = np.random.uniform(0.1, 5.0)
        self.phase = np.random.uniform(0, np.pi)
        self.K = K
        self.noise_percent = noise_percent
        self.mini_train = None
        self.mini_test = None

    def mini_train_set(self):
        if self.mini_train is None:
            x = np.random.uniform(-5, 5, (self.K, 1))
            y = self.f(x) + np.random.normal(0, self.noise_percent * self.amp, (self.K, 1))
            self.mini_train = (x, y)
        return torch.Tensor(self.mini_train[0]), torch.Tensor(self.mini_train[1])

    def mini_test_set(self):
        self.mini_test = x = np.random.uniform(-5, 5, (self.K, 1))
        y = self.f(x)
        return torch.Tensor(x), torch.Tensor(y)

    def eval_set(self, size=50):
        x = np.linspace(-5, 5, size).reshape((size, 1))
        y = self.f(x)
        return torch.Tensor(x), torch.Tensor(y)

class SineWave(Wave):
    def __init__(self, K=10, noise_percent=0):
        super().__init__(K, noise_percent)

    def f(self, x):
        return self.amp * np.sin(x + self.phase)


class SquareWave(Wave):
    def __init__(self, K=10, noise_percent=0):
        super().__init__(K, noise_percent)

    def f(self, x):
        return self.amp * sp.square(x + self.phase)


class SawtoothWave(Wave):
    def __init__(self, K=10, noise_percent=0):
        super().__init__(K, noise_percent)

    def f(self, x):
        return self.amp * sp.sawtooth(x + self.phase)


class DataGenerator:
    def __init__(self, function, size=50000, K=10, noise_percent=0):
        self.size = size
        self.K = K
        self.function = function
        self.noise_percent = noise_percent
        self.tasks = None

    def generate_set(self):
        config = {"sine": SineWave, "square": SquareWave, "sawtooth": SawtoothWave}
        self.tasks = tasks = [config[self.function](self.K, self.noise_percent) for _ in range(self.size)]
        return tasks

    def shuffled_set(self):
        if self.tasks is None:
            self.generate_set()
        return random.sample(self.tasks, len(self.tasks))