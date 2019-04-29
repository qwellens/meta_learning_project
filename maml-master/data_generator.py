import os, sys
import csv
import random
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from math import pi

# filename = 'sinusoids.csv'
# spacing = 0.1
# num_samples = 100000 # number of sinusoids
# sinusoids = list() # [[y0], [y1], [y2]] -> y-values for each sinusoid

# for _ in range(num_samples):
#     f = random.uniform(0.5, 10)
#     b = random.uniform(0, 2*pi)
#     a = random.uniform(0.1, 10)

#     x = np.arange(0, 2*pi, spacing)
#     y = a*np.sin(f*x + b)

#     sinusoids.append([f, a, b]) # parameters: every odd row
#     sinusoids.append(y) # y-values: every even row

# file = open(filename, 'w')
# writer = csv.writer(file)
# writer.writerows(sinusoids)
# file.close()

# plt.scatter(x, y, s=3)
# plt.show()

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, function, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        self.function = function
        print('Function: ' + function)

        self.generate = self.generate_batch
        self.amp_range = config.get('amp_range', [0.1, 5.0])
        self.phase_range = config.get('phase_range', [0, np.pi])
        self.input_range = config.get('input_range', [-5.0, 5.0])
        self.dim_input = 1
        self.dim_output = 1

    def generate_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)

            # VARIOUS FUNCTION TYPES
            if self.function == 'sinusoid':
                outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
            elif self.function == 'square':
                outputs[func] = amp[func] * sp.square(init_inputs[func]-phase[func])
            elif self.function == 'triangular':
                outputs[func] = amp[func] * sp.sawtooth(init_inputs[func]-phase[func])

        return init_inputs, outputs, amp, phase

    def generate_full_function(self, function):
        x = np.arange(self.input_range[0], self.input_range[1], 0.01)
        x = x.reshape((1, x.size, 1))
        y = function(x)
        return x, y


















