import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from copy import deepcopy


class MLP:
    # def __init__(self, x, y, hidden, n=0.1, max_it=10000, sim_annealing_iterations=300):
    def __init__(self, x, y, hidden, n=0.1, max_it=2500):
        self.max_it = max_it
        self.n = n
        # self.classes = {}
        self.idx_to_class = {v: k for v, k in enumerate(y.astype('category').cat.categories.tolist())}
        self.class_vector = np.identity(len(self.idx_to_class))
        self.class_to_idx = {k: v for v, k in enumerate(y.astype('category').cat.categories.tolist())}
        self.erro_var = []
        self.hidden = hidden
        # self.sim_annealing_iterations = sim_annealing_iterations

        # print(self.idx_to_class)
        # print(self.class_to_idx)
        # print(self.class_vector)
        # print()

        # Append bias column
        my_x = np.matrix(x, copy=True)

        my_y = []

        for out in y:
            value = self.class_to_idx[out]
            my_y.append(self.class_vector[value])

        my_y = np.matrix(my_y)

        self.w = None

        self.random_start(my_x, my_y)

        # print(self.w)

        self.train(my_x, my_y)

    def predict(self, x):
        my_x = deepcopy(x)

        for i in range(len(self.w)):
            my_x = np.append(
                np.ones((my_x.shape[0], 1)),
                my_x,
                axis=1
            )

            my_x = np.dot(my_x, self.w[i])
            my_x = 1 / (1 + np.exp(-my_x))

        p = pd.DataFrame({'predicted': np.asarray(np.argmax(my_x, axis=1).transpose())[0, :]})
        p = p.replace(self.idx_to_class)

        return p

    def random_start(self, x, y):
        self.w = []  # List of weight matrices

        # Build weight matrix for the input with the first hidden layer
        temp_shape = (x.shape[1] + 1, self.hidden[0])
        self.w.append(np.matrix(np.random.rand(temp_shape[0], temp_shape[1]) * 0.1))

        # Build weight matrices for hidden layers
        for i in range(1, len(self.hidden)):
            temp_shape = (self.hidden[i - 1] + 1, self.hidden[i])
            self.w.append(np.matrix(np.random.rand(temp_shape[0], temp_shape[1]) * 0.1))

        # Build weight matrix for the last hidden layer and the output
        temp_shape = (self.hidden[-1] + 1, y.shape[1])
        self.w.append(np.random.rand(temp_shape[0], temp_shape[1]) * 0.1)

    def train(self, x, y):
        best_w = deepcopy(self.w)
        max_precision = 0.0  # Stores the number of misclassified observations of the best iteration
        max_it = self.max_it  # Count the number of iterations
        y_class = np.argmax(y, axis=1)
        # no_update_iterations = 0

        while max_it:
            # no_update_iterations += 1

            # if no_update_iterations >= self.sim_annealing_iterations:
            #     self.random_start(x, y)  # Simulated Annealing
            #     no_update_iterations = 0

            max_it -= 1
            in_layer = [np.copy(x)]
            delta_net = deepcopy(self.w)

            # Compute MLP output
            for i in range(len(self.w)):
                in_layer[i] = np.append(
                    np.ones((in_layer[i].shape[0], 1)),
                    in_layer[i],
                    axis=1
                )

                in_layer.append(in_layer[i] * self.w[i])
                in_layer[i + 1] = 1 / (1 + np.exp(-in_layer[i + 1]))  # Sigmoid function

            # Compute error at output layer
            error = deepcopy(in_layer)

            actual_precision = np.argmax(error[-1], axis=1)

            actual_precision = precision_score(
                np.asarray(y_class.transpose())[0, :],
                np.asarray(actual_precision.transpose())[0, :],
                average='micro'
            )

            self.erro_var.append(np.fabs(error[-1] - y).sum())

            if actual_precision > max_precision:
                max_precision = actual_precision
                best_w = deepcopy(self.w)

                # no_update_iterations = 0

                # print(self.max_it - max_it)

                if max_precision == 1.0:
                    break

            for i in range(1, len(error)):
                error[i] = np.multiply(error[i], (1 - error[i]))

            error[-1] = np.multiply(y - in_layer[-1], error[-1])

            delta_net[-1] = (error[-1].transpose() * self.n * in_layer[-2]).transpose()

            for i in range(len(error) - 2, 0, -1):
                # print(error[i][:, 1:])
                error[i] = np.multiply(error[i + 1] * self.w[i].transpose(), error[i])[:, 1:]

                delta_net[i - 1] = error[i - 1].transpose() * self.n * error[i]

            for i in range(len(self.w)):
                self.w[i] += delta_net[i]

            # print(delta_net)
            # exit()

        self.w = deepcopy(best_w)
