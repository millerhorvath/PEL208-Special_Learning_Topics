import numpy as np


class Perceptron:
    def __init__(self, x, y, n=0.01, max_it=10000):
        """
        :type x: np.array
        :param x:
        :type y: np.array
        :param y:
        """
        self.max_it = max_it
        self.n = n
        my_x = np.copy(x)

        # Append bias column
        my_x = np.append(
            np.matrix(np.zeros((x.shape[0], 1)) + 1.0),
            my_x, axis=1
        )
        temp_shape = (my_x.shape[1], 1)

        # Set up initial weights (using random numbers close to 0.0)
        self.w = np.matrix(np.zeros(shape=temp_shape) + np.random.rand(temp_shape[0], temp_shape[1]) - 0.5)

        # Train perceptron
        self.train(my_x, y)

    def predict(self, x):
        my_x = np.copy(x)

        # Append bias column
        my_x = np.append(
            np.matrix(np.zeros((x.shape[0], 1)) + 1.0),
            my_x, axis=1
        )

        # # Heaviside step activation function
        # return ((my_x * self.w) <= 0) * 2.0 - 1.0
        return ((my_x * self.w) >= 0) * 1.0

    def train(self, x, y):
        updated = True  # Flag to check whether os not was weights update in the iteration (stop condition)
        temp_w = np.copy(self.w)
        min_error = len(x)  # Stores the number of misclassified observations of the best iteration
        max_it = self.max_it  # Count the number of iterations

        while updated and max_it:
            # # Print progress at each 10% milestone
            # if (self.max_it - max_it) % (self.max_it / 10) == 0:
            #     print('{}% of max iterations already run'.format((self.max_it - max_it) * 100 / self.max_it))

            max_it -= 1

            # # Heaviside step activation function
            # p_y = ((x * temp_w) <= 0) * 2.0 - 1.0
            p_y = ((x * temp_w) >= 0) * 1.0

            # Check the number os misclassified observations and compare with the best result so far
            y_diff = y - p_y
            error = np.fabs(y_diff).sum()

            if error < min_error:
                min_error = error
                self.w = np.copy(temp_w)

            # Update delta
            delta_w = x.transpose() * (y_diff * self.n)

            if (delta_w != 0).any():
                temp_w += delta_w
            else:
                updated = False
