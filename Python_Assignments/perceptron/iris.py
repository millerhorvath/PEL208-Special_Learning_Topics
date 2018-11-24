import pandas as pd
import numpy as np
import os
from Perceptron import Perceptron

if __name__ == '__main__':
    # Read csv file
    data = pd.read_csv(os.path.join('..', 'iris.txt'))

    # Remove "Iris-setosa" observations
    data = data[data['class'] != 'Iris-setosa']

    # Format class variable
    data.loc[data['class'] == 'Iris-versicolor', 'class'] = 0
    data.loc[data['class'] == 'Iris-virginica', 'class'] = 1

    # Convert pandas.DataFrame into numpy.matrix
    data = np.asmatrix(data.values)

    # Split features and target variable
    x = data[:, :-1]
    y = data[:, -1]

    # Train perceptron
    p = Perceptron(x, y)

    # Classify dataset using the trained perceptron
    p_y = p.predict(x)

    # Build confusion matrix
    conf_m = np.zeros((2, 2))

    for i in range(len(y)):
        conf_m[int(y[i]), int(p_y[i])] += 1

    # Print confusion matrix
    print('##### Iris Experiment\n')
    print('Weights:')
    print(p.w)
    print('')
    print('Confusion Matrix:')
    print(conf_m)
