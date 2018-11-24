from Perceptron import Perceptron
import numpy as np
import pandas as pd
import os


def experiment_ex1(dataset_file, exp_label):
    print('\n#####', exp_label, 'Experiment')
    print('')

    # Read dataset file
    data = pd.read_csv(dataset_file, header=None, dtype=np.float).values

    # Split features and target variable
    my_in = np.asmatrix(data[:, :2])
    my_out = np.asmatrix(data[:, -1:])

    # Train perceptron model
    p = Perceptron(x=my_in, y=my_out)

    # Print Results
    print('Input')
    print(my_in)
    print('')
    print('Weights')
    print(p.w)
    print('')
    print('Expected output:')
    print(my_out)
    print('')
    print('Perceptron output:')
    print(p.predict(x=my_in))
    print('')


if __name__ == '__main__':
    experiment_ex1('or.txt', 'OR')
    os.system('PAUSE')

    experiment_ex1('and.txt', 'AND')
    os.system('PAUSE')

    experiment_ex1('xor.txt', 'XOR')
    os.system('PAUSE')
