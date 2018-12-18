from MLP import MLP
import numpy as np
import pandas as pd
import os


def experiment_ex1(dataset_file, exp_label, n=0.1):
    print('\n#####', exp_label, 'Experiment')
    print('')

    # Read dataset file
    df = pd.read_csv(dataset_file, header=None, names=['x1', 'x2', 'observed'], dtype=np.float)
    data = df.values

    # Split features and target variable
    my_in = np.asmatrix(data[:, :2])

    my_out = df[['observed']]

    # Train perceptron model
    mlp = MLP(x=my_in, y=my_out['observed'], hidden=[3], n=n)

    p_out = mlp.predict(x=my_in)

    # Print Results
    print('Input')
    print(my_in)
    print('')
    print('Weights')
    print(mlp.w)
    print('')
    print('Results:')
    print(my_out.join(p_out))
    print('')
    print('Back-Propagation Iterations:')
    print(mlp.back_propagation_iterations)
    print('')


if __name__ == '__main__':
    experiment_ex1('or.txt', 'OR')
    os.system('PAUSE')

    experiment_ex1('and.txt', 'AND')
    os.system('PAUSE')

    experiment_ex1('xor.txt', 'XOR')
    os.system('PAUSE')
