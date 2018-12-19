import pandas as pd
import numpy as np
import os
from MLP import MLP
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import itertools
from datetime import datetime


if __name__ == '__main__':
    # Read csv file
    data = pd.read_csv(os.path.join('..', 'seeds.txt'))

    outpath = os.path.join('plots', 'seeds')

    orig_columns = data.columns

    # Normalize Columns and add polynomial features
    for col in orig_columns[:-1]:
        data[col] = (data[col] - data[col].min()) / (data[col].max()-data[col].min())
        data = data.join(data[col].pow(2), rsuffix='2')
        data = data.join(data[col].pow(3), rsuffix='3')

    # Shuffle and sampling dataset keeping class proportion
    train_df = pd.DataFrame(columns=data.columns)
    test_df = pd.DataFrame(columns=data.columns)
    validation_df = pd.DataFrame(columns=data.columns)

    grouped = data.groupby('class')

    np.random.seed(28021992)

    for name, group in grouped:
        idx_list = list(group.index)
        np.random.shuffle(idx_list)
        group = group.loc[idx_list]

        train_size = int(np.round(len(group) * 0.6))
        test_size = int(np.round(len(group) * 0.2))

        train_df = train_df.append(group[:train_size])
        test_df = test_df.append(group[train_size:train_size+test_size])
        validation_df = validation_df.append(group[train_size+test_size:])

    y_train = train_df[['class']].copy()
    y_test = test_df[['class']].copy()
    y_validation = validation_df[['class']].copy()

    # Convert pandas.DataFrame into numpy.matrix
    x_train = np.asmatrix(train_df.drop('class', axis=1).values)
    x_test = np.asmatrix(test_df.drop('class', axis=1).values)
    x_validation = np.asmatrix(validation_df.drop('class', axis=1).values)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    n_list = [0.3, 0.1, 0.03, 0.01, 0.003]

    best_mlp = None
    best_precision = 0.0
    best_n = None
    best_topology = None

    for i in range(1, 4):
        for hidden in itertools.product([i for i in range(3, 7)], repeat=i):
            hidden = list(hidden)
            best_topology_mlp = None
            best_topology_precision = 0.0
            best_topology_n = None

            for n in n_list:
                print('[' + datetime.now().__str__() + '] training MLP {} witn n={}'.format(hidden, n))

                for it in range(5):
                    print('[' + datetime.now().__str__() + '] iteration {}'.format(it))
                    # Train perceptron
                    mlp = MLP(x_train, y_train['class'], hidden=hidden, n=n)

                    # Classify dataset using the trained perceptron
                    p_y = mlp.predict(x_test)

                    actual_precision = precision_score(
                        y_test.values.astype(int),
                        p_y.values.astype(int),
                        average='micro'
                    )

                    if actual_precision > best_topology_precision:
                        best_topology_precision = actual_precision
                        best_topology_mlp = mlp
                        best_topology_n = n

            if best_topology_precision > best_precision:
                best_precision = best_topology_precision
                best_n = best_topology_n
                best_mlp = best_topology_mlp
                best_topology = hidden.copy()

            # Print confusion matrix
            p_y = best_topology_mlp.predict(x_test)
            conf_m = confusion_matrix(y_test.values.astype(int), p_y.values.astype(int))
            print('##### Seeds Experiment - Layers {}\n'.format(hidden))
            # print('Weights:')
            # print(best_topology_mlp.w)
            # print()
            print('Best n:')
            print(best_topology_n)
            print('Confusion Matrix:')
            print(conf_m)
            print('Precision:')
            print(best_topology_precision)

            # Plot Error Graph to check learning rate
            plt.plot([i for i in range(len(best_topology_mlp.erro_var))], best_topology_mlp.erro_var)
            plt.grid(True, linestyle=':')
            plt.savefig(os.path.join(outpath, '{}.png'.format(hidden)))
            plt.close()

    # Best MLP validation
    p_y = best_mlp.predict(x_validation)
    conf_m = confusion_matrix(y_validation.values.astype(int), p_y.values.astype(int))
    print('##### Seeds Experiment - Best Topology {}\n'.format(best_topology))
    # print('Weights:')
    # print(best_mlp.w)
    # print()
    print('Best n:')
    print(best_n)
    print('Confusion Matrix:')
    print(conf_m)
    print('Test Precision:')
    print(best_precision)
    actual_precision = precision_score(y_validation.values.astype(int), p_y.values.astype(int), average='micro')
    print('Validation Precision:')
    print(actual_precision)
