# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import os

n_features = 3  # Number of Feature
# data = list()  # List used to read rebuilt data with multiple principal components


# Read rebuilt data after dimensionality reduction
# f = open("booksXgrades_rebuilt_comp_1.csv", "r")  # Open file
f = open("Books_attend_grade.dat", "r")  # Open file
f.readline()
data = f.readlines()  # Read all data in file
f.close()  # Close file

A = list()  # List used to store structured data

# Format data into numpy matrices
# Add a zero line in A (Coding trick to define numpy array shape)
A.append(np.array([0.0 for temp in range(n_features)]))

# For all lines in data
for j in range(len(data)):
    # Split data into a list where there is a ',' character
    # data[j] = data[j].split(',')
    # Split data into a list where there is a blank character
    data[j] = data[j].split()

    # Append the data[i][j] as a new line into the A[i] numpy matrix
    A = np.append(A, [np.array(data[j], dtype=np.float)], axis=0)
A = np.delete(A, 0, 0)  # Remove zero line at the beginning of A

# # Print all A matrices
# for a in A:
#     print a
print A[:5]

components = []  # List of eigenvectors (principal components)

# Initialize components vector
for i in range(n_features):
    components.append([])
    for j in range(n_features):
        components[i].append(0.0)

components = np.array(components)

# Read components from file
f = open("booksXgrades_components.csv", "r")  # Open file
data = f.readlines()  # Read all data
f.close()  # Close file

# Split string data into vectors
for i in range(len(data)):
    data[i] = data[i].split(',')

# Store read data into the numpy matrix
for i in range(n_features):
    for j in range(n_features):
        components[i][j] = np.float(data[i][j])

# components = np.transpose(components)

# Read Linear regression coefficients from file
f = open("booksXgrades_coefs.csv", "r")
beta = f.readlines()
f.close()

beta = np.array(beta[0].split(','), dtype=np.float)

# Compute the mean of each column in the original data
A_mean = np.mean(A, axis=0)
print A_mean
# A = A - A_mean

if not os.path.exists('plots'):
    os.makedirs('plots')

# Generate plots
curve_X = np.arange(0.0, 4.0, 0.1)
curve_Y = np.arange(6.0, 20, 0.35)
curve_X, curve_Y = np.meshgrid(curve_X, curve_Y)
curve_z = np.array(beta[0] +
                   beta[1] * curve_X +
                   beta[2] * curve_Y)

X = curve_X - A_mean[0]
Y = curve_Y - A_mean[1]
Z = []

for comp in components:
    print comp[0], comp[1], comp[2]
    # Y.append([(x*comp[0]/comp[1]) for x in X])
    Z.append(np.array([(x * comp[0] / comp[1]) for x in X]))
    Z[-1] += np.array([(y * comp[2] / comp[1]) for y in Y])
    # print Z[-1]

# print len(Z[0])
# exit()
# B = A - A_mean

# for i in range(3):
views = [(0, 0), (10, -30)]
for i in range(len(views)):
    (vx, vy) = views[i]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(X + A_mean[0], Y + A_mean[1], Z[0] + A_mean[2], color='b',
                    label='PC 1'.decode('utf-8'), alpha=0.5)
    # ax.plot_surface(X, Y, Z[1], color='m',
    #                 label='PC 2'.decode('utf-8'), alpha=0.5)
    # ax.plot_surface(X, Y, Z[2], color='y',
    #                 label='PC 3'.decode('utf-8'), alpha=0.5)
    # ax.plot_surface(X + A_mean[0], Y + A_mean[1], Z[i] + A_mean[2], color='y',
    #                 label='PC 1'.decode('utf-8'), alpha=0.5)
    ax.plot_surface(curve_X, curve_Y, curve_z, color='g',
                    label='Regressão MMQ'.decode('utf-8'))
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='r', marker='x',
               label='observações'.decode('utf-8'))
    # ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='r', marker='x',
    #            label='observações'.decode('utf-8'))

    l1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    l2 = mpl.lines.Line2D([0], [0], linestyle="none", c='g', marker='o')
    l3 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='x')
    ax.legend([l1, l2, l3], ['PC 1', 'Regressão MMQ'.decode('utf-8'),
                         'observações'.decode('utf-8')])

    ax.set_xlabel('Livros Lidos')
    ax.set_ylabel('Assiduidade')
    ax.set_zlabel('Nota')

    # ax.view_init(10, -45)
    ax.view_init(vx, vy)
    plt.savefig(os.path.join('plots', 'booksXgrades_PCAxLSM_plot_{}.png'.format(i)))
    plt.show()
