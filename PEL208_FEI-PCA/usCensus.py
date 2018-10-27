# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
import os

n_features = 2  # Number of Feature


# Read rebuilt data after dimensionality reduction
f = open("usCensus.txt", "r")  # Open file
# f = open("usCensus_rebuilt_comp_1.csv", "r")  # Open file
data = f.readlines()  # Read all data in file
f.close()  # Close file

A = list()  # List used to store structured data

# Format data into numpy matrices
# Add a zero line in A (Coding trick to define numpy array shape)
A.append(np.array([0.0 for temp in range(n_features)]))

# For all lines in data
for j in range(len(data)):
    # Split data into a list where there is a blank character
    # data[j] = data[j].split()
    # Split data into a list where there is a ',' character
    data[j] = data[j].split(',')

    # Append the data[i][j] as a new line into the A[i] numpy matrix
    A = np.append(A, [np.array(data[j], dtype=np.float)], axis=0)
    # A = np.append(A, [np.array(data[j][1:], dtype=np.float)], axis=0)
A = np.delete(A, 0, 0)  # Remove zero line at the beginning of A


# Print all A matrices
print A


components = []  # List of eigenvectors (principal components)

# Initialize components vector
for i in range(n_features):
    components.append([])
    for j in range(n_features):
        components[i].append(0.0)

components = np.array(components)


# Read components from file
f = open("usCensus_components.csv", "r")  # Open file
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
f = open("usCensus_coefs.csv", "r")
beta = f.readlines()
f.close()

beta = np.array(beta[0].split(','), dtype=np.float)

X = np.array([1890, 2010])
Y_int = [60, 300]
Y = []
Y2 = [beta[0] + x*beta[1] for x in X]

# Compute the mean of each column in the original data
A_mean = np.mean(A, axis=0)
# A = A - A_mean

for comp in components:
    print comp[0], comp[1]
    # Y.append([(x*comp[0]/comp[1]) for x in X])
    Y.append([(x*comp[1]/comp[0]) for x in (X - A_mean[0])])
Y = np.array(Y)

if not os.path.exists('plots'):
    os.makedirs('plots')

# # Generate plots
# for i in range(n_features-1, -1, -1):
#     fig = plt.figure()
#     plt.plot(X, [0 for temp in X], 'k-', linewidth=0.5)
#     plt.plot([0 for temp in X], X, 'k-', linewidth=0.5)
#     # plt.plot(X, Y[0], 'b--', linewidth=1,
#     #          label='PC 1 - ({}/{}) * x'.format(components[0][0], components[0][1]))
#     # plt.plot(X, Y[1], 'b-.', linewidth=1,
#     #          label='PC 2 - ({}/{}) * x'.format(components[1][0], components[1][1]))
#     # plt.plot(X + A_mean[0], Y[0] + A_mean[1], 'b--', linewidth=1,
#     #          label='PC 1 - ({}/{}) * x'.format(components[0][1], components[0][0]))
#     # plt.plot(X + A_mean[0], Y[1] + A_mean[1], 'b-.', linewidth=1,
#     #          label='PC 2 - ({}/{}) * x'.format(components[1][1], components[1][0]))
#     plt.plot(X + A_mean[0], Y[0] + A_mean[1], 'b--', linewidth=1, label='CP 1')
#     plt.plot(X + A_mean[0], Y[1] + A_mean[1], 'b-.', linewidth=1, label='CP 2')
#     # plt.plot(X, Y2, 'g-', label='Least Squares Regression')
#     plt.plot(A[i][:, 0], A[i][:, 1], 'rx', label='observações'.decode('utf-8'))
#     plt.xlim(-1.5, 4.5)
#     plt.ylim(-1.5, 4.5)
#     plt.grid(linestyle=':')
#     plt.legend()
#
#     plt.savefig(os.path.join('plots', 'usCensus_{}comp_plot.png'.format(i+1)))
#     plt.show()

fig = plt.figure()
# plt.plot(X, [0 for temp in X], 'k-', linewidth=0.5)
# plt.plot([0 for temp in X], X, 'k-', linewidth=0.5)
plt.plot(X, Y[0] + A_mean[1], 'b--', linewidth=1, label='CP 1')
plt.plot(X, Y[1] + A_mean[1], 'b-.', linewidth=1, label='CP 2')
plt.plot(X, Y2, 'g-', label='Regressão MMQ'.decode('utf-8'))
plt.plot(A[:, 0], A[:, 1], 'rx', label='observações'.decode('utf-8'))
plt.xlabel('Ano')
plt.ylabel('População'.decode('utf-8'))
plt.xlim(X[0], X[1])
plt.ylim(Y_int[0], Y_int[1])
# plt.xlabel('x'.decode('utf-8'))
# plt.ylabel('y'.decode('utf-8'))
plt.grid(linestyle=':')
plt.legend()

plt.savefig(os.path.join('plots', 'usCensus_PCAxLSM_plot.png'))
plt.show()
