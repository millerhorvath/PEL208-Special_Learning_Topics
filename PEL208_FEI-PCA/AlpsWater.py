# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
import os

n_features = 2  # Number of Feature
# data = list()  # List used to read rebuilt data with multiple principal components


# Read rebuilt data after dimensionality reduction
f = open("alpswater.txt", "r")  # Open file
# f = open("alpsWater_rebuilt_comp_1.csv", "r")  # Open file
f.readline()
data = f.readlines()  # Read all data in file
f.close()  # Close file

A = list()  # List used to store structured data

# Format data into numpy matrices
# Add a zero line in A (Coding trick to define numpy array shape)
A.append(np.array([0.0 for temp in range(n_features)]))

# For all lines in data
for j in range(len(data)):
    # Split data into a list where there is a blank character
    data[j] = data[j].split()
    # Split data into a list where there is a ',' character
    # data[j] = data[j].split(',')

    # Append the data[i][j] as a new line into the A[i] numpy matrix
    # A = np.append(A, [np.array(data[j], dtype=np.float)], axis=0)
    A = np.append(A, [np.array(data[j][1:], dtype=np.float)], axis=0)
A = np.delete(A, 0, 0)  # Remove zero line at the beginning of A


# # Print all A matrices
# for a in A:
#     print a
print A

components = []  # List of eigenvectors (principal components)

# Initialize components vector
for i in range(n_features):
    components.append([])
    for j in range(n_features):
        components[i].append(0.0)

components = np.array(components)


# Read components from file
f = open("alpsWater_components.csv", "r")  # Open file
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
f = open("alpsWater_coefs.csv", "r")
beta = f.readlines()
f.close()

beta = np.array(beta[0].split(','), dtype=np.float)

X = [190, 215, -100]
Y_int = [18, 32]
Y = []
Y2 = [beta[0] + x*beta[1] for x in X]

for comp in components:
    # print comp[0], comp[1]
    # Y.append([(x*comp[0]/comp[1]) for x in X])
    Y.append([(x*comp[0]/comp[1]) for x in X])
    print Y[-1]

# Compute the mean of each column in the original data
A_mean = np.mean(A, axis=0)
print A_mean
# A = A - A_mean

if not os.path.exists('plots'):
    os.makedirs('plots')

# Generate plots
# for i in range(n_features-1, -1, -1):
#     fig = plt.figure()
#     # plt.plot(X, [0 for temp in X], 'k-', linewidth=0.5)
#     # plt.plot([0 for temp in X], Y_int, 'k-', linewidth=0.5)
#     plt.plot(X + A_mean[1], Y[0] + A_mean[0], 'b--', linewidth=1, label='CP 1')
#     plt.plot(X + A_mean[1], Y[1] + A_mean[0], 'b-.', linewidth=1, label='CP 2')
#     plt.plot(A[i][:, 1], A[i][:, 0], 'rx', label='observações'.decode('utf-8'))
#     plt.xlim(X[0], X[1])
#     plt.ylim(Y_int[0], Y_int[1])
#     plt.grid(linestyle=':')
#     plt.legend()
#
#     plt.savefig(os.path.join('plots', 'alpsWater_{}comp_plot.png'.format(i+1)))
#     plt.show()

fig = plt.figure()
plt.plot(X, [0 for temp in X], 'k-', linewidth=0.5)
plt.plot([0 for temp in X], X, 'k-', linewidth=0.5)
plt.plot(X + A_mean[1], Y[0] + A_mean[0], 'b--', linewidth=1, label='CP 1')
plt.plot(X + A_mean[1], Y[1] + A_mean[0], 'b-.', linewidth=1, label='CP 2')
plt.plot(X, Y2, 'g-', label='Regressão MMQ'.decode('utf-8'), alpha=0.7)
plt.plot(A[:, 1], A[:, 0], 'rx', label='observações'.decode('utf-8'))
plt.xlim(X[0], X[1])
plt.ylim(Y_int[0], Y_int[1])
plt.xlabel('Temperatura (F)')
plt.ylabel('Pressão ("Hg)'.decode('utf-8'))
plt.grid(linestyle=':')
plt.legend()

plt.savefig(os.path.join('plots', 'alpsWater_PCAxLSM_plot.png'))
plt.show()
