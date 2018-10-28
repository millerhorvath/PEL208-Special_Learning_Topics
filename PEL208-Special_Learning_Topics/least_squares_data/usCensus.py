# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
import os

X = np.array([[1, 1900, 1900 * 1900], [1, 1910, 1910 * 1910],
              [1, 1920, 1920 * 1920], [1, 1930, 1930 * 1930],
              [1, 1940, 1940 * 1940], [1, 1950, 1950 * 1950],
              [1, 1960, 1960 * 1960], [1, 1970, 1970 * 1970],
              [1, 1980, 1980 * 1980], [1, 1990, 1990 * 1990],
              [1, 2000, 2000 * 2000]])

y = np.array([75.9950, 91.9720, 105.7110, 123.2030,
              131.6690, 150.6970, 179.3230, 203.2120,
              226.5050, 249.6330, 281.4220])

print X
print y

# Read least squares coefficients from file
f = open("usCensus_coefs.csv", "r")
f.readline()  # Read header
B = f.readlines()  # Read coefficients
f.close()

for i in range(len(B)):
    B[i] = np.array(B[i].split(','), dtype=np.float)
B = np.array(B)

print B

curve_X = np.arange(1880, 2020)
curve_y = []

for i in range(len(B)):
    curve_y.append(np.array(B[i][0] + B[i][1] * curve_X + B[i][2] * curve_X * curve_X))

file_label = ['inv', 'pinv']

for i in range(2):
    plt.plot(curve_X, curve_y[i*4+0], 'b-', label='linear')
    plt.plot(curve_X, curve_y[i*4+1], 'g-', label='quadrática'.decode('utf-8'))
    plt.plot(curve_X, curve_y[i*4+2], 'b--', label='linear ponderada')
    plt.plot(curve_X, curve_y[i*4+3], 'g--', label='quadrática ponderada'.decode('utf-8'))
    plt.plot(X[:, 1], y, 'rx', label='observações'.decode('utf-8'))
    plt.xlabel('Ano')
    plt.ylabel('População'.decode('utf-8'))
    # plt.title("US Census")
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join('plots', 'usCensus_plot_{}.png'.format(file_label[i])))
    plt.show()

curve_X = np.arange(0, 2020)

for i in range(len(B)):
    curve_y[i] = np.array(B[i][0] + B[i][1] * curve_X + B[i][2] * curve_X * curve_X)

for i in range(2):
    plt.plot(curve_X, curve_y[i*4+0], 'b-', label='linear')
    plt.plot(curve_X, curve_y[i*4+1], 'g-', label='quadrática'.decode('utf-8'))
    plt.plot(curve_X, curve_y[i*4+2], 'b--', label='linear ponderada')
    plt.plot(curve_X, curve_y[i*4+3], 'g--', label='quadrática ponderada'.decode('utf-8'))
    plt.plot(X[:, 1], y, 'rx', label='observações'.decode('utf-8'))
    plt.xlabel('Ano')
    plt.ylabel('População'.decode('utf-8'))
    # plt.title("US Census")
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join('plots', 'usCensus_plot2_{}.png'.format(file_label[i])))
    plt.show()
