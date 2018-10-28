# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
import os

# Read boilling points at alps data from file
f = open(os.path.join("..", "alpswater.txt"), "r")
f.readline()
data = f.readlines()
f.close()

# Organize data into X and y
X = np.array([], dtype=np.float)
y = np.array([], dtype=np.float)

for i in range(len(data)):
    data[i] = data[i].split()
    X = np.append(X, np.array(data[i][2], dtype=np.float))
    y = np.append(y, np.array(data[i][1], dtype=np.float))

print X
print y

# Read least squares coefficients from file
f = open("boilingPointsAtAlps_coefs.csv", "r")
f.readline()  # Read header
B = f.readlines()  # Read coefficients
f.close()

for i in range(len(B)):
    B[i] = np.array(B[i].split(','), dtype=np.float)
B = np.array(B)

print B

curve_X = np.arange(190.0, 215.0, 0.5)
curve_y = []

for i in range(len(B)):
    curve_y.append(np.array(B[i][0] + B[i][1]*curve_X + B[i][2]*curve_X*curve_X))

file_label = ['inv', 'pinv']

for i in range(2):
    plt.plot(curve_X, curve_y[i*4+0], 'b-', label='linear')
    plt.plot(curve_X, curve_y[i*4+1], 'g-', label='quadrática'.decode('utf-8'))
    plt.plot(curve_X, curve_y[i*4+2], 'b--', label='linear ponderada')
    plt.plot(curve_X, curve_y[i*4+3], 'g--', label='quadrática ponderada'.decode('utf-8'))
    plt.plot(X, y, 'rx', label='observações'.decode('utf-8'))
    plt.xlabel('Temperatura (F)')
    plt.ylabel('Pressão ("Hg)'.decode('utf-8'))
    # plt.title("Boiling point at the Alps")
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join('plots', 'boilingPointsAtAlps_plot_{}.png'.format(file_label[i])))
    plt.show()

curve_X = np.arange(0.0, 215.0, 0.5)
curve_y = []

for i in range(len(B)):
    curve_y.append(np.array(B[i][0] + B[i][1]*curve_X + B[i][2]*curve_X*curve_X))

for i in range(2):
    plt.plot(curve_X, curve_y[i*4+0], 'b-', label='linear')
    plt.plot(curve_X, curve_y[i*4+1], 'g-', label='quadrática'.decode('utf-8'))
    plt.plot(curve_X, curve_y[i*4+2], 'b--', label='linear ponderada')
    plt.plot(curve_X, curve_y[i*4+3], 'g--', label='quadrática ponderada'.decode('utf-8'))
    plt.plot(X, y, 'rx', label='observações'.decode('utf-8'))
    plt.xlabel('Temperatura (F)')
    plt.ylabel('Pressão ("Hg)'.decode('utf-8'))
    # plt.title("Boiling point at the Alps")
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join('plots', 'boilingPointsAtAlps_plot2_{}.png'.format(file_label[i])))
    plt.show()
