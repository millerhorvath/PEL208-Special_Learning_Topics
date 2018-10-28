# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os

X = np.array([[1, 69], [1, 67], [1, 71], [1, 65],
              [1, 72], [1, 68], [1, 74], [1, 65],
              [1, 66], [1, 72]])

y = np.array([9.5, 8.5, 11.5, 10.5,
              11, 7.5, 12, 7,
              7.5, 13])

print X
print y

# Read least squares coefficients from file
f = open("inClassExample_coefs.csv", "r")
f.readline()  # Read header
B = f.readlines()  # Read coefficients
f.close()

for i in range(len(B)):
    B[i] = np.array(B[i].split(','), dtype=np.float)
B = np.array(B)

print B

curve_X = np.arange(65, 75)

curve_y = np.array(0.5145 * curve_X) - 25.6512

plt.plot(curve_X, curve_y, 'b-', label='y = 0.5145x - 25.6512')
# plt.plot(curve_X, curve_y[1], 'g--', label='w_i = selective')
# plt.plot(curve_X, curve_y[2], 'k-.', label="w_i = |1 / (y - y')|")
plt.plot(X[:, 1], y, 'rx', label='observações'.decode('utf-8'))
plt.xlabel('Altura (polegadas)'.decode('utf-8'))
plt.ylabel('Tamanho do Calçado (padrão US)'.decode('utf-8'))
# plt.title("Altura X Tamanho do Calçado".decode('utf-8'))
plt.grid()
plt.legend()

plt.savefig(os.path.join('plots', 'inClassExample_plot.png'))
plt.show()

curve_y = []
X2 = [X[i, 1] for i in [3, 5, 9]]
y2 = [y[i] for i in [3, 5, 9]]

print X2
print y2

for i in range(len(B)):
    curve_y.append(np.array(B[i][0] + B[i][1] * curve_X))

plt.plot(curve_X, curve_y[0], 'b-', label='Ponderação Igualitária'.decode('utf-8'))
plt.plot(curve_X, curve_y[1], 'g--', label='Ponderação Seletiva'.decode('utf-8'))
plt.plot(curve_X, curve_y[2], 'k-.', label="$W_{i,i} = 1 / |y_i - \hat{y}_i|$")
plt.plot(X[:, 1], y, 'rx', label='observações'.decode('utf-8'))
plt.plot(X2, y2, 'mX', label='outliers'.decode('utf-8'))
plt.xlabel('Altura (polegadas)'.decode('utf-8'))
plt.ylabel('Tamanho do Calçado (padrão US)'.decode('utf-8'))
# plt.title("Altura X Tamanho do Calçado".decode('utf-8'))
plt.grid()
plt.legend()

plt.savefig(os.path.join('plots', 'inClassExample2_plot.png'))
plt.show()
