# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import os

# Read Books X Grades from file
f = open("Books_attend_grade.dat", "r")
data = f.readlines()
f.close()

# Organize data into X and y
X = np.array([[0, 0]], dtype=np.float)
y = np.array([], dtype=np.float)

for i in range(len(data)):
    data[i] = data[i].split()
    X = np.append(X, [np.array(data[i][0:-1], dtype=np.float)], axis=0)
    y = np.append(y, np.array(data[i][-1], dtype=np.float))
X = np.delete(X, 0, 0)

print X[:10]
print y[:10]

# Read least squares coefficients from file
f = open("booksXgrades_coefs.csv", "r")
f.readline()  # Read header
B = f.readlines()  # Read coefficients
f.close()

for i in range(len(B)):
    B[i] = np.array(B[i].split(','), dtype=np.float)
B = np.array(B)

print B

curve_X = np.arange(0.0, 4.0, 0.1)
curve_Y = np.arange(0.0, 20, 0.5)
curve_X, curve_Y = np.meshgrid(curve_X, curve_Y)
curve_z = []

for i in range(len(B)):
    curve_z.append(np.array(B[i][0] +
                            B[i][1] * curve_X + B[i][2] * curve_Y +
                            B[i][3] * curve_X * curve_X + B[i][4] * curve_Y * curve_Y))


file_label = ['inv', 'pinv']

for i in range(2):
    # fig, ax2 = plt.subplots()
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(curve_X, curve_Y, curve_z[i*4+0], color='b', label='linear')
    # ax.plot_surface(curve_X, curve_Y, curve_z[i*4+1], color='b', label='quadrática'.decode('utf-8'))
    ax.plot_surface(curve_X, curve_Y, curve_z[i*4+2], color='g', label='linear ponderada')
    # ax.plot_surface(curve_X, curve_Y, curve_z[i*4+3], color='g', label='quadrática ponderada'.decode('utf-8'))
    ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='x', label='observações'.decode('utf-8'))

    l1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    l2 = mpl.lines.Line2D([0], [0], linestyle="none", c='g', marker='o')
    l3 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='x')
    ax.legend([l1, l2, l3], ['linear', 'linear ponderada', 'observações'.decode('utf-8')])
    ax.set_xlabel('Livros Lidos')
    ax.set_ylabel('Assiduidade')
    ax.set_zlabel('Nota')
    # ax.set_title("Boiling point at the Alps")

    ax.view_init(10, -20)
    plt.savefig(os.path.join('plots', 'booksXgrades_plot_{}.png'.format(file_label[i])))
    plt.show()


for i in range(2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # ax.plot_surface(curve_X, curve_Y, curve_z[i*4+0], color='b', label='linear')
    ax.plot_surface(curve_X, curve_Y, curve_z[i*4+1], color='b', label='quadrática'.decode('utf-8'))
    # ax.plot_surface(curve_X, curve_Y, curve_z[i*4+2], color='g', label='linear ponderada')
    ax.plot_surface(curve_X, curve_Y, curve_z[i*4+3], color='g', label='quadrática ponderada'.decode('utf-8'))
    ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='x', label='observações'.decode('utf-8'))

    l1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    l2 = mpl.lines.Line2D([0], [0], linestyle="none", c='g', marker='o')
    l3 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='x')
    ax.legend([l1, l2, l3], ['quadrática'.decode('utf-8'), 'quadrática ponderada'.decode('utf-8'), 'observações'.decode('utf-8')])
    ax.set_xlabel('Livros Lidos')
    ax.set_ylabel('Assiduidade')
    ax.set_zlabel('Nota')
    # ax.set_title("Boiling point at the Alps")


    ax.view_init(10, -20)
    plt.savefig(os.path.join('plots', 'booksXgrades_plot2_{}.png'.format(file_label[i])))
    plt.show()
