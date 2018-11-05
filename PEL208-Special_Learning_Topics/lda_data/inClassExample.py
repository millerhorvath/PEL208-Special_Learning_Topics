import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

f = open(os.path.join('..', 'inClassExample_LDA.txt'))
original_data = f.readlines()
f.close()

f = open('inClassExample_rebuilt_lda_1.csv')
data_lda = f.readlines()
f.close()

f = open('inClassExample_rebuilt_comp_1.csv')
data_pca = f.readlines()
f.close()

for i in range(len(data_lda)):
    original_data[i] = np.array(original_data[i].split())

    data_lda[i] = ''.join(data_lda[i].split())
    data_lda[i] = np.array(data_lda[i].split(','))

    data_pca[i] = ''.join(data_pca[i].split())
    data_pca[i] = np.array(data_pca[i].split(','))
data_lda = np.array(data_lda)
original_data = np.array(original_data)
data_pca = np.array(data_pca)

# f = open(os.path.join('inClassExample_lda_vectors.csv'))
# components = f.readlines()
# f.close()

# for i in range(len(components)):
#     components[i] = np.array(components[i].split(','), dtype=np.float)
# components = np.array(components)

# print components

X_lda = {}
X_orig = {}
X_pca = {}
X2 = data_lda[:, :2].astype(dtype=np.float)
y = data_lda[:, -1].astype(dtype=str)\

for i in range(len(data_lda)):
    obs_class = y[i]
    if obs_class not in X_lda.keys():
        X_lda[obs_class] = []
        X_orig[obs_class] = []
        X_pca[obs_class] = []
    X_lda[obs_class].append(data_lda[i, :2])
    X_orig[obs_class].append(original_data[i, :2])
    X_pca[obs_class].append(data_pca[i, :2])

# components = np.matrix(components)

# lda = LinearDiscriminantAnalysis(solver='eigen')
# lda.fit(X2, y)
#
# lda_coef = np.array(lda.coef_[0])
# print lda_coef

for key in X_lda.keys():
    X_lda[key] = np.matrix(X_lda[key], dtype=np.float)
    X_orig[key] = np.matrix(X_orig[key], dtype=np.float)
    X_pca[key] = np.matrix(X_pca[key], dtype=np.float)

    # X[key] = components[:, 0].transpose() * X[key].transpose()
    # X[key] = (np.linalg.inv(components.transpose())[:, 0] * X[key]).transpose()
    # print '{}:\n'.format(key), X[key]

    # X[key] = lda_coef * X[key].transpose()
    # X[key] = (np.linalg.inv(lda_coef) * X[key]).transpose()
    # print '{}:\n'.format(key), X[key]

# cp_order = [x for x in range(len(X2[0]))]
cp_order = [x for x in range(len(X2[0])-1, -1, -1)]


data_mean = np.array([3.18182, 2.72727])


# comp_X = np.array([-3, 0, 4])
# comp_Y = [x*components[cp_order[0], 0]/components[cp_order[1], 0] for x in comp_X]
# comp_Y = [x*components[cp_order[0], 0] for x in comp_X]
# comp_Y = [x*components[cp_order[0], 1]/components[cp_order[1], 1] for x in comp_X]
# comp_Y2 = [x*components[cp_order[0], 1] for x in comp_X]

# comp_Y2 = [x*lda_coef[cp_order[0]]/lda_coef[cp_order[1]] for x in comp_X]

plt.grid(linestyle=':')
# plt.plot(comp_X + data_mean[0], comp_Y + data_mean[1], 'b')
# plt.plot(comp_X + data_mean[0], comp_Y2 + data_mean[1], 'g')
x = X_orig['1']
plt.plot(x[:, 0], x[:, 1], 'rx', label='classe 1')
x = X_orig['2']
plt.plot(x[:, 0], x[:, 1], 'bo', label='classe 2')
plt.legend()
# plt.plot(data_mean[0], data_mean[1], 'kX')

plt.show()

plt.grid(linestyle=':')
# plt.plot(comp_X + data_mean[0], comp_Y + data_mean[1], 'b')
# plt.plot(comp_X + data_mean[0], comp_Y2 + data_mean[1], 'g')
x = X_pca['1']
plt.plot(x[:, 0], x[:, 1], 'rx', label='classe 1')
x = X_pca['2']
plt.plot(x[:, 0], x[:, 1], 'bo', label='classe 2')
plt.legend()
# plt.plot(data_mean[0], data_mean[1], 'kX')

plt.show()

plt.grid(linestyle=':')
# plt.plot(comp_X + data_mean[0], comp_Y + data_mean[1], 'b')
# plt.plot(comp_X + data_mean[0], comp_Y2 + data_mean[1], 'g')
x = X_lda['1']
plt.plot(x[:, 0], x[:, 1], 'rx', label='classe 1')
x = X_lda['2']
plt.plot(x[:, 0], x[:, 1], 'bo', label='classe 2')
plt.legend()
# plt.plot(data_mean[0], data_mean[1], 'kX')

plt.show()

