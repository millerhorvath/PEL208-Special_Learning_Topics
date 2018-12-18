from MLP import MLP
import numpy as np
import pandas as pd

x = np.matrix([[2, 1], [-1, 1], [3, 2]])
# x = np.matrix([[2, 1], [-1, 1]])
y = pd.DataFrame({'observed': [2, 3, 3]})
# y = np.matrix([0, 1])

mlp = MLP(x, y['observed'], [2])

p_y = mlp.predict(x)

print(y.join(p_y))
print(mlp.back_propagation_iterations)

# x = np.matrix([[2, 1], [-1, 1], [3, 2]])
# # x = np.matrix([[2, 1], [-1, 1]])
# y = np.matrix([[0], [1], [1]])
# # y = np.matrix([0, 1])
#
# mlp = MLP(x, y, [2])
#
# p_y = mlp.predict(x)
#
# print(np.append(p_y, y, axis=1))
# print(mlp.back_propagation_iterations)
