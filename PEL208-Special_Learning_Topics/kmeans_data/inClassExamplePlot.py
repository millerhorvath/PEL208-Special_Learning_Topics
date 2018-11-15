import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

output_file_name = 'inClassExample.png'
csv_path = os.path.join('..', 'inClassExample_KMeans.txt')

if not os.path.exists('plots'):
    os.makedirs('plots')

pd_data = pd.read_csv(csv_path, header=None, names=['x', 'y'])
pd_centroids = pd.read_csv('inClassExample_kmeans_centroids.csv', header=None, names=['x', 'y'])
pd_classes = pd.read_csv('inClassExample_kmeans_classes.csv', header=None, names=['class'])

data = pd_data.join(pd_classes, lsuffix='_l', rsuffix='_r')

color_list = []

for c, d in data.groupby('class'):
    temp = plt.scatter(d['x'], d['y'], label='Class {} data'.format(c))
    color_list.append(temp.get_edgecolor())

for i, d in pd_centroids.iterrows():
    plt.scatter(d['x'], d['y'], c=color_list[i], marker='X', label='Class {} centroid'.format(i))

plt.grid(linestyle=':')
plt.legend()

plt.savefig(os.path.join('plots', output_file_name))
plt.show()
