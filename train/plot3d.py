''' Evaluate Frustum PointNets

Visualize point cloud activation.

Author: Shiyi Lan
Date: March 2019
'''

import pickle
import mpl_toolkits as mplot3d
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="detection_results_v1", help="Model name [detection_results_v1]")

FLAGS = parser.parse_args()

path = 'train/' + FLAGS.model + '.pickle'

with open(path, "rb") as f:
    act = pickle.load(f)
    pc  = pickle.load(f)

fig = plt.figure()
ax = plt.axes(projection='3d')

x, y, z = pc
plt.scatter3d(x, y, z, act)

plt.show()
