from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

visualization=True

def create_dataset():
    x, y = make_circles(n_samples = 100, shuffle = True, noise = 0.1, random_state = None, factor = 0.2)

    if visualization:
        plt.scatter(x[:,0], x[:,1], c = y + 1)
        plt.show()
    return x, y

def project_dataset(data):
    x = data[:,0]
    y = data[:,1]
    z = x**2 + y**2
    return x, y, z


def twoD_projection():
    x, y = make_circles(n_samples = 100, shuffle = True, noise = 0.1, random_state = None, factor = 0.2)

    if visualization:
        plt.scatter(x[:,0], x[:,1], c = y + 1)
        c_2d = SVC(kernel='rbf', C=1.0, gamma=0.1)
        c2d_fit = c_2d.fit(x,y)

        a = plt.gca()
        x_limit = a.get_xlim()
        y_limit = a.get_ylim()

        x1 = np.linspace(x_limit[0], x_limit[1], 100)
        y1 = np.linspace(y_limit[0], y_limit[1], 100)
        YY, XX = np.meshgrid(y1, x1)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = c_2d.decision_function(xy).reshape(XX.shape)

        a.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        a.scatter(c_2d.support_vectors_[:, 0], c_2d.support_vectors_[:, 1], s=100, linewidth=2, facecolors='none')
        plt.show()
        
    return c2d_fit



data,label=create_dataset()
x,y,z=project_dataset(data)
if visualization:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x[label==0],y[label==0],z[label==0],c='g')
	ax.scatter(x[label==1],y[label==1],z[label==1],c='r')
	plt.show()

twoD_projection()
