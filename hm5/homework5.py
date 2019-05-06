"""
Generate sampling points from 2 classes with non-linear distributions (in 2D)
Apply kernel functions for the classification modeling: linear kernel, Gaussian kernel or other kernels
Solve the problem by using QP (quadratic programming) via the active set method
Plot the classification results
"""

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs


def generate_data():
    return make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=2)

def gauss_kernel(X, y):
    kernel = euclidean_distances(X, y) ** 2
    kernel = kernel * (-1 / (0.01 ** 2))
    kernel = np.exp(kernel)
    return kernel


def draw_point(ax, clf, tn):
    # 绘制样本点
    for i in X_train:
        ax.set_title(titles[tn])
        res = clf.predict(np.array(i).reshape(1, -1))
        if res > 0:
            ax.scatter(i[0], i[1], c='r', marker='*')
        else:
            ax.scatter(i[0], i[1], c='g', marker='*')
    # 绘制实验点
    for i in X_test:
        res = clf.predict(np.array(i).reshape(1, -1))
        if res > 0:
            ax.scatter(i[0], i[1], c='r', marker='.')
        else:
            ax.scatter(i[0], i[1], c='g', marker='.')


# 设置子图数量
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
ax0, ax1, ax2, ax3 = axes.flatten()
# 准备训练样本
# x = [[1, 8], [3, 20], [1, 15], [3, 35], [5, 35], [4, 40], [7, 80], [6, 49]]
# y = [1, 1, -1, -1, 1, -1, -1, 1]
X, y = generate_data()
X_train = X[:150]
y_train = y[:150]
X_test = X[151:]
y_test = y[151:]

# 设置子图的标题
titles = ['LinearSVM (linear kernel)',
          'SVM with polynomial (degree 3) kernel',
          'SVM with RBF kernel',
          'SVM with Gauss kernel']

for n in range(0, 4):
    if n == 0:
        clf = svm.SVC(kernel='linear').fit(X_train, y_train)
        draw_point(ax0, clf, 0)
    elif n == 1:
        clf = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)
        draw_point(ax1, clf, 1)
    elif n == 2:
        clf = svm.SVC(kernel='rbf').fit(X_train, y_train)
        draw_point(ax2, clf, 2)
    else:
        clf = svm.SVC(kernel=gauss_kernel).fit(X_train, y_train)
        draw_point(ax3, clf, 3)
plt.show()
