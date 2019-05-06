# Created on 2019年5月6日
# @author: lumiaoshan-James

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift


# Generate 2D Gaussian distribution
def generate_gaussian_distribution_samples(N):
    samples1 = np.random.multivariate_normal([0, 10], [[10, 10], [20, 30]], N)
    samples2 = np.random.multivariate_normal([10, 0], [[30, 0], [0, 10]], N)

    return samples1, samples2


samples1, samples2 = generate_gaussian_distribution_samples(500)
# draw_samples(samples1, samples2)

samples = np.concatenate((samples1, samples2))

# 搭建Mean-Shift聚类器
clf = MeanShift()

# 对样本数据进行聚类
predicted = clf.fit_predict(samples)

colors = [['red', 'green'][i] for i in predicted]
# 绘制聚类图
plt.scatter(samples[:, 0], samples[:, 1], c=colors, s=10)
plt.title('Mean Shift for 2D Gauss Distribution')
plt.show()
