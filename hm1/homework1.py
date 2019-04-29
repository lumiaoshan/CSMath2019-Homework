# Created on 2019年4月29日
# @author: lumiaoshan-James

import numpy as np
import random
import matplotlib.pylab as plt
from scipy.optimize import leastsq


# 生成SinX函数点
def generate_sin_samples(x):
    y = np.sin(2 * np.pi * x)
    return y


# 生成带有高斯噪声的SinX函数点
def generate_sin_samples_with_gauss_noise(x):
    y = generate_sin_samples(x)
    for i in range(len(y)):
        y[i] = random.gauss(0, 0.1) + y[i]
    return y


# 计算最小二乘估计的残差函数
def loss_function(start_weights, y, x):
    residuals = y - np.polyval(start_weights, x)
    np.append(residuals, 0.01 * start_weights)
    return residuals


# 绘制最终的图案
def curve_fitting(degree, samples_count, useRegularization):
    # base info
    title = "Degree:" + str(degree) + ";Samples:" + str(
        samples_count) + ";"
    plt.title(title + "Regularization" if useRegularization else title)
    plt.xlabel("x-value")
    plt.ylabel("y-value")

    # Step1.打印sinx函数
    x_1000 = np.linspace(0, 1, 1000)
    y_1000 = generate_sin_samples(x_1000)
    plt.plot(x_1000, y_1000)

    # Step2.打印生成的散点
    x_samples_count = np.linspace(0, 1, samples_count)
    y_samples_count = generate_sin_samples_with_gauss_noise(x_samples_count)
    plt.scatter(x_samples_count, y_samples_count)

    # Step3.计算权重值,按照是否进行正则化优化区分计算
    if useRegularization:
        start_weights = range(1, 10)
        # Minimize the sum of squares of a set of equations.
        lsq = leastsq(loss_function, start_weights, args=(y_samples_count, x_samples_count))
        weights = lsq[0]
    else:
        weights = np.polyfit(x_samples_count, y_samples_count, degree)

    # Step4.根据权重值打印回归的散点
    x_fit = np.linspace(0, 1, 1000)
    y_fit = []
    for i in range(len(x_fit)):
        y_fit.append(np.polyval(weights, x_fit[i]))

    plt.plot(x_fit, y_fit)
    plt.legend(("SinX", "Curve Fitting", "Sample Points"))
    plt.show()


# degree 3, sample 10 with gauss noise
curve_fitting(3, 10, False)
# degree 9, sample 10 with gauss noise
curve_fitting(9, 10, False)
# degree 9, sample 15 with gauss noise
curve_fitting(9, 15, False)
# degree 9, sample 100 with gauss noise
curve_fitting(9, 100, False)
# degree 9, sample 100 with gauss noise and use regularization with weights sum = 0.5
curve_fitting(9, 10, True)
