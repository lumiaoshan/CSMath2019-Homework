import numpy as np
from numpy import matrix as mat
from matplotlib import pyplot as plt
import random


def test_function(params, x):  # 需要拟合的函数，abc是包含三个参数的一个矩阵[[a],[b],[c]]
    a = params[0, 0]
    b = params[1, 0]
    c = params[2, 0]
    y = np.exp(a * x ** 2 + b * x + c)
    return y


def deriv(params, x, n):  # 对函数求偏导
    x1 = params.copy()
    x2 = params.copy()
    x1[n, 0] -= 0.000001
    x2[n, 0] += 0.000001
    y1 = test_function(x1, x)
    y2 = test_function(x2, x)
    d = (y2 - y1) * 1.0 / (0.000002)
    return d


# a1, b1, c1是需要拟合的函数y(x) 的真实参数
def LM(x, y):
    jacobian_matrix = mat(np.zeros((points_count, 3)))  # 雅克比矩阵
    fx = mat(np.zeros((points_count, 1)))  # f(x)  100*1  误差
    fx_tmp = mat(np.zeros((points_count, 1)))
    init_params = mat([[0.8], [2.7], [1.5]])  # 参数初始化
    lase_mse = 0
    step = 0
    u, v = 1, 2
    MAX_ITER = 100
    step_log = ""
    while (MAX_ITER):

        mse, mse_tmp = 0, 0
        step += 1
        for i in range(points_count):
            fx[i] = test_function(init_params, x[i]) - y[0, i]  # 注意不能写成  y - Func  ,否则发散
            mse += fx[i, 0] ** 2

            for j in range(3):
                jacobian_matrix[i, j] = deriv(init_params, x[i], j)  # 数值求导
        mse /= points_count  # 范围约束

        H = jacobian_matrix.T * jacobian_matrix + u * np.eye(3)  # 3*3
        dx = -H.I * jacobian_matrix.T * fx  # 注意这里有一个负号，和fx = Func - y的符号要对应
        xk_tmp = init_params.copy()
        xk_tmp += dx

        for j in range(points_count):
            fx_tmp[i] = test_function(xk_tmp, x[i]) - y[0, i]
            mse_tmp += fx_tmp[i, 0] ** 2
        mse_tmp /= points_count

        q = (mse - mse_tmp) / ((0.5 * dx.T * (u * dx - jacobian_matrix.T * fx))[0, 0])

        if q > 0:
            s = 1.0 / 3.0
            v = 2
            mse = mse_tmp
            init_params = xk_tmp
            temp = 1 - pow(2 * q - 1, 3)

            if s > temp:
                u = u * s
            else:
                u = u * temp
        else:
            u = u * v
            v = 2 * v
            init_params = xk_tmp

        print("step = %d,abs(mse-lase_mse) = %.8f" % (step, abs(mse - lase_mse)))
        step_log += "step = %d,abs(mse-lase_mse) = %.8f" % (step, abs(mse - lase_mse)) + "\n"
        if abs(mse - lase_mse) < 0.000001:
            break

        lase_mse = mse  # 记录上一个 mse 的位置
        MAX_ITER -= 1

    return init_params, step_log


points_count = 100
x = np.linspace(0, 1, points_count)  # 产生包含噪声的数据
y = [np.exp(1 * i ** 2 + 3 * i + 2) + random.gauss(0, 4) for i in x]
plt.scatter(x, y, s=4)
y = mat(y)  # 转变为矩阵形式

init_params, step_log = LM(x, y)
z = [test_function(init_params, i) for i in x]  # 用拟合好的参数画图
title = "Params:\na:" + str(init_params[0, 0]) + ",\nb:" + str(init_params[1, 0]) + ",\nc:" + str(
    init_params[2, 0]) + "\nReal:\na:1,b:3,c:2"
plt.title(title, loc="left")
plt.suptitle(step_log, fontsize=8, x=0.7)
plt.plot(x, z, 'r')
plt.show()
