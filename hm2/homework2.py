# Created on 2019年4月29日
# @author: lumiaoshan-James

import numpy as np
from numpy.linalg import svd
import matplotlib.pylab as plt
from pylab import *


def read_number_img_from_file(target_number):
    number_metrics = []
    line_number = []
    file_name = 'optdigits-orig.wdep'
    line_count = 0
    for line in open(file_name):
        line_count = line_count + 1
        if not line:
            break
        line = line.strip('\n')
        # the label line
        if len(line) == 2:
            number = int(line)
            if number == target_number:
                number_metrics.append(line_number)
            # 如果不是指定的数字,那么不保存,直接清空
            line_number = []
        else:
            if line.isalnum():
                for num in line:
                    line_number.append(int(num))
            else:
                continue

    return number_metrics


def mean(data_array):
    return np.mean(data_array, axis=0)


def pca(data_array, k):
    # Step1 计算均值
    average = mean(data_array)

    # Step2 将原矩阵的每一列都减去均值
    m, n = np.shape(data_array)
    avgs = np.tile(average, (m, 1))
    data_adjust = data_array - avgs

    # Step3 计算协方差矩阵,n*n矩阵,n为特征维度
    covX = np.cov(data_adjust.T)

    # Step4 计算协方差矩阵的特征值和特征向量,w为特征值,v为特征向量
    eigen_value, eigen_vector = np.linalg.eig(covX)

    # Step5 按照特征值进行从大到小的排序
    sorted_index = np.argsort(-eigen_value)

    if k > n:
        print("k must lower than feature number")
        return

    # Step6 选出特征值最大的k个特征向量
    selected_vector = eigen_vector[sorted_index[:k]]

    # Step7 y是降维以后的数据
    y = np.dot(data_adjust, selected_vector.T)

    # Step8 恢复数据
    recover_array = np.dot(data_array.T, y)
    return recover_array.T


def pca_2(data_array):
    V, D, UT = svd(data_array.T)
    U = array(matrix(UT).T)
    results = []
    results.append(U[:, 0])
    results.append(U[:, 1])
    results = dot(results, data_array)
    return results


def draw(X):
    dim, N = shape(X)
    data = []
    datax = []
    datay = []
    for i in range(N):
        data.append([X[0][i], X[1][i]])
        datax.append(X[0][i])
        datay.append(X[1][i])
    print(len(data))
    plt.plot(datax, datay, 'g.')
    plt.title("Principal Components Analysis")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.grid()
    plt.show()


number_metrics = read_number_img_from_file(3)
number_array = np.array(number_metrics, dtype=float)
results = pca(number_array, 2)
# results2 = pca_2(number_array)

draw(results)
