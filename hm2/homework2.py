# Created on 2019年4月29日
# @author: lumiaoshan-James

import numpy as np
from numpy.linalg import svd
import matplotlib.pylab as plt
from pylab import *


def read_number_img_from_file(target_number):
    total_data = []
    current_data = []
    filename = 'optdigits-orig.wdep'
    for line in open(filename):
        if not line:
            break
        line = line.strip('\n')
        # the label line
        if len(line) == 2:
            number = int(line)
            if number == target_number:
                total_data.append(current_data)
            current_data = []
        else:
            for str in line:
                current_data.append(int(str))

    data = array(matrix(total_data).T)