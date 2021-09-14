from array import array
from time import time

import numpy as np

st1 = time()


def f1():
    a = range(0, 50000000)

    result = []
    for i in a:
        result.append(i)


f1()
print("RUN TIME : {0}".format(time() - st1))

st2 = time()


def f2():
    a = range(0, 50000000)
    arr = np.empty(50000000)
    for i in a:
        arr[i] = i


f2()
print("RUN TIME : {0}".format(time() - st2))
