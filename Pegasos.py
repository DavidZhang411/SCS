import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from Utils import *
from data_class import *

from Pegasos_class import *

X_train, X_test, Y_train, Y_test = breast_cancer()


sigma = 0.001
print(compute_kernel_matrix(X_train,sigma))
alpha = pegasos_kernel(X_train, Y_train, 1, 200, sigma)

pred = predict(X_train, Y_train, X_test, alpha, sigma)
acc = compute_accuracy(Y_test, pred)

obj = pegasos_obj(X_train, Y_train, 1, 200, sigma)

print(acc)