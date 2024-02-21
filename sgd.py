import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from data_class import *
from Utils import *
from Pegasos_class import *

X_train, X_test, Y_train, Y_test = breast_cancer()

sigma = 0.1
Kernel_matrix = compute_kernel_matrix(X_train, sigma)

print(Kernel_matrix)

training_size = len(Kernel_matrix)
alpha = np.zeros(training_size)
iter = 300
obj = np.zeros(iter)
for i in range(iter):
    grad = compute_subgradient(Kernel_matrix, Y_train, alpha)
    alpha = alpha - 0.01/(i+1) * grad
    for j in range(training_size):
        obj[i] += max(0,1-np.dot(alpha,Kernel_matrix[j]))/training_size
    obj[i] += 1/2*alpha.T@Kernel_matrix@alpha
    print(obj[i])

test_size = len(X_test)
accuracy = 0

for i in range(test_size):
  Y_predict = 0
  for j in range(training_size):
    Y_predict += alpha[j] * np.exp(-np.linalg.norm(X_train[j]-X_test[i])**2/(2*sigma**2))

  print(Y_predict,Y_test[i])
  if np.sign(Y_predict) == Y_test[i]:
    accuracy += 1/test_size

print(accuracy)
