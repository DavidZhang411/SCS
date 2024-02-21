import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from data_class import *
from Utils import *

X_train, X_test, Y_train, Y_test = breast_cancer()


sigma = 0.001
Kernel_matrix = rbf_kernel_matrix(X_train, sigma)


training_size = len(Kernel_matrix)
alpha = np.zeros(training_size)
iter = 200
obj = np.zeros(iter)
for i in range(iter):
    grad = compute_subgradient(Kernel_matrix, Y_train, alpha)
    alpha = alpha - 0.1/(i+1)**2 * grad
    for j in range(training_size):
        obj[i] += max(0,1-Y_train[j]*np.dot(alpha,Kernel_matrix[j,:]))/training_size
    obj[i] += 1/2*alpha.T@Kernel_matrix@alpha
    


alpha = np.zeros(training_size)
obj_scs = np.zeros(iter)
d = compute_subgradient(Kernel_matrix, Y_train, alpha)
delta = 0.1
for i in range(iter):
    grad = compute_subgradient(Kernel_matrix, Y_train, alpha)
    d = find_smallest_norm_convex_combination (d,grad)
    # print(np.dot(d,d))
    alpha = alpha - delta/(i+1)**2 * d
    for j in range(training_size):
        obj_scs[i] += max(0,1-Y_train[j]*np.dot(alpha,Kernel_matrix[j,:]))/training_size
    obj_scs[i] += 1/2*alpha.T@Kernel_matrix@alpha
    print(obj_scs[i])
test_size = 50
accuracy = 0



for i in range(test_size):
  Y_predict = 0
  for j in range(training_size):
    Y_predict += alpha[j] * Y_train[j] * np.exp(-np.linalg.norm((X_train[j]-Y_test[i]))**2/(2*sigma**2))

  print(Y_predict,Y_test[i])
  if np.sign(Y_predict) == Y_test[i]:
    accuracy += 1/test_size

print(accuracy)

# Plotting the objective function values over iterations
plt.plot(np.log(obj), label = "sgd")
plt.plot(np.log(obj_scs), label = "scs")
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Objectives Function Value Over Iterations')
plt.show()