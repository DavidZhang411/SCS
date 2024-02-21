import numpy as np

def rbf_kernel(x1, x2, gamma):
    """
    Compute the RBF (Gaussian) kernel between two vectors.
    """
    return np.exp(-np.linalg.norm(x1 - x2) ** 2/(2*gamma**2))
    #return np.dot(x1,x2)

def compute_kernel_matrix(X, gamma):
    """
    Compute the RBF kernel matrix for a dataset X.
    """
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = rbf_kernel(X[i], X[j], gamma)
    return K

def pegasos_kernel(X, y, lambda_param, iterations, gamma):
    """
    Pegasos SVM training algorithm with kernel support (RBF kernel).

    Parameters:
    - X: numpy array of shape (n_samples, n_features), training data.
    - y: numpy array of shape (n_samples,), training labels, should be -1 or 1.
    - lambda_param: regularization parameter.
    - iterations: number of iterations.
    - gamma: parameter for the RBF kernel.

    Returns:
    - alpha: The final alpha coefficients of the SVM in the dual form.
    """
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    K = compute_kernel_matrix(X, gamma)
    
    for it in range(1, iterations + 1):
        i = np.random.randint(0, n_samples)
        
        # Compute the decision value using the kernel matrix and alpha values
        decision_value = y[i] * np.dot(alpha * y, K[i,:])/(lambda_param * (it))
        if decision_value < 1:
            alpha[i] = alpha[i] + 1
    
    return alpha

# Example usage:
# X_train: Your training data matrix
# y_train: Your training labels (should be -1 or 1)
# lambda_param: Regularization parameter (e.g., 0.01)
# iterations: Number of iterations (e.g., 1000)
# gamma: Parameter for the RBF kernel (e.g., 0.1)

# alpha = pegasos_kernel(X_train, y_train, lambda_param=0.01, iterations=1000, gamma=0.1)
def predict(X_train, y_train, X_test, alpha, gamma):
    """
    Predict the labels for new data using the trained kernelized Pegasos SVM.

    Parameters:
    - X_train: numpy array of shape (n_samples, n_features), training data.
    - y_train: numpy array of shape (n_samples,), training labels.
    - X_test: numpy array of shape (n_test_samples, n_features), data to predict.
    - alpha: numpy array of shape (n_samples,), the trained alpha coefficients.
    - gamma: the parameter for the RBF kernel.

    Returns:
    - predictions: numpy array of shape (n_test_samples,), predicted labels for X_test.
    """
    n_test_samples = X_test.shape[0]
    predictions = np.zeros(n_test_samples)

    # Compute the kernel matrix between training and test data
    for i in range(n_test_samples):
        decision_function = 0
        for j in range(len(alpha)):
            if alpha[j] > 0:  # Consider only non-zero alphas (support vectors)
                decision_function += alpha[j] * y_train[j] * rbf_kernel(X_train[j], X_test[i], gamma)
        #print(decision_function)
        predictions[i] = np.sign(decision_function)
    
    return predictions

# Example usage:
# Assuming X_train, y_train, and alpha are as before, and you have new test data X_test
# gamma: Parameter for the RBF kernel (e.g., 0.1)

# predictions = predict(X_train, y_train, X_test, alpha, gamma)

def pegasos_obj(X, y, lambda_param, iterations, gamma):
    """
    Pegasos SVM training algorithm with kernel support (RBF kernel).

    Parameters:
    - X: numpy array of shape (n_samples, n_features), training data.
    - y: numpy array of shape (n_samples,), training labels, should be -1 or 1.
    - lambda_param: regularization parameter.
    - iterations: number of iterations.
    - gamma: parameter for the RBF kernel.

    Returns:
    - alpha: The final alpha coefficients of the SVM in the dual form.
    """
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    K = compute_kernel_matrix(X, gamma)
    obj = np.zeros(iterations)
    for it in range(0, iterations):
        i = np.random.randint(0, n_samples)
        
        # Compute the decision value using the kernel matrix and alpha values
        decision_value = y[i] * np.dot(alpha * y, K[i,:])/(lambda_param*(it+1))
        if decision_value < 1:
            alpha[i] = alpha[i] + 1
        
        for j in range(n_samples):
            obj[it] += max(0,1-np.dot(alpha/(it+1) * y,K[j,:]))/n_samples
        obj[it] += 1/2*alpha.T@K@alpha/(it+1)**2
        print(obj[it])
    return obj
