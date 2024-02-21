
import numpy as np
from scipy.optimize import minimize

def rbf_kernel_matrix(X, sigma):
    """
    Compute the RBF kernel matrix for a dataset X.

    Parameters:
    - X: A numpy array of shape (n_samples, n_features) where each row represents a sample.
    - sigma: The bandwidth parameter of the RBF kernel.

    Returns:
    - K: The RBF kernel matrix of shape (n_samples, n_samples).
    """
    # Compute the squared Euclidean distance between samples.
    # The efficient way is to use the identity ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
    X_norm = np.sum(X**2, axis=1)
    distance_sq = X_norm[:, np.newaxis] + X_norm[np.newaxis, :] - 2 * np.dot(X, X.T)

    # Compute the RBF kernel matrix.
    K = np.exp(-distance_sq / (2 * sigma**2))

    return K

def compute_subgradient(Kernel_matrix, Y, alpha):

    grad = np.zeros(len(Kernel_matrix))
    for i in range(len(Kernel_matrix)):
      hinge_loss_indicator = (1 - Y[i] * np.dot(Kernel_matrix[i],alpha)) > 0

    # Subgradient with respect to w
      grad += hinge_loss_indicator * (-Y[i] * Kernel_matrix[i])/len(Y)

    grad += np.dot(Kernel_matrix,alpha)
    return grad

def find_smallest_norm_convex_combination(v, w):
    """
    Find the convex combination of vectors v and w that has the smallest norm.

    Parameters:
    - v: First vector as a NumPy array.
    - w: Second vector as a NumPy array.

    Returns:
    - optimal_combination: Convex combination of v and w with the smallest norm.
    - alpha_opt: The optimal alpha value for the smallest norm convex combination.
    """
    # Define the objective function: norm of the convex combination of v and w
    def objective(alpha, v, w):
        return np.linalg.norm(alpha * v + (1 - alpha) * w)

    # Define the constraint for alpha (0 <= alpha <= 1)
    constraints = ({'type': 'ineq', 'fun': lambda alpha: alpha},  # alpha >= 0
                   {'type': 'ineq', 'fun': lambda alpha: 1 - alpha})  # alpha <= 1

    # Initial guess for alpha
    alpha_init = 0.5

    # Perform the optimization
    result = minimize(objective, alpha_init, args=(v, w), bounds=[(0, 1)], constraints=constraints)

    # Extract the optimal alpha
    alpha_opt = result.x[0]

    # Compute the smallest norm convex combination
    optimal_combination = alpha_opt * v + (1 - alpha_opt) * w

    return optimal_combination

def compute_accuracy(y_true, y_pred):
    """
    Compute the accuracy of predictions.

    Parameters:
    - y_true: numpy array of shape (n_samples,), true labels.
    - y_pred: numpy array of shape (n_samples,), predicted labels.

    Returns:
    - accuracy: float, the accuracy of the predictions.
    """
    correct_predictions = np.sum(y_true == np.sign(y_pred))
    accuracy = correct_predictions / len(y_true)
    return accuracy