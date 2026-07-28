import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def calculate_gradient(theta, X, y):
    m = y.size # number of examples
    return (X.T @ (sigmoid(X @ theta) - y))/ m

def gradient_descent(theta, X, y, alpha, num_iters):
    X_b = np.c_[np.ones((X.shape[0], 1)), X] # add bias term
    theta = np.zeros(X_b.shape[1]) # initialize theta

    tol = 1e-6 # tolerance for convergence

    for _ in range(num_iters):
        grad = calculate_gradient(theta, X_b, y)
        theta -= alpha * grad
        if np.linalg.norm(grad) < tol:
            break
    return theta
def predict_prob(theta, X):
    X_b = np.c_[np.ones((X.shape[0], 1)), X] # add bias term
    return sigmoid(X_b @ theta)

def predict(theta, X, threshold=0.5):
    return (predict_prob(theta, X) >= threshold).astype(int)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train logistic regression model
alpha = 0.01

theta_hat = gradient_descent(X_train_scaled, y_train, alpha, num_iters=1000)
y_pred_train = predict(theta_hat, X_train_scaled)
y_pred_test = predict(theta_hat, X_test_scaled)
# Evaluate model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")