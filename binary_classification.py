from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import pandas as pd
import numpy as np
import keras


data = pd.DataFrame(pd.read_csv('data.csv'))
X = data[['acousticness', 'danceability', 'energy', 'valence', 'instrumentalness', 'loudness', 'tempo']]
Y = data['popularity']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=420, shuffle=True)


# implementing a sigmoid activation function
def sigmoid(z):
  s = 1.0 / (1 + np.exp(-z))
  return s


def network_architecture(X, Y):
  # nodes in input layer
  n_x = X.shape[0]
  # nodes in hidden layer
  n_h = 10
  # nodes in output layer
  n_y = Y.shape[0]
  return (n_x, n_h, n_y)


def define_network_parameters(n_x, n_h, n_y):
  W1 = np.random.randn(n_h, n_x) * 0.01  # random initialization
  b1 = np.zeros((n_h, 1))  # zero initialization
  W2 = np.random.randn(n_y, n_h) * 0.01
  b2 = np.zeros((n_y, 1))
  return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_propagation(X, params):
  Z1 = np.dot(params['W1'], X) + params['b1']
  A1 = sigmoid(Z1)

  Z2 = np.dot(params['W2'], A1) + params['b2']
  A2 = sigmoid(Z2)
  return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


def compute_error(Predicted, Actual):
  logprobs = np.multiply(np.log(Predicted), Actual) + np.multiply(np.log(1 - Predicted), 1 - Actual)
  cost = -np.sum(logprobs) / Actual.shape[1]
  return np.squeeze(cost)


def backward_propagation(params, activations, X, Y):
  m = X.shape[1]

  # output layer
  dZ2 = activations['A2'] - Y  # compute the error derivative
  dW2 = np.dot(dZ2, activations['A1'].T) / m  # compute the weight derivative
  db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # compute the bias derivative

  # hidden layer
  dZ1 = np.dot(params['W2'].T, dZ2) * (1 - np.power(activations['A1'], 2))
  dW1 = np.dot(dZ1, X.T) / m
  db1 = np.sum(dZ1, axis=1, keepdims=True) / m

  return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(params, derivatives, alpha=1.2):
  # alpha is the model's learning rate

  params['W1'] = params['W1'] - alpha * derivatives['dW1']
  params['b1'] = params['b1'] - alpha * derivatives['db1']
  params['W2'] = params['W2'] - alpha * derivatives['dW2']
  params['b2'] = params['b2'] - alpha * derivatives['db2']
  return params


def neural_network(X, Y, n_h, num_iterations=100):
  n_x = network_architecture(X, Y)[0]
  n_y = network_architecture(X, Y)[2]

  params = define_network_parameters(n_x, n_h, n_y)
  for i in range(0, num_iterations):
    results = forward_propagation(X, params)
    error = compute_error(results['A2'], Y)
    derivatives = backward_propagation(params, results, X, Y)
    params = update_parameters(params, derivatives)
  return params


y = Y.values.reshape(1, Y.size)
x = X.T.as_matrix()
model = neural_network(x, y, n_h=10, num_iterations=10)


def predict(parameters, X):
  results = forward_propagation(X, parameters)
  print(results['A2'][0])
  predictions = np.around(results['A2'])
  return predictions


predictions = predict(model, x)
print('Accuracy: %d' % float((np.dot(y, predictions.T) + np.dot(1 - y, 1 - predictions.T)) / float(y.size) * 100) + '%')
