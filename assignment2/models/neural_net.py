"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """
    A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.
    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """
        Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following

        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        # Adam parameters
        self.t = 0
        self.m = {}
        self.v = {}

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            self.m["W" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
            self.v["W" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
            self.m["b" + str(i)] = np.zeros(self.params["b" + str(i)].shape)
            self.v["b" + str(i)] = np.zeros(self.params["b" + str(i)].shape)


    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        return X @ W + b


    def linear_grad(self, W: np.ndarray) -> np.ndarray:
        """Gradient of the linear layer.
        Parameters:
            W: the weight matrix
            X: the input data
        Returns:
            the gradient of the linear layer
        """
        return W.T


    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        return np.maximum(X, 0)


    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        return X > 0


    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # avoid overflow
        score = X - np.max(X, axis=1, keepdims=True)
        softmax_score = np.exp(score) / np.sum(np.exp(score), axis=1, keepdims=True)

        return softmax_score


    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        X = self.normalize(X)
        self.outputs = {'Relu0': X}
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.

        for i in range(1, self.num_layers + 1):
            W, b = self.params[f'W{i}'], self.params[f'b{i}']

            # x = f(x, W)
            X = self.linear(W, X, b)
            self.outputs[f'Linear{i}'] = X

            # x = max(0, x)
            if i < self.num_layers:
                X = self.relu(X)
                self.outputs[f'Relu{i}'] = X

        X = self.softmax(X)
        self.outputs['Softmax'] = X

        return X


    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Note: both gradients and loss should include regularization.
        Parameters:
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            reg: Regularization strength
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        # one-hot encode y
        y_one_hot = np.zeros((y.shape[0], self.output_size))
        y_one_hot[np.arange(y.shape[0]), y] = 1

        # compute upstream gradient
        grads = self.outputs['Softmax'] - y_one_hot # (N, C)

        # compute gradients for each layer
        for i in range(self.num_layers, 0, -1):
            if i != self.num_layers:
                grads = grads * self.relu_grad(self.outputs[f'Linear{i}'])

            self.gradients[f'W{i}'] = self.outputs[f'Relu{i-1}'].T @ grads + reg * self.params[f'W{i}']
            self.gradients[f'b{i}'] = np.sum(grads, axis=0)
            grads = grads @ self.params[f'W{i}'].T

        assert (self.outputs['Softmax'] < 0).sum() == 0

        prob = self.outputs['Softmax'][np.arange(y.shape[0]), y]

        return np.sum(-np.log(prob)) / y.shape[0]


    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
            for key, val in self.gradients.items():
                self.params[key] -= lr * val
        elif opt == "Adam":
            self.t += 1
            for param_key in self.gradients.keys():
                self.adam(lr, b1, b2, eps, param_key)


    def adam(self, lr: float, b1: float, b2: float, eps: float, param_key: str):
        """Adam update.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            param_key: the key for the parameter to update
        """
        grad = self.gradients[param_key]
        self.m[param_key] = b1 * self.m[param_key] + (1 - b1) * grad
        self.v[param_key] = b2 * self.v[param_key] + (1 - b2) * grad ** 2
        m_hat = self.m[param_key] / (1 - b1 ** self.t)
        v_hat = self.v[param_key] / (1 - b2 ** self.t)
        self.params[param_key] -= lr * m_hat / (np.sqrt(v_hat) + eps)


    def save_checkpoint(self, file_name: str):
        """Save the model's state to a file.
        Parameters:
            file_name: Name of the file to save the state to
        """
        checkpoint = {
            'params': self.params,
            'm': self.m,
            'v': self.v,
            't': self.t,
        }
        np.save(file_name, checkpoint)


    def load_checkpoint(self, file_name: str):
        """Load the model's state from a file.
        Parameters:
            file_name: Name of the file to load the state from
        """
        checkpoint = np.load(file_name, allow_pickle=True).item()
        self.params = checkpoint['params']
        self.m = checkpoint['m']
        self.v = checkpoint['v']
        self.t = checkpoint['t']


    def normalize(self, X):
        X = X.astype(np.float64)
        X -= X.mean(0, keepdims=True)
        X /= X.std(0, keepdims=True) + (X.std(0, keepdims=True) == 0.0) * 1e-15

        return X
