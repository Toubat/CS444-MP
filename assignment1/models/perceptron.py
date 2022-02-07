"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, decay_rate: float = 0.03, lr_decay: bool = True):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.decay_rate = decay_rate
        self.lr_decay = lr_decay


    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        # set random seed
        np.random.seed(42)
        # Initialize weights from the standard normal distribution
        self.w = np.random.randn(X_train.shape[1], self.n_class) # (D, C)

        accuracy = np.zeros((self.epochs, 1))

        print(f"Training Perceptron...")

        for epoch in range(self.epochs):
            w_grad = np.zeros(self.w.shape) # (D, C)

            scores = X_train @ self.w  # (N, C)
            scores_y = scores[np.arange(scores.shape[0]), y_train][:, None] # (N, 1)
            I = scores > scores_y # (N, C)

            for i in range(X_train.shape[0]):
                x_i, y_i = X_train[i].T, y_train[i] # (D, 1), (1, 1)
                w_grad[:, y_i] -= np.sum(I[i, :]) * x_i # (D, 1)

            w_grad[:, :] += X_train.T @ I # (D, C)

            # proceed to update weights by gradient descent
            if self.lr_decay:
                self.w -= self.exp_decay(epoch) * w_grad
            else:
                self.w -= self.lr * w_grad

            print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {self.get_acc(self.predict(X_train), y_train):.2f}%")
            accuracy[epoch] = self.get_test_acc(X_test, y_test)

        return accuracy


    # function that takes training, validation and test data and an array, return the accuracy of each of them
    def get_test_acc(self, X_test, y_test):
        # get the predictions
        pred_test = self.predict(X_test)
        acc_test = self.get_acc(pred_test, y_test)

        return np.array([acc_test])

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """

        scores = X_test @ self.w  # (N, C)
        pred = np.argmax(scores, axis=1)

        return pred


    def get_acc(self, pred, y_test):

        return np.sum(y_test == pred) / len(y_test) * 100


    def exp_decay(self, epoch):
        return self.lr * np.exp(-self.decay_rate * epoch)