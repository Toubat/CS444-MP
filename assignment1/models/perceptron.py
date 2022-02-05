"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
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

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        # Initialize weights from the standard normal distribution
        self.w = np.random.randn(X_train.shape[1], self.n_class) # (D, C)

        print(f"Training Perceptron...")

        for epoch in range(self.epochs):
            scores = X_train @ self.w   # (N, C)


            for i in range(X_train.shape[0]):
                x_i, y_i = X_train[i].T, y_train[i]  # (D, 1), (1, 1)

                for c in range(self.n_class):
                    if c == y_i or scores[i, c] <= scores[i, y_i]:
                        continue
                    self.w[:, c] -= self.lr * x_i
                    self.w[:, y_train[i]] += self.lr * x_i

            print(f"Epoch {epoch}, Accuracy: {self.get_acc(self.predict(X_train), y_train):.2f}%")

        return None


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
