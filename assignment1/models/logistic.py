"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.decay_rate = 0.01


    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        return 1 / (1 + np.exp(-z))


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        # set random seed
        # np.random.seed(42)
        # Initialize weights from the standard normal distribution
        self.w = np.random.randn(X_train.shape[1], 1) # (D, 1)

        # normalize X_train
        X_train = self.normalize(X_train)
        y_norm = np.where(y_train == 0, -1, y_train)

        for epoch in range(self.epochs):
            w_grad = -(y_norm[:, None] * X_train).T @ self.sigmoid(-y_norm[:,None] * (X_train @ self.w))

            # (N, 1) * (N, D) -> (N, D) -> (D, N) @ (N, 1) -> (D, 1)

            # for i in range(X_train.shape[0]):
            #     x_i, y_i = X_train[i].T, y_train[i] # (D, 1), (1, 1)

            #     w_grad -= self.sigmoid(-y_i * (self.w.T @ x_i)) *  y_i * x_i[:, None] # (D, 1)

            self.w -= self.exp_decay(epoch) * w_grad

            print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {self.get_acc(self.predict(X_train), y_train):.2f}%")

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
        # normalize X_train
        X_test = self.normalize(X_test)
        pred = self.sigmoid(X_test @ self.w) > self.threshold

        return pred.squeeze()


    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100


    def exp_decay(self, epoch):
        return self.lr * np.exp(-self.decay_rate * epoch)


    def normalize(self, X):
        X = X.astype(np.float64)
        X -= X.mean(0, keepdims=True)
        X /= X.std(0, keepdims=True) + (X.std(0, keepdims=True) == 0.0) * 1e-15

        return X
