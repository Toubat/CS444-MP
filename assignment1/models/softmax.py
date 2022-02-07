"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, batch_size: int = 64):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.decay_rate = 0.05
        self.batch_size = batch_size


    def softmax(self, scores: np.ndarray) -> np.ndarray:
        """Compute softmax values for each sets of scores in x."""
        # avoid overflow
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores) # (N, C)

        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


    def load_weights(self, weights_path: str):
        """Load weights from a file.

        Parameters:
            weights_path: path to the file containing the weights
        """
        self.w = np.load(weights_path)


    def calc_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        scores = X_batch @ self.w
        w_grad = np.zeros(self.w.shape) # (D, C)

        softmax_scores = self.softmax(scores) # (N, C)

        # one hot ecode y_batch
        y_batch_one_hot = np.zeros((y_batch.shape[0], self.n_class)) # (N, C)
        y_batch_one_hot[np.arange(y_batch.shape[0]), y_batch] = 1 # (N, C)

        w_grad[:, :] += X_batch.T @ (softmax_scores - y_batch_one_hot) # (D, C)
        w_grad[:, :] += self.reg_const * self.w

        return w_grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # set random seed
        # np.random.seed(42)

        # add bias column to X_train
        X_norm = self.normalize(X_train)
        self.w = np.random.randn(X_norm.shape[1], self.n_class)

        max_acc = 0

        print(f"Training Softmax...")
        for epoch in range(self.epochs):

            for i in range(0, X_norm.shape[0], self.batch_size):
                X_batch = X_norm[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                w_grad = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * w_grad

            accuracy = self.get_acc(self.predict(X_train), y_train)
            if accuracy > max_acc:
                # store self.w to a file called svm_weights.npy
                max_acc = accuracy
                np.save('softmax_weights.npy', self.w)

            print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {accuracy:.2f}")


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
        X_norm = self.normalize(X_test)
        scores = X_norm @ self.w
        return np.argmax(scores, axis=1)


    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100


    def normalize(self, X):
        mean = X.mean(0, keepdims=True)
        std = X.std(0, keepdims=True)
        std += (std == 0.0) * 1e-15
        X = X.astype(np.float64)
        X -= mean
        X /= std

        return X


    def exp_decay(self, epoch):
        return self.lr * np.exp(-self.decay_rate * epoch)