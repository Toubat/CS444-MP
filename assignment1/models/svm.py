"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
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
        self.batch_size = batch_size
        self.decay_rate = 0.03


    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """

        N = X_train.shape[0] # current batch size
        w_grad = np.zeros(self.w.shape)
        scores = X_train @ self.w  # (N, C)

        for i in range(N):
            x_i, y_i = X_train[i].T, y_train[i] # (D+1, 1), (1, 1)
            I = scores[i, y_i] - scores[i, :] < 1 # (1, C)

            w_grad[:, y_i] -= np.sum(I) * x_i # (D+1, 1)
            w_grad[:, :] += x_i[:, None] @ I[None, :] # (D+1, C)

        # regularization
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

        y_raw = y_train
        max_acc = 0

        # set random seed
        np.random.seed(42)

        # add bias column to X_train
        X_norm = self.normalize(X_train)
        X_norm = np.hstack((X_norm, np.ones((X_norm.shape[0], 1)))) # (N, D+1)

        # Initialize weights from the standard normal distribution
        self.w = np.random.randn(X_norm.shape[1], self.n_class) # (D+1, C)

        print(f"Training SVM...")
        for epoch in range(self.epochs):

            # shuffle data
            idx = np.arange(X_norm.shape[0])
            np.random.shuffle(idx)
            X_norm, y_train = X_norm[idx], y_train[idx]

            for i in range(0, X_norm.shape[0], self.batch_size):
                X_batch = X_norm[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                w_grad = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * w_grad


            accuracy = self.get_acc(self.predict(X_train), y_raw)
            if accuracy > max_acc:
                # store self.w to a file called svm_weights.npy
                max_acc = accuracy
                np.save('svm_weights.npy', self.w)

            print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {accuracy:.2f}")


        return None


    def load_weights(self, weights_path = 'svm_weights.npy'):
        self.w = np.load(weights_path)


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
        X_test = self.normalize(X_test)
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # (N, D+1)

        scores = X_test @ self.w
        pred = np.argmax(scores, axis=1)
        return pred


    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100


    def exp_decay(self, epoch):
        return self.lr * np.exp(-self.decay_rate * epoch)

    def normalize(self, X):
        mean = X.mean(0, keepdims=True)
        std = X.std(0, keepdims=True)
        std += (std == 0.0) * 1e-15
        X = X.astype(np.float64)
        X -= mean
        X /= std

        return X