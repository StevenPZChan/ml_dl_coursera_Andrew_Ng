import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    lr = LogisticRegression(verbose=False)
    lr.fit(x_train, y_train)
    print('theta:', lr.theta)
    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    util.plot(x_eval, y_eval, lr.theta, 'output/tmp.png')
    # Use np.savetxt to save predictions on eval set to pred_path
    y_pred = lr.predict(x_eval)
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        for i in range(self.max_iter):
            z = x @ self.theta
            g = 1 / (1 + np.exp(-z))
            dJ = -1 / m * x.T @ (y - g)
            if self.verbose:
                print(dJ)

            ddJ = 1 / m * x.T @ np.diag(g * (1 - g)) @ x
            dtheta = -np.linalg.inv(ddJ) @ dJ
            if np.linalg.norm(dtheta) < self.eps:
                break
            else:
                self.theta += dtheta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            print('Not fit yet!')
            return None

        z = x @ self.theta
        return 1 / (1 + np.exp(-z))
        # *** END CODE HERE ***
