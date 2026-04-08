import numpy as np
import util

from linear_model import LinearModel
import os

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # Create model
    clf = LogisticRegression()

    # Fit model on dataset
    clf.fit(x_train, y_train)
    
    # Plot data and decision boundary
    util.plot(x_train, y_train, clf.theta, f"{'.'.join(pred_path.split('.')[:-1])}.png")

    # Evaluate model on validation set
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        # Save dimensions of x
        m, n = x.shape

        # Initialize theta
        self.theta = self.theta if self.theta is not None else np.zeros(n)

        # Newton's method
        # theta := theta - H^-1 * grad 
        while True:

            # Save current theta
            theta_old = np.copy(self.theta)
            
            # Apply Newton's method
            # Compute gradient of the cost function
            sigm = 1 / (1 + np.exp(-x @ (self.theta)))
            grad = (x.T @ (sigm - y)) / m

            # Compute Hessian
            H = (x.T * sigm * (1-sigm)) @ (x) / m
            H_inv = np.linalg.inv(H)
            self.theta -= H_inv @ grad
            
            # Check exit condition
            if np.sum(np.abs(self.theta - theta_old)) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        return 1 / (1 + np.exp(-x.dot(self.theta)))

        # *** END CODE HERE ***
