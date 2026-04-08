import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # Create model
    clf = GDA()

    # Fit model on dataset
    clf.fit(x_train, y_train)
    
    # Plot data and decision boundary
    util.plot(x_train, y_train, clf.theta, f"{'.'.join(pred_path.split('.')[:-1])}.png")

    # Evaluate model on validation set
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***

        # Save length of training set as convenience variable
        m = len(x)

        # Estimate parameters
        phi = (1/m) * sum(y)
        mu_0 = np.array([np.mean(x[y==0,i]) for i in range(x.shape[1])])
        mu_1 = np.array([np.mean(x[y==1,i]) for i in range(x.shape[1])])
        diff_0 = x[y==0] - mu_0
        sums_0 = diff_0.T.dot(diff_0)
        diff_1 = x[y==1] - mu_1
        sums_1 = diff_1.T.dot(diff_1)
        sigma = (1/m)*(sums_0+sums_1)

        # Compute theta
        self.theta = np.zeros(3)
        sigma_inv = np.linalg.inv(sigma)
        self.theta[1:] = sigma_inv.dot(mu_1-mu_0)
        self.theta[0] = 0.5*(mu_0+mu_1).T.dot(sigma_inv).dot(mu_0-mu_1) - np.log((1-phi)/phi)

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
        
        # *** END CODE HERE
