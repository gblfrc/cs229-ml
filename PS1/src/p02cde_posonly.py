import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    # Load dataset
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    # Create and fit model
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    # Plot data and decision boundary
    util.plot(x_test, y_test, clf.theta, f"{'.'.join(pred_path_c.split('.')[:-1])}.png")
    # Evaluate model on validation set
    y_pred = clf.predict(x_test)
    np.savetxt(pred_path_c, y_pred > 0.5, fmt='%d')

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    # Load dataset
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    # Create and fit model
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    # Plot data and decision boundary
    util.plot(x_test, y_test, clf.theta, f"{'.'.join(pred_path_d.split('.')[:-1])}.png")
    # Evaluate model on validation set
    y_pred = clf.predict(x_test)
    np.savetxt(pred_path_d, y_pred > 0.5, fmt='%d')

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    # Load dataset
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    # Estimate alpha
    v_plus = x_valid[y_valid == 1]
    h_plus = clf.predict(v_plus)
    alpha = (1/len(v_plus))*np.sum(h_plus)
    # Compute new theta for decision boundary
    theta_new = clf.theta.copy()
    theta_new[0] += np.log(2/alpha - 1)
    # Plot data and decision boundary
    util.plot(x_test, y_test, theta_new, f"{'.'.join(pred_path_e.split('.')[:-1])}.png")
    # Evaluate model on validation set
    y_pred = clf.predict(x_test)/alpha
    np.savetxt(pred_path_e, y_pred > 0.5, fmt='%d')

    # *** END CODER HERE
