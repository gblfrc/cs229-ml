import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    best_mse = np.inf   # Initialize variable to store best MSE
    for tau in tau_values:
        # Create and fit model
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train,y_train)
        # Predict on validation set
        y_pred = model.predict(x_valid)
        # Compute MSE and check if best
        mse = np.sum((y_pred-y_valid)**2)/y_pred.shape[0]
        if mse < best_mse:
            best_mse = mse
            best_tau = tau
        # Plot data
        plt.figure()
        plt.plot(x_valid, y_valid, 'bx')
        plt.plot(x_valid, y_pred, 'ro')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"tau: {tau}, MSE: {mse:.3f}")
    
    # Fit a LWR model with the best tau value
    tau = best_tau
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = model.predict(x_test)
    mse = np.sum((y_pred-y_test)**2)/y_pred.shape[0]

    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)

    # Plot data
    plt.figure()
    plt.plot(x_test, y_test, 'bx')
    plt.plot(x_test, y_pred, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"tau: {tau}, MSE: {mse:.3f}")
    plt.savefig(f"{'.'.join(pred_path.split('.')[:-1])}.png")

    # *** END CODE HERE ***