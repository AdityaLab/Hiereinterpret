import numpy as np
import pandas as pd


def generate_synthetic_timeseries(
    phi, means, std_devs, correlation_matrix, num_timesteps=100
):
    """
    Generates a multivariate time series using an autoregressive process (AR) and covariance matrix.

    Parameters:
    - phi: List of AR coefficients (e.g., [0.8] for AR(1), [0.6, 0.2] for AR(2)).
    - covariance_matrix: Covariance matrix (num_variables x num_variables).
    - num_timesteps: Number of time steps to generate.

    Returns:
    - Pandas DataFrame containing the time series.
    """
    assert std_devs.ndim == 1, "std_devs must be a 1D array"
    assert correlation_matrix.ndim == 2, "correlation_matrix must be a 2D array"
    assert (
        correlation_matrix.shape[0] == correlation_matrix.shape[1]
    ), "correlation_matrix must be square"
    assert (
        len(std_devs) == correlation_matrix.shape[0]
    ), "std_devs and correlation_matrix must have the same length"
    covariance_matrix = np.outer(std_devs, std_devs) * correlation_matrix
    num_variables = covariance_matrix.shape[0]  # Number of variables

    # Check that the length of phi is consistent with the AR order
    ar_order = len(phi)

    # Initialize time series with zeros
    time_series = np.zeros((num_timesteps, num_variables))

    # Generate the first `ar_order` steps using multivariate normal (initial condition)
    initial_steps = np.random.multivariate_normal(
        means, covariance_matrix, size=ar_order
    )
    time_series[:ar_order, :] = initial_steps

    # Generate AR time series from step `ar_order` onwards
    for t in range(ar_order, num_timesteps):
        # Initialize current time step with zeros
        time_step_value = np.zeros(num_variables)

        # Add contributions from past `ar_order` steps
        for lag in range(1, ar_order + 1):
            time_step_value += phi[lag - 1] * time_series[t - lag, :]

        # Add noise (Gaussian multivariate noise)
        noise = np.random.multivariate_normal(
            np.zeros(num_variables), covariance_matrix
        )
        time_series[t, :] = time_step_value + noise

    # Create a DataFrame for better representation
    columns = [f"Variable_{i+1}" for i in range(num_variables)]
    time_series_df = pd.DataFrame(time_series, columns=columns)

    return time_series_df


if __name__ == "__main__":
    num_variables = 5
    phi = [0.8, 0.2]
    correlation_matrix = np.eye(num_variables)  # Identity matrix
    # Add some correlation between variables
    correlation_matrix += np.random.normal(0, 0.4, size=(num_variables, num_variables))
    # Make the correlation matrix symmetric
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    # make the diagonal values equal to 1
    np.fill_diagonal(correlation_matrix, 1)

    std_devs = np.random.uniform(0.5, 2, num_variables)
    means = np.random.uniform(-1, 1, num_variables)
    time_series = generate_synthetic_timeseries(
        phi, means, std_devs, correlation_matrix, num_timesteps=100
    )
