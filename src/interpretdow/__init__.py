import numpy as np

np.set_printoptions(suppress=True)


def hello() -> str:
    return "Hello from interpretdow!"


def sigmoid(x: np.ndarray):
    """
    Applies the sigmoid function to each element of a numpy array

    Args:
        x (np.ndarray): Numpy array

    Returns:
        (np.ndarray): The sigmoid of each element in `x
    """
    return 1 / (1 + np.exp(-x))
