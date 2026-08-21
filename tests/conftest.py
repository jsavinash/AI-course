"""Shared pytest fixtures for the monorepo."""
import numpy as np
import pytest


@pytest.fixture
def sample_pizza_data():
    X = np.array([6, 8, 10, 14, 18], dtype=float)
    y = np.array([7.0, 9.0, 13.0, 17.5, 18.0], dtype=float)
    return X, y

@pytest.fixture
def sample_spam_data():
    X = np.array(
        [
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    return X, y
