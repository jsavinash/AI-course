"""Data loading and preprocessing for GNN social network analysis."""

from pathlib import Path

import numpy as np

N_FEATURES = 32
N_CLASSES = 2
N_NODES = 20

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    n_nodes: int = N_NODES,
    n_features: int = N_FEATURES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic social network data.

    Creates a random graph with clustered node features where connected
    nodes tend to share similar labels (community structure).

    Returns:
        X: (n_nodes, n_features) node features
        A: (n_nodes, n_nodes) adjacency matrix
        y: (n_nodes,) node labels
    """
    rng = np.random.default_rng(random_seed)
    X = rng.random((n_nodes, n_features))

    A = np.zeros((n_nodes, n_nodes))
    n_communities = 4
    nodes_per_comm = n_nodes // n_communities
    for comm in range(n_communities):
        start = comm * nodes_per_comm
        end = start + nodes_per_comm
        comm_nodes = list(range(start, end))
        for i in range(len(comm_nodes)):
            for j in range(i + 1, len(comm_nodes)):
                if rng.random() > 0.3:
                    A[comm_nodes[i], comm_nodes[j]] = 1.0
                    A[comm_nodes[j], comm_nodes[i]] = 1.0

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() > 0.9:
                A[i, j] = 1.0
                A[j, i] = 1.0

    y = np.zeros(n_nodes, dtype=int)
    for comm in range(n_communities):
        start = comm * nodes_per_comm
        end = start + nodes_per_comm
        y[start:end] = comm % N_CLASSES

    perm = rng.permutation(n_nodes)
    X = X[perm]
    y = y[perm]
    A = A[np.ix_(perm, perm)]

    return X, A, y


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["A"], data["y"]
    return generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)


def train_test_split_graph(
    X: np.ndarray, A: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    n_test = max(1, int(n * test_ratio))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[indices[:n_test]] = True
    train_mask = ~test_mask

    return X[train_mask], A[np.ix_(train_mask, train_mask)], y[train_mask], X[test_mask], A[np.ix_(test_mask, test_mask)], y[test_mask], train_mask, test_mask


def save_training_data(X: np.ndarray, A: np.ndarray, y: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, A=A, y=y)
