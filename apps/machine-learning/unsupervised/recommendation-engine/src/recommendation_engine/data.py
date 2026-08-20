"""Data loading and preprocessing for the recommendation engine.

Generates a realistic synthetic grocery transaction dataset with
known product associations, designed for Apriori association rule mining.

Product categories and their associations:
- Bread, Milk, Butter  - breakfast staples (frequently bought together)
- Beer, Diapers, Chips - classic "beer & diapers" association
- Coffee, Sugar, Cream - coffee lovers
- Pasta, Sauce, Cheese - Italian dinner
- Shampoo, Soap, Toothpaste - personal care
"""

from pathlib import Path

import numpy as np
import pandas as pd

# All possible products in the catalog
PRODUCTS = [
    "Bread",
    "Milk",
    "Butter",
    "Beer",
    "Diapers",
    "Chips",
    "Coffee",
    "Sugar",
    "Cream",
    "Pasta",
    "Sauce",
    "Cheese",
    "Shampoo",
    "Soap",
    "Toothpaste",
    "Eggs",
    "Juice",
    "Cereal",
    "Yogurt",
    "Chocolate",
    "Fruit",
    "Granola",
]

# Product groups with strong associations (for generating realistic data)
PRODUCT_GROUPS = [
    ["Bread", "Milk", "Butter"],
    ["Beer", "Diapers", "Chips"],
    ["Coffee", "Sugar", "Cream"],
    ["Pasta", "Sauce", "Cheese"],
    ["Shampoo", "Soap", "Toothpaste"],
    ["Eggs", "Milk", "Bread"],
    ["Cereal", "Milk", "Sugar"],
    ["Yogurt", "Fruit", "Granola"],
    ["Chocolate", "Milk", "Sugar"],
    ["Juice", "Cereal", "Bread"],
]

# Default number of transactions
DEFAULT_N_TRANSACTIONS = 1000

# Maximum items per transaction
MAX_ITEMS_PER_TRANSACTION = 8


def _generate_transactions(
    n_transactions: int = DEFAULT_N_TRANSACTIONS,
    random_seed: int = 42,
) -> list[list[str]]:
    """Generate synthetic transaction data with known product associations.

    Returns:
        List of transactions, where each transaction is a list of product names
    """
    rng = np.random.default_rng(random_seed)

    transactions: list[list[str]] = []

    for _ in range(n_transactions):
        # Each transaction has 1-8 items
        n_items = rng.integers(1, MAX_ITEMS_PER_TRANSACTION + 1)

        # 70% chance to pick from a product group (creates strong associations)
        if rng.random() < 0.7:
            group = PRODUCT_GROUPS[rng.integers(0, len(PRODUCT_GROUPS))]
            # Pick 2-3 items from the group
            n_group_items = rng.integers(2, min(4, len(group) + 1))
            items = list(rng.choice(group, size=n_group_items, replace=False))
        else:
            items = []

        # Fill remaining slots with random products
        while len(items) < n_items:
            product = PRODUCTS[rng.integers(0, len(PRODUCTS))]
            if product not in items:
                items.append(product)

        transactions.append(items)

    return transactions


def load_training_data(
    data_path: Path | None = None,
    n_transactions: int = DEFAULT_N_TRANSACTIONS,
    random_seed: int = 42,
) -> list[list[str]]:
    """Load transaction data from CSV or generate a synthetic dataset.

    Expected CSV format (one transaction per row, comma-separated items):
        Bread,Milk,Butter
        Beer,Diapers,Chips
        ...

    Returns:
        List of transactions, where each transaction is a list of product names
    """
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path, header=None)
        transactions = []
        for _, row in df.iterrows():
            items = [str(item).strip() for item in row.dropna().tolist() if str(item).strip()]
            if items:
                transactions.append(items)
        return transactions

    return _generate_transactions(n_transactions=n_transactions, random_seed=random_seed)


def save_training_data(transactions: list[list[str]], path: Path) -> None:
    """Save transaction data to CSV for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Pad transactions to same length for CSV
    max_len = max(len(t) for t in transactions)
    padded = [t + [""] * (max_len - len(t)) for t in transactions]
    df = pd.DataFrame(padded)
    df.to_csv(path, index=False, header=False)


def transactions_to_onehot(
    transactions: list[list[str]],
    products: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert transactions to a one-hot encoded matrix.

    Args:
        transactions: List of transactions (each a list of product names)
        products: Optional list of all products. If None, derived from data.

    Returns:
        Tuple of (one-hot matrix, product names)
    """
    if products is None:
        products = sorted(set(item for t in transactions for item in t))

    product_to_idx = {p: i for i, p in enumerate(products)}
    n_transactions = len(transactions)
    n_products = len(products)

    X = np.zeros((n_transactions, n_products), dtype=int)
    for i, t in enumerate(transactions):
        for item in t:
            if item in product_to_idx:
                X[i, product_to_idx[item]] = 1

    return X, products
