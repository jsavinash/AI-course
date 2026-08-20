"""Semi-supervised learning model using self-training with logistic regression.

Implements a production-ready semi-supervised learning pipeline with:
- Base model: Logistic Regression (from scratch, numpy-only)
- Self-training: iteratively labels high-confidence unlabeled samples
- Confidence thresholding: only adds pseudo-labels above confidence threshold
- Early stopping: prevents overfitting to noisy pseudo-labels
- Proper serialization with metadata

Semi-supervised learning is useful when:
- Labeled data is scarce or expensive to obtain
- Large amounts of unlabeled data are available
- The model can benefit from the structure of the unlabeled data distribution
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation function with numerical stability."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


@dataclass
class LogisticRegression:
    """Logistic regression for binary classification (from scratch).

    Model: z = X·w + b,  p = sigmoid(z),  prediction = 1 if p >= threshold else 0
    """

    learning_rate: float = 0.1
    n_iterations: int = 2000
    weights: np.ndarray | None = None
    bias: float = 0.0
    loss_history: list[float] = field(default_factory=list)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class for each sample."""
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        z = np.dot(X, self.weights) + self.bias
        return _sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return 1 (positive) if probability >= threshold, else 0 (negative)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Train using gradient descent on Binary Cross-Entropy."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            probs = self.predict_proba(X)
            loss = -np.mean(
                y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9)
            )
            self.loss_history.append(float(loss))

            dw = (1 / n_samples) * np.dot(X.T, (probs - y))
            db = (1 / n_samples) * np.sum(probs - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate the model on labeled data."""
        predictions = self.predict(X)
        accuracy = float(np.mean(predictions == y))

        tp = int(np.sum((predictions == 1) & (y == 1)))
        fp = int(np.sum((predictions == 1) & (y == 0)))
        fn = int(np.sum((predictions == 0) & (y == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        if self.weights is None:
            raise ValueError("Cannot save untrained model")
        np.savez(
            path,
            weights=self.weights,
            bias=self.bias,
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            loss_history=np.array(self.loss_history),
        )

    @classmethod
    def load(cls, path: str) -> "LogisticRegression":
        """Load model parameters from disk."""
        data = np.load(path)
        model = cls(
            learning_rate=float(data["learning_rate"]),
            n_iterations=int(data["n_iterations"]),
        )
        model.weights = data["weights"]
        model.bias = float(data["bias"])
        model.loss_history = list(data["loss_history"])
        return model


@dataclass
class SelfTrainingClassifier:
    """Self-training classifier for semi-supervised learning.

    Iteratively:
    1. Train on labeled data
    2. Predict on unlabeled data
    3. Add high-confidence predictions to labeled set
    4. Retrain until convergence or max iterations

    Args:
        base_model: Base classifier to use (default: LogisticRegression)
        confidence_threshold: Minimum probability to add pseudo-label (0.0 to 1.0)
        max_iterations: Maximum number of self-training iterations
        min_labeled_ratio: Stop if labeled ratio exceeds this (prevents overfitting)
        random_seed: Random seed for reproducibility
    """

    confidence_threshold: float = 0.95
    max_iterations: int = 10
    min_labeled_ratio: float = 0.8
    random_seed: int = 42

    # Learned state
    model: LogisticRegression | None = None
    n_features: int = 0
    n_labeled_history: list[int] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)
    n_iterations_used: int = 0
    training_mode: Literal["supervised", "semi-supervised"] = "supervised"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
    ) -> "SelfTrainingClassifier":
        """Fit the self-training classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Label vector with -1 for unlabeled samples
            X_test: Optional test features for tracking accuracy
            y_test: Optional test labels for tracking accuracy

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples = len(X)
        self.n_features = X.shape[1]

        # Get initial labeled data
        X_labeled, y_labeled = self._get_labeled(X, y)
        X_unlabeled = self._get_unlabeled(X, y)

        self.n_labeled_history = [len(X_labeled)]
        self.accuracy_history = []
        self.n_iterations_used = 0

        for _iteration in range(self.max_iterations):
            # Train base model on current labeled data
            self.model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
            self.model.fit(X_labeled, y_labeled)
            self.n_iterations_used += 1

            # Track accuracy on test set if provided
            if X_test is not None and y_test is not None:
                test_metrics = self.model.evaluate(X_test, y_test)
                self.accuracy_history.append(test_metrics["accuracy"])

            # Check if we should stop
            labeled_ratio = len(X_labeled) / n_samples
            if labeled_ratio >= self.min_labeled_ratio:
                self.training_mode = "semi-supervised"
                break

            if len(X_unlabeled) == 0:
                self.training_mode = "semi-supervised"
                break

            # Predict on unlabeled data
            probas = self.model.predict_proba(X_unlabeled)

            # For binary classification, confidence is max(proba, 1 - proba)
            max_probas = np.maximum(probas, 1 - probas)

            # Find high-confidence predictions
            confident_mask = max_probas >= self.confidence_threshold
            confident_indices = np.where(confident_mask)[0]

            if len(confident_indices) == 0:
                # No confident predictions, stop early
                self.training_mode = "semi-supervised" if len(X_labeled) > np.sum(y != -1) else "supervised"
                break

            # Add confident predictions to labeled set
            # Pseudo-label: 1 if proba >= 0.5, else 0
            pseudo_labels = (probas[confident_indices] >= 0.5).astype(int)
            X_labeled = np.vstack([X_labeled, X_unlabeled[confident_indices]])
            y_labeled = np.concatenate([y_labeled, pseudo_labels])

            # Remove added samples from unlabeled set
            mask = np.ones(len(X_unlabeled), dtype=bool)
            mask[confident_indices] = False
            X_unlabeled = X_unlabeled[mask]

            self.n_labeled_history.append(len(X_labeled))
            self.training_mode = "semi-supervised"

        # Final training on all accumulated labeled data
        self.model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
        self.model.fit(X_labeled, y_labeled)
        self.n_iterations_used += 1

        # If we never added pseudo-labels, mark as supervised
        if len(self.n_labeled_history) == 1 or self.n_labeled_history[-1] == self.n_labeled_history[0]:
            self.training_mode = "supervised"

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate the model on labeled data."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.evaluate(X, y)

    def _get_labeled(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract labeled samples."""
        mask = y != -1
        return X[mask], y[mask]

    def _get_unlabeled(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Extract unlabeled samples."""
        mask = y == -1
        return X[mask]

    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        np.savez(
            path,
            model_weights=self.model.weights,
            model_bias=self.model.bias,
            model_learning_rate=np.array([self.model.learning_rate]),
            model_n_iterations=np.array([self.model.n_iterations]),
            model_loss_history=np.array(self.model.loss_history),
            n_features=np.array([self.n_features]),
            confidence_threshold=np.array([self.confidence_threshold]),
            max_iterations=np.array([self.max_iterations]),
            min_labeled_ratio=np.array([self.min_labeled_ratio]),
            random_seed=np.array([self.random_seed]),
            n_iterations_used=np.array([self.n_iterations_used]),
            training_mode=np.array([self.training_mode]),
            n_labeled_history=np.array(self.n_labeled_history),
            accuracy_history=np.array(self.accuracy_history),
        )

    @classmethod
    def load(cls, path: str) -> "SelfTrainingClassifier":
        """Load model parameters from disk."""
        data = np.load(path)

        model = LogisticRegression(
            learning_rate=float(data["model_learning_rate"].item()),
            n_iterations=int(data["model_n_iterations"].item()),
        )
        model.weights = data["model_weights"]
        model.bias = float(data["model_bias"].item())
        model.loss_history = list(data["model_loss_history"])

        clf = cls(
            confidence_threshold=float(data["confidence_threshold"].item()),
            max_iterations=int(data["max_iterations"].item()),
            min_labeled_ratio=float(data["min_labeled_ratio"].item()),
            random_seed=int(data["random_seed"].item()),
        )
        clf.model = model
        clf.n_features = int(data["n_features"].item())
        clf.n_iterations_used = int(data["n_iterations_used"].item())
        clf.training_mode = str(data["training_mode"].item())
        clf.n_labeled_history = list(data["n_labeled_history"])
        clf.accuracy_history = list(data["accuracy_history"])

        return clf

    def to_dict(self) -> dict:
        """Return model parameters as a dict."""
        return {
            "n_features": self.n_features,
            "confidence_threshold": self.confidence_threshold,
            "max_iterations": self.max_iterations,
            "min_labeled_ratio": self.min_labeled_ratio,
            "n_iterations_used": self.n_iterations_used,
            "training_mode": self.training_mode,
            "n_labeled_history": self.n_labeled_history,
            "accuracy_history": self.accuracy_history,
            "final_n_labeled": self.n_labeled_history[-1] if self.n_labeled_history else 0,
            "final_accuracy": self.accuracy_history[-1] if self.accuracy_history else 0.0,
        }
