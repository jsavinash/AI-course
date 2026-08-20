"""Apriori algorithm for association rule mining in recommendation engines.

Implements a production-ready Apriori algorithm with:
- Frequent itemset mining using the Apriori principle
- Association rule generation with support, confidence, and lift metrics
- Proper serialization with metadata
- Recommendation generation from learned rules
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AssociationRule:
    """An association rule: antecedent -> consequent."""

    antecedent: frozenset[str]
    consequent: frozenset[str]
    support: float
    confidence: float
    lift: float
    conviction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "antecedent": sorted(self.antecedent),
            "consequent": sorted(self.consequent),
            "support": round(self.support, 4),
            "confidence": round(self.confidence, 4),
            "lift": round(self.lift, 4),
            "conviction": round(self.conviction, 4),
        }


@dataclass
class Apriori:
    """Apriori algorithm for mining association rules from transaction data.

    Args:
        min_support: Minimum support threshold (0.0-1.0)
        min_confidence: Minimum confidence threshold (0.0-1.0)
        min_lift: Minimum lift threshold (>= 1.0)
        max_itemset_size: Maximum size of itemsets to consider
    """

    min_support: float = 0.05
    min_confidence: float = 0.5
    min_lift: float = 1.0
    max_itemset_size: int = 4

    # Learned state
    frequent_itemsets: dict[frozenset[str], float] = field(default_factory=dict)
    rules: list[AssociationRule] = field(default_factory=list)
    n_transactions: int = 0
    products: list[str] = field(default_factory=list)

    def fit(self, transactions: list[list[str]]) -> "Apriori":
        """Mine frequent itemsets and generate association rules.

        Args:
            transactions: List of transactions, each a list of product names

        Returns:
            self
        """
        self.n_transactions = len(transactions)
        self.products = sorted(set(item for t in transactions for item in t))

        # Convert transactions to frozensets for efficient operations
        transaction_sets = [frozenset(t) for t in transactions]

        # Step 1: Find frequent itemsets using Apriori
        self.frequent_itemsets = self._find_frequent_itemsets(transaction_sets)

        # Step 2: Generate association rules
        self.rules = self._generate_rules(transaction_sets)

        return self

    def _find_frequent_itemsets(
        self, transaction_sets: list[frozenset[str]]
    ) -> dict[frozenset[str], float]:
        """Find all frequent itemsets using the Apriori algorithm."""
        frequent_itemsets: dict[frozenset[str], float] = {}

        # Generate candidate 1-itemsets
        candidates: set[frozenset[str]] = set()
        for t in transaction_sets:
            for item in t:
                candidates.add(frozenset([item]))

        k = 1
        while candidates and k <= self.max_itemset_size:
            # Count support for each candidate
            support_counts: dict[frozenset[str], int] = {}
            for itemset in candidates:
                count = sum(1 for t in transaction_sets if itemset.issubset(t))
                support_counts[itemset] = count

            # Filter by minimum support
            frequent_k: dict[frozenset[str], float] = {}
            for itemset, count in support_counts.items():
                support = count / self.n_transactions
                if support >= self.min_support:
                    frequent_k[itemset] = support

            if not frequent_k:
                break

            frequent_itemsets.update(frequent_k)

            # Generate next candidates using Apriori principle
            candidates = self._generate_candidates(list(frequent_k.keys()), k + 1)
            k += 1

        return frequent_itemsets

    def _generate_candidates(
        self, frequent_itemsets: list[frozenset[str]], k: int
    ) -> set[frozenset[str]]:
        """Generate candidate itemsets of size k from frequent itemsets of size k-1."""
        candidates: set[frozenset[str]] = set()
        n = len(frequent_itemsets)

        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = frequent_itemsets[i]
                itemset2 = frequent_itemsets[j]

                # Join step: combine if first k-2 items are the same
                list1 = sorted(itemset1)
                list2 = sorted(itemset2)
                if list1[:-1] == list2[:-1]:
                    new_itemset = itemset1 | itemset2
                    if len(new_itemset) == k and self._has_frequent_subsets(
                        new_itemset, frequent_itemsets
                    ):
                        # Prune step: all subsets must be frequent
                        candidates.add(new_itemset)

        return candidates

    def _has_frequent_subsets(
        self, itemset: frozenset[str], frequent_itemsets: list[frozenset[str]]
    ) -> bool:
        """Check if all (k-1)-subsets of the itemset are frequent."""
        frequent_set = set(frequent_itemsets)
        for subset in combinations(itemset, len(itemset) - 1):
            if frozenset(subset) not in frequent_set:
                return False
        return True

    def _generate_rules(self, transaction_sets: list[frozenset[str]]) -> list[AssociationRule]:
        """Generate association rules from frequent itemsets."""
        rules: list[AssociationRule] = []

        for itemset, support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue

            # Generate all possible antecedent/consequent splits
            items = list(itemset)
            for r in range(1, len(items)):
                for antecedent_tuple in combinations(items, r):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent

                    if not consequent:
                        continue

                    # Compute confidence
                    antecedent_support = self.frequent_itemsets.get(antecedent, 0.0)
                    if antecedent_support == 0:
                        continue

                    confidence = support / antecedent_support

                    # Compute lift
                    consequent_support = self.frequent_itemsets.get(consequent, 0.0)
                    if consequent_support == 0:
                        continue

                    lift = confidence / consequent_support

                    # Compute conviction
                    if confidence < 1.0:
                        conviction = (1 - consequent_support) / (1 - confidence)
                    else:
                        conviction = float("inf")

                    # Apply thresholds
                    if confidence >= self.min_confidence and lift >= self.min_lift:
                        rules.append(
                            AssociationRule(
                                antecedent=antecedent,
                                consequent=consequent,
                                support=support,
                                confidence=confidence,
                                lift=lift,
                                conviction=conviction,
                            )
                        )

        # Sort by lift (descending) then confidence (descending)
        rules.sort(key=lambda r: (r.lift, r.confidence), reverse=True)
        return rules

    def recommend(
        self,
        items: list[str],
        top_k: int = 5,
        exclude_purchased: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on the items in the basket.

        Args:
            items: List of items in the user's basket
            top_k: Number of recommendations to return
            exclude_purchased: Whether to exclude items already in the basket

        Returns:
            List of recommendation dicts with item, rule, and metrics
        """
        basket = frozenset(items)
        recommendations: dict[str, dict[str, Any]] = {}

        for rule in self.rules:
            # Check if the antecedent is a subset of the basket
            if rule.antecedent.issubset(basket):
                for item in rule.consequent:
                    if exclude_purchased and item in basket:
                        continue

                    # Score: weighted combination of lift, confidence, and support
                    score = rule.lift * rule.confidence * (1 + rule.support)

                    if item not in recommendations or score > recommendations[item]["score"]:
                        recommendations[item] = {
                            "item": item,
                            "score": round(score, 4),
                            "lift": round(rule.lift, 4),
                            "confidence": round(rule.confidence, 4),
                            "support": round(rule.support, 4),
                            "rule": f"{sorted(rule.antecedent)} -> {item}",
                        }

        # Sort by score descending and return top_k
        sorted_recs = sorted(recommendations.values(), key=lambda r: r["score"], reverse=True)
        return sorted_recs[:top_k]

    def get_rules_for_item(self, item: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Get association rules where the item appears in the consequent."""
        matching = [rule.to_dict() for rule in self.rules if item in rule.consequent]
        return matching[:top_k]

    def get_rules_with_item(self, item: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Get association rules where the item appears in the antecedent."""
        matching = [rule.to_dict() for rule in self.rules if item in rule.antecedent]
        return matching[:top_k]

    def evaluate(self, transactions: list[list[str]]) -> dict[str, float]:
        """Evaluate the quality of mined rules.

        Args:
            transactions: Transaction data to evaluate against

        Returns:
            Dict with evaluation metrics
        """
        n_transactions = len(transactions)
        transaction_sets = [frozenset(t) for t in transactions]

        # Coverage: fraction of transactions that match at least one rule antecedent
        covered = 0
        for t in transaction_sets:
            for rule in self.rules:
                if rule.antecedent.issubset(t):
                    covered += 1
                    break

        coverage = covered / n_transactions if n_transactions > 0 else 0.0

        # Average metrics across rules
        if self.rules:
            avg_confidence = np.mean([r.confidence for r in self.rules])
            avg_lift = np.mean([r.lift for r in self.rules])
            avg_support = np.mean([r.support for r in self.rules])
        else:
            avg_confidence = 0.0
            avg_lift = 0.0
            avg_support = 0.0

        return {
            "n_rules": float(len(self.rules)),
            "n_frequent_itemsets": float(len(self.frequent_itemsets)),
            "coverage": float(coverage),
            "avg_confidence": float(avg_confidence),
            "avg_lift": float(avg_lift),
            "avg_support": float(avg_support),
        }

    # ---------- Serialization ----------

    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        np.savez(
            path,
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            min_lift=self.min_lift,
            max_itemset_size=self.max_itemset_size,
            n_transactions=self.n_transactions,
            products=np.array(self.products, dtype=object),
            # Serialize frequent itemsets as JSON-compatible data
            frequent_itemsets_items=np.array(
                [sorted(keyset) for keyset in self.frequent_itemsets], dtype=object
            ),
            frequent_itemsets_supports=np.array(
                [self.frequent_itemsets[keyset] for keyset in self.frequent_itemsets]
            ),
            # Serialize rules
            rules_antecedent=np.array([sorted(r.antecedent) for r in self.rules], dtype=object),
            rules_consequent=np.array([sorted(r.consequent) for r in self.rules], dtype=object),
            rules_support=np.array([r.support for r in self.rules]),
            rules_confidence=np.array([r.confidence for r in self.rules]),
            rules_lift=np.array([r.lift for r in self.rules]),
            rules_conviction=np.array([r.conviction for r in self.rules]),
        )

    @classmethod
    def load(cls, path: str) -> "Apriori":
        """Load model parameters from disk."""
        data = np.load(path, allow_pickle=True)

        model = cls(
            min_support=float(data["min_support"]),
            min_confidence=float(data["min_confidence"]),
            min_lift=float(data["min_lift"]),
            max_itemset_size=int(data["max_itemset_size"]),
        )
        model.n_transactions = int(data["n_transactions"])
        model.products = [str(p) for p in data["products"]]

        # Restore frequent itemsets
        itemsets_items = data["frequent_itemsets_items"]
        itemsets_supports = data["frequent_itemsets_supports"]
        model.frequent_itemsets = {}
        for items, support in zip(itemsets_items, itemsets_supports, strict=False):
            model.frequent_itemsets[frozenset(str(i) for i in items)] = float(support)

        # Restore rules
        antecedents = data["rules_antecedent"]
        consequents = data["rules_consequent"]
        supports = data["rules_support"]
        confidences = data["rules_confidence"]
        lifts = data["rules_lift"]
        convictions = data["rules_conviction"]

        model.rules = []
        for ant, cons, sup, conf, lift, conv in zip(
            antecedents,
            consequents,
            supports,
            confidences,
            lifts,
            convictions,
            strict=False,
        ):
            model.rules.append(
                AssociationRule(
                    antecedent=frozenset(str(i) for i in ant),
                    consequent=frozenset(str(i) for i in cons),
                    support=float(sup),
                    confidence=float(conf),
                    lift=float(lift),
                    conviction=float(conv),
                )
            )

        return model

    def to_dict(self) -> dict[str, Any]:
        """Return model parameters as a dict."""
        return {
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "min_lift": self.min_lift,
            "max_itemset_size": self.max_itemset_size,
            "n_transactions": self.n_transactions,
            "n_products": len(self.products),
            "n_frequent_itemsets": len(self.frequent_itemsets),
            "n_rules": len(self.rules),
        }
