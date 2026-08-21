"""Unit tests for recommendation-engine model."""

from recommendation_engine.model import Apriori


class TestRecommendationEngineApriori:
    """Tests for the recommendation engine Apriori model."""

    def _make_transactions(self) -> list[list[str]]:
        """Generate simple transaction data with known associations."""
        return [
            ["Bread", "Milk", "Butter"],
            ["Bread", "Milk"],
            ["Bread", "Butter"],
            ["Milk", "Butter"],
            ["Bread", "Milk", "Butter"],
            ["Beer", "Diapers", "Chips"],
            ["Beer", "Diapers"],
            ["Beer", "Chips"],
            ["Diapers", "Chips"],
            ["Beer", "Diapers", "Chips"],
            ["Coffee", "Sugar"],
            ["Coffee", "Cream"],
            ["Sugar", "Cream"],
            ["Coffee", "Sugar", "Cream"],
            ["Coffee", "Sugar"],
        ]

    def test_fit_finds_frequent_itemsets(self):
        """Test that fit discovers frequent itemsets."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        assert len(model.frequent_itemsets) > 0
        assert model.n_transactions == len(transactions)
        assert len(model.products) > 0

    def test_fit_generates_rules(self):
        """Test that fit generates association rules."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        assert len(model.rules) > 0
        for rule in model.rules:
            assert rule.confidence >= 0.5
            assert rule.lift >= 1.0
            assert rule.support >= 0.2

    def test_recommend_returns_items(self):
        """Test that recommend returns relevant items."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        recs = model.recommend(["Bread", "Milk"], top_k=5)
        assert len(recs) > 0
        for rec in recs:
            assert "item" in rec
            assert "score" in rec
            assert "lift" in rec
            assert "confidence" in rec

    def test_recommend_excludes_purchased(self):
        """Test that recommend excludes items already in the basket."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        recs = model.recommend(["Bread", "Milk"], top_k=10, exclude_purchased=True)
        for rec in recs:
            assert rec["item"] not in ["Bread", "Milk"]

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = Apriori.load(path)

        assert loaded.n_transactions == model.n_transactions
        assert loaded.products == model.products
        assert len(loaded.rules) == len(model.rules)
        assert len(loaded.frequent_itemsets) == len(model.frequent_itemsets)

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns rule quality metrics."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        metrics = model.evaluate(transactions)
        assert "n_rules" in metrics
        assert "n_frequent_itemsets" in metrics
        assert "coverage" in metrics
        assert "avg_confidence" in metrics
        assert "avg_lift" in metrics
        assert metrics["n_rules"] > 0
        assert 0.0 <= metrics["coverage"] <= 1.0

