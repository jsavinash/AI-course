# Recommendation Engine (Unsupervised Learning)

Production-ready Apriori algorithm for association rule mining and product recommendations.

## Learning Type: Unsupervised (Association Rule Mining)

This example demonstrates **unsupervised learning** — the model is trained on transaction data without any labels and discovers association rules (e.g., "customers who bought X also bought Y").

## Problem

Generate product recommendations for customers based on their current purchases using association rule mining.

## Approach

- **Algorithm**: Apriori algorithm for frequent itemset mining
- **Metrics**: Support, confidence, lift, and conviction for association rules
- **Implementation**: Pure NumPy with Python sets, no external ML libraries
- **Recommendation**: Find rules that match customer's current items and suggest consequents

## Architecture

```
apps/machine-learning/unsupervised/recommendation-engine/
├── pyproject.toml              # Package configuration
├── src/recommendation_engine/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Transaction data generation
│   ├── model.py                # Apriori (from scratch)
│   ├── train.py                # Training pipeline script
│   └── api.py                  # FastAPI serving API
└── README.md                   # This file
```

## Quick Start

### Training

```bash
make train-recommendation-engine
```

Or directly:

```bash
uv run python -m recommendation_engine.train --model-dir ./artifacts/models --model-version 1.0.0
```

### Serving

```bash
make serve-recommendation-engine
```

Or directly:

```bash
MODEL_DIR=./artifacts/models uv run uvicorn recommendation_engine.api:app --host 0.0.0.0 --port 8003
```

### API Usage

```bash
# Health check
curl http://localhost:8003/health

# Get recommendations
curl -X POST http://localhost:8003/recommend \
  -H "Content-Type: application/json" \
  -d '{"items": ["Bread", "Milk"], "top_k": 5}'

# Get association rules
curl http://localhost:8003/rules

# Get model statistics
curl http://localhost:8003/stats
```

## API Endpoints

| Method | Path         | Description                     |
|--------|--------------|---------------------------------|
| GET    | `/health`    | Health check                    |
| POST   | `/recommend` | Get product recommendations     |
| GET    | `/rules`     | Get all association rules       |
| GET    | `/stats`     | Model statistics                |
| GET    | `/metrics`   | Prometheus metrics              |
| POST   | `/reload`    | Reload model                    |

## Model Parameters

| Parameter          | Default | Description                              |
|--------------------|---------|------------------------------------------|
| `min_support`      | `0.2`   | Minimum support threshold (0-1)          |
| `min_confidence`   | `0.5`   | Minimum confidence threshold (0-1)       |
| `min_lift`         | `1.0`   | Minimum lift threshold (>= 1.0)          |
| `max_itemset_size` | `4`     | Maximum size of itemsets to consider     |

## Association Rule Metrics

- **Support**: Frequency of itemset occurrence
- **Confidence**: P(consequent | antecedent)
- **Lift**: How much more likely consequent is given antecedent
- **Conviction**: Measures implication strength

## Key Concepts Demonstrated

- Apriori principle for frequent itemset mining
- Association rule generation with multiple metrics
- Transaction data processing
- Production API serving with FastAPI
- Model versioning with registry
