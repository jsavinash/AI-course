"""Data loading, tool registries, and synthetic dataset generation for Tool Use and Functional Calling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_N_TOOLS = 5
DEFAULT_N_SAMPLES = 200
DEFAULT_VOCAB_SIZE = 500

PREDEFINED_TOOLS: dict[str, dict[str, Any]] = {
    "get_stock_price": {
        "name": "get_stock_price",
        "description": "Gives the current stock price of a given company ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Stock ticker symbol, e.g. TCS, INFY, AAPL"}
            },
            "required": ["name"],
        },
        "return_type": "float",
        "keywords": ["stock", "price", "ticker", "TCS", "INFY", "AAPL", "share", "market"],
    },
    "get_weather": {
        "name": "get_weather",
        "description": "Returns the current weather for a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Mumbai, Delhi, London"}
            },
            "required": ["city"],
        },
        "return_type": "dict",
        "keywords": ["weather", "temperature", "rain", "city", "forecast", "mumbai", "delhi", "london"],
    },
    "calculator": {
        "name": "calculator",
        "description": "Performs basic arithmetic operations: add, subtract, multiply, divide",
        "parameters": {
            "type": "object",
            "properties": {
                "operand1": {"type": "number", "description": "First number"},
                "operation": {"type": "string", "description": "One of: add, subtract, multiply, divide"},
                "operand2": {"type": "number", "description": "Second number"},
            },
            "required": ["operation", "operand1", "operand2"],
        },
        "return_type": "float",
        "keywords": ["calculate", "add", "subtract", "multiply", "divide", "math", "arithmetic", "sum", "product"],
    },
    "get_order_status": {
        "name": "get_order_status",
        "description": "Retrieves the status of a customer order by order ID",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Unique order identifier, e.g. ORD-12345"}
            },
            "required": ["order_id"],
        },
        "return_type": "dict",
        "keywords": ["order", "status", "delivery", "shipped", "track", "ORD"],
    },
    "execute_sql_query": {
        "name": "execute_sql_query",
        "description": "Safely executes a read-only SQL SELECT query against the database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A read-only SQL SELECT query"}
            },
            "required": ["query"],
        },
        "return_type": "list[dict]",
        "keywords": ["sql", "query", "database", "select", "from", "where", "show", "list", "count", "find", "table", "records", "rows", "users"],
    },
}

PREDEFINED_WORKFLOWS: dict[str, dict[str, Any]] = {
    "stock-price": {
        "name": "stock-price",
        "description": "Get current stock price for a company",
        "tool_name": "get_stock_price",
        "example_query": "What is the current TCS stock price?",
        "expected_tool": "get_stock_price",
        "example_args": {"name": "TCS"},
    },
    "weather-lookup": {
        "name": "weather-lookup",
        "description": "Get current weather for a city",
        "tool_name": "get_weather",
        "example_query": "What is the weather in Mumbai?",
        "expected_tool": "get_weather",
        "example_args": {"city": "Mumbai"},
    },
    "calculator": {
        "name": "calculator",
        "description": "Perform arithmetic operation",
        "tool_name": "calculator",
        "example_query": "Calculate 25 + 37",
        "expected_tool": "calculator",
        "example_args": {"operation": "add", "operand1": 25, "operand2": 37},
    },
    "order-status": {
        "name": "order-status",
        "description": "Check order delivery status",
        "tool_name": "get_order_status",
        "example_query": "What is the status of order ORD-12345?",
        "expected_tool": "get_order_status",
        "example_args": {"order_id": "ORD-12345"},
    },
    "sql-query": {
        "name": "sql-query",
        "description": "Generate and safely execute a SQL SELECT query",
        "tool_name": "execute_sql_query",
        "example_query": "Select all users who signed up in the last 7 days",
        "expected_tool": "execute_sql_query",
        "example_args": {"query": "SELECT * FROM users WHERE signup_date >= NOW() - INTERVAL 7 DAY"},
    },
}

PREDEFINED_APPLICATIONS: list[dict[str, str]] = [
    {
        "name": "customer-support",
        "description": "Chatbot resolving customer issues via get_order_status() and check_delivery()",
    },
    {
        "name": "travel-planning",
        "description": "Chatbot searching hotels, checking vacancy, and booking via backend APIs",
    },
    {
        "name": "hr-operations",
        "description": "Chatbot answering employee queries about leave policy and working hours",
    },
    {
        "name": "automated-sql",
        "description": "LLM generating read-only SQL queries and executing them safely",
    },
]

PREDEFINED_ADVANTAGES: list[dict[str, str]] = [
    {
        "name": "real-time-data",
        "description": "Eliminates stale training data by accessing current information via tools",
    },
    {
        "name": "reduce-hallucinations",
        "description": "Grounds responses in actual tool outputs instead of model priors",
    },
    {
        "name": "extends-capability",
        "description": "Equips LLMs with external capabilities like calculator, search, and database access",
    },
]

PREDEFINED_LIMITATIONS: list[dict[str, str]] = [
    {
        "name": "token-cost",
        "description": "Many tools increase JSON schema size, raising token count and API costs",
    },
    {
        "name": "latency",
        "description": "Tool selection, execution, and final generation adds latency; unsuitable for low-latency apps",
    },
    {
        "name": "security",
        "description": "In healthcare/defense, LLM-driven tool calls can cause real-world harm via errors",
    },
]


def _tokenize(text: str, vocab_size: int = DEFAULT_VOCAB_SIZE, rng: np.random.Generator | None = None) -> np.ndarray:
    words = text.lower().split()
    tokens = [hash(w) % vocab_size for w in words]
    return np.array(tokens, dtype=int)


def generate_tool_call_dataset(
    n_samples: int = DEFAULT_N_SAMPLES,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    tool_names = list(PREDEFINED_TOOLS.keys())
    max_len = 20
    X = np.zeros((n_samples, max_len), dtype=int)
    y = np.zeros(n_samples, dtype=int)

    query_templates = {
        "get_stock_price": [
            "What is the price of {ticker} stock?",
            "Tell me the current {ticker} share price",
            "How much is {ticker} trading at?",
            "What is {ticker} stock price today?",
        ],
        "get_weather": [
            "What is the weather in {city}?",
            "Tell me the temperature in {city}",
            "How is the weather in {city} today?",
            "Is it raining in {city}?",
        ],
        "calculator": [
            "Calculate {a} plus {b}",
            "What is {a} multiplied by {b}?",
            "Compute {a} minus {b}",
            "Divide {a} by {b}",
        ],
        "get_order_status": [
            "What is the status of order {order_id}?",
            "Track my order {order_id}",
            "Has order {order_id} been shipped?",
            "Where is my order {order_id}?",
        ],
        "execute_sql_query": [
            "Show me all records from the users table",
            "List all orders placed this month",
            "Count how many customers registered last week",
            "Find all products with price greater than 100",
        ],
    }

    tickers = ["TCS", "INFY", "AAPL", "GOOGL", "MSFT"]
    cities = ["Mumbai", "Delhi", "London", "New York", "Tokyo"]

    for i in range(n_samples):
        tool_idx = rng.integers(0, len(tool_names))
        tool_name = tool_names[tool_idx]
        templates = query_templates[tool_name]
        template = templates[rng.integers(0, len(templates))]
        query = template.format(
            ticker=rng.choice(tickers),
            city=rng.choice(cities),
            a=int(rng.integers(1, 100)),
            b=int(rng.integers(1, 100)),
            order_id=f"ORD-{rng.integers(10000, 99999)}",
        )
        tokens = _tokenize(query, vocab_size, rng)
        X[i, : len(tokens)] = tokens[:max_len]
        y[i] = tool_idx

    return X, y


def load_tool_dataset(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_tool_call_dataset(n_samples=n_samples, vocab_size=vocab_size, random_seed=random_seed)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    n_test = max(1, int(n * test_size))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def save_dataset(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)


def get_tool_by_name(name: str) -> dict[str, Any] | None:
    return PREDEFINED_TOOLS.get(name)


def get_all_tools() -> list[dict[str, Any]]:
    return list(PREDEFINED_TOOLS.values())


def get_workflow_by_name(name: str) -> dict[str, Any] | None:
    return PREDEFINED_WORKFLOWS.get(name)


def get_all_workflows() -> list[dict[str, Any]]:
    return list(PREDEFINED_WORKFLOWS.values())


def get_applications() -> list[dict[str, str]]:
    return PREDEFINED_APPLICATIONS


def get_advantages() -> list[dict[str, str]]:
    return PREDEFINED_ADVANTAGES


def get_limitations() -> list[dict[str, str]]:
    return PREDEFINED_LIMITATIONS


def serialize_tools_to_json(tools: list[dict[str, Any]]) -> str:
    return json.dumps(tools, indent=2)
