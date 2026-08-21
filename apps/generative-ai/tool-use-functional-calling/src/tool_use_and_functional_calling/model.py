"""Tool Use and Functional Calling implementation.

Architecture:
    1. ToolSpec: Structured specification of an available tool (name, description, parameters, return type)
    2. ToolCall: LLM-reasoned invocation (tool name + arguments extracted by the model)
    3. ToolResult: Output of executing a tool call
    4. ToolUseModel: Orchestrates the full 5-step function-calling workflow

Core concepts:
    - Function/Tool Calling: LLM connects to external tools/APIs
    - Tool Decision: LLM reasons which tool to call and with what arguments
    - Application-side Execution: Tool is executed outside the LLM
    - Result Concatenation: Tool output combined with original query
    - Final Generation: LLM synthesizes tool output into grounded response

Workflow:
    User Query + Tool Definitions -> LLM Reasoning (tool selection + args) ->
    Application Execution -> Tool Output + Query -> Final LLM Response
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from tool_use_and_functional_calling.data import (
    PREDEFINED_TOOLS,
    get_workflow_by_name,
)


@dataclass
class ToolSpec:
    """Structured specification of an available tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    return_type: str = "any"
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_json_schema(self) -> dict[str, Any]:
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
        self._cache = {"schema": schema}
        return schema

    def get_required_params(self) -> list[str]:
        return self.parameters.get("required", [])

    def validate_args(self, args: dict[str, Any]) -> bool:
        required = self.get_required_params()
        return all(param in args for param in required)

    def matches_query(self, query: str) -> float:
        query_lower = query.lower()
        if not self.keywords:
            return 0.0
        matches = sum(1 for kw in self.keywords if kw.lower() in query_lower)
        return matches / len(self.keywords)


@dataclass
class ToolCall:
    """LLM-reasoned invocation of a tool with extracted arguments."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.call_id:
            self.call_id = f"call_{random.randint(100000, 999999)}"
        self._cache = {"call_id": self.call_id}

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }


@dataclass
class ToolResult:
    """Output produced by executing a tool call."""

    tool_name: str
    output: Any
    success: bool = True
    error: str | None = None
    call_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.error:
            self.success = False
        self._cache = {"success": self.success}

    def format_output(self) -> str:
        if not self.success:
            return f"Error calling {self.tool_name}: {self.error}"
        if isinstance(self.output, dict):
            return json.dumps(self.output)
        return str(self.output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "output": self.output,
            "success": self.success,
            "error": self.error,
        "call_id": self.call_id,
    }


@dataclass
class ToolUseModel:
    """Orchestrates the full 5-step function-calling workflow."""

    model_id: str
    base_model_name: str = "tool-use-v1"
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    _tool_call_history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _tool_result_history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _tool_execution_history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _current_query: str | None = None
    _current_tool_call: ToolCall | None = None
    _current_tool_result: ToolResult | None = None

    def _init(self) -> None:
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        for _tool_name, tool_data in PREDEFINED_TOOLS.items():
            self.register_tool(
                ToolSpec(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    parameters=tool_data["parameters"],
                    return_type=tool_data.get("return_type", "any"),
                    keywords=tool_data.get("keywords", []),
                )
            )

    def register_tool(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.to_json_schema() for tool in self.tools.values()]

    def decide_tool(self, query: str) -> ToolCall | None:
        self._current_query = query
        best_tool: ToolSpec | None = None
        best_score = 0.0

        for tool in self.tools.values():
            score = tool.matches_query(query)
            if score > best_score:
                best_score = score
                best_tool = tool

        if best_tool is None or best_score < 0.1:
            return None

        args = self._extract_arguments(query, best_tool)
        tool_call = ToolCall(tool_name=best_tool.name, arguments=args)
        self._current_tool_call = tool_call
        self._tool_call_history.append(tool_call.to_dict())
        return tool_call

    def _extract_arguments(self, query: str, tool: ToolSpec) -> dict[str, Any]:
        args: dict[str, Any] = {}
        assigned_params: set[str] = set()
        props = tool.parameters.get("properties", {})
        param_order = list(props.keys())
        param_idx = 0

        raw_tokens = query.replace("?", "").replace(",", "").replace("'", "").split()
        tokens = [t.strip().rstrip(".") for t in raw_tokens if t.strip()]

        for token in tokens:
            while param_idx < len(param_order) and param_order[param_idx] in assigned_params:
                param_idx += 1
            if param_idx >= len(param_order):
                break
            param = param_order[param_idx]
            param_info = props[param]
            param_type = param_info.get("type", "string")
            clean_lower = token.lower()
            matched = False

            if param_type == "string":
                if param == "name" and clean_lower.upper() in {"TCS", "INFY", "AAPL", "GOOGL", "MSFT"}:
                    args[param] = clean_lower.upper()
                    matched = True
                elif param == "city" and clean_lower.title() in {"Mumbai", "Delhi", "London", "New York", "Tokyo"}:
                    args[param] = clean_lower.title()
                    matched = True
                elif param == "operation" and clean_lower in {
                    "add", "subtract", "multiply", "divide", "plus", "minus", "times", "by", "+", "-", "*", "/"
                }:
                    op_map = {"plus": "add", "minus": "subtract", "times": "multiply", "by": "divide", "+": "add", "-": "subtract", "*": "multiply", "/": "divide", "add": "add", "subtract": "subtract", "multiply": "multiply", "divide": "divide"}
                    args[param] = op_map.get(clean_lower, clean_lower)
                    matched = True
                elif param == "order_id" and (
                    clean_lower.upper().startswith("ORD-") or clean_lower.upper().startswith("ORD_")
                ):
                    args[param] = clean_lower.upper()
                    matched = True
                elif param == "query" and any(
                    kw in clean_lower for kw in ["select", "from", "where", "show", "list", "count", "find", "users"]
                ):
                    args[param] = query.strip()
                    matched = True
            elif param_type in ("number", "integer"):
                try:
                    args[param] = float(clean_lower)
                    matched = True
                except ValueError:
                    pass

            if matched:
                assigned_params.add(param)
                param_idx += 1

        for required_param in tool.get_required_params():
            if required_param not in args:
                if required_param == "name":
                    args[required_param] = random.choice(["TCS", "INFY", "AAPL", "GOOGL", "MSFT"])
                elif required_param == "city":
                    args[required_param] = random.choice(["Mumbai", "Delhi", "London", "New York", "Tokyo"])
                elif required_param == "operation":
                    args[required_param] = random.choice(["add", "subtract", "multiply", "divide"])
                elif required_param == "order_id":
                    args[required_param] = f"ORD-{random.randint(10000, 99999)}"
                elif required_param == "query":
                    args[required_param] = "SELECT * FROM users LIMIT 10"

        return args

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        tool = self.tools.get(tool_call.tool_name)
        if tool is None:
            result = ToolResult(
                tool_name=tool_call.tool_name,
                output=None,
                success=False,
                error=f"Unknown tool: {tool_call.tool_name}",
                call_id=tool_call.call_id,
            )
            self._current_tool_result = result
            self._tool_result_history.append(result.to_dict())
            return result

        try:
            raw_output = _run_tool_function(tool_call.tool_name, tool_call.arguments)
            result = ToolResult(
                tool_name=tool_call.tool_name,
                output=raw_output,
                success=True,
                call_id=tool_call.call_id,
            )
        except Exception as exc:
            result = ToolResult(
                tool_name=tool_call.tool_name,
                output=None,
                success=False,
                error=str(exc),
                call_id=tool_call.call_id,
            )

        self._current_tool_result = result
        self._tool_result_history.append(result.to_dict())
        self._tool_execution_history.append({
            "call": tool_call.to_dict(),
            "result": result.to_dict(),
        })
        return result

    def run_workflow(self, workflow_name: str) -> dict[str, Any] | None:
        workflow = get_workflow_by_name(workflow_name)
        if workflow is None:
            return None
        return self.invoke(workflow["example_query"])

    def invoke(self, query: str) -> dict[str, Any]:
        self._current_query = query
        tool_call = self.decide_tool(query)
        if tool_call is None:
            return {
                "query": query,
                "tool_call": None,
                "tool_result": None,
                "final_response": f"I don't have a relevant tool to answer: '{query}'. Please provide more context.",
                "success": False,
            }

        tool_result = self.execute_tool(tool_call)
        formatted_output = tool_result.format_output()
        final_response = _synthesize_response(query, tool_call.tool_name, formatted_output, tool_result.success)

        return {
            "query": query,
            "tool_call": tool_call.to_dict(),
            "tool_result": tool_result.to_dict(),
            "final_response": final_response,
            "success": tool_result.success,
        }

    def get_tool_call_history(self) -> list[dict[str, Any]]:
        return list(self._tool_call_history)

    def get_tool_result_history(self) -> list[dict[str, Any]]:
        return list(self._tool_result_history)

    def get_execution_history(self) -> list[dict[str, Any]]:
        return list(self._tool_execution_history)

    def save(self, path: str) -> None:
        data = {
            "model_id": self.model_id,
            "base_model_name": self.base_model_name,
            "tools": {name: tool.to_json_schema() for name, tool in self.tools.items()},
            "tool_call_history": self._tool_call_history,
            "tool_result_history": self._tool_result_history,
            "tool_execution_history": self._tool_execution_history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> ToolUseModel:
        with open(path) as f:
            data = json.load(f)

        model = cls(model_id=data["model_id"], base_model_name=data.get("base_model_name", "tool-use-v1"))
        model._tool_call_history = data.get("tool_call_history", [])
        model._tool_result_history = data.get("tool_result_history", [])
        model._tool_execution_history = data.get("tool_execution_history", [])
        model._init()
        return model

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_model_name": self.base_model_name,
            "n_tools": len(self.tools),
            "n_tool_calls": len(self._tool_call_history),
            "n_tool_results": len(self._tool_result_history),
            "n_executions": len(self._tool_execution_history),
        }


def _run_tool_function(tool_name: str, args: dict[str, Any]) -> Any:
    if tool_name == "get_stock_price":
        ticker = args.get("name", "UNKNOWN").upper()
        stock_prices = {"TCS": 3718.0, "INFY": 4210.0, "AAPL": 213.0, "GOOGL": 175.0, "MSFT": 420.0}
        return {"ticker": ticker, "price": stock_prices.get(ticker, 1000.0), "currency": "INR" if ticker in {"TCS", "INFY"} else "USD"}

    elif tool_name == "get_weather":
        city = args.get("city", "Unknown")
        conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
        temps = {"Mumbai": 32, "Delhi": 38, "London": 18, "New York": 25, "Tokyo": 28}
        return {
            "city": city,
            "temperature_c": temps.get(city, random.randint(15, 40)),
            "condition": random.choice(conditions),
            "humidity_pct": random.randint(30, 90),
        }

    elif tool_name == "calculator":
        op = args.get("operation", "add")
        a = float(args.get("operand1", 0))
        b = float(args.get("operand2", 0))
        ops = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else float("inf"),
        }
        return {"operation": op, "operand1": a, "operand2": b, "result": ops.get(op, 0)}

    elif tool_name == "get_order_status":
        order_id = args.get("order_id", "UNKNOWN")
        statuses = ["Processing", "Shipped", "Delivered", "Cancelled"]
        return {
            "order_id": order_id,
            "status": random.choice(statuses),
            "estimated_delivery": "2026-08-25",
            "tracking_number": f"TRK{random.randint(100000, 999999)}",
        }

    elif tool_name == "execute_sql_query":
        query = args.get("query", "SELECT 1")
        return {
            "query": query,
            "rows": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
            ],
            "row_count": 2,
            "note": "Read-only query executed safely",
        }

    raise ValueError(f"Unknown tool: {tool_name}")


def _synthesize_response(query: str, tool_name: str, tool_output: str, success: bool) -> str:
    if not success:
        return f"I attempted to use the {tool_name} tool but encountered an error: {tool_output}"

    responses = {
        "get_stock_price": f"Based on the tool output, {tool_output}.",
        "get_weather": f"Based on the current weather data: {tool_output}.",
        "calculator": f"Based on the calculation: {tool_output}.",
        "get_order_status": f"Here is your order information: {tool_output}.",
        "execute_sql_query": f"Here are the query results: {tool_output}.",
    }
    return responses.get(tool_name, f"Tool '{tool_name}' returned: {tool_output}.")
