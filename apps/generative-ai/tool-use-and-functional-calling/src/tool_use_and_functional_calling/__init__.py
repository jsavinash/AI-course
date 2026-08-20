"""Tool Use and Functional Calling in Generative AI.

Implementation of function calling (tool use) in LLMs covering:
- ToolSpec: Structured tool definitions
- ToolCall: LLM-reasoned tool invocations
- ToolResult: Output from tool execution
- ToolUseModel: Full 5-step workflow orchestration
- FastAPI serving with observability
"""

from tool_use_and_functional_calling.model import (
    ToolCall,
    ToolResult,
    ToolSpec,
    ToolUseModel,
)

__all__ = [
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "ToolUseModel",
]
