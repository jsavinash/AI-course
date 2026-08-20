# Tool Use and Functional Calling in Generative AI

## What is Function Calling (Tool Calling)?

Function calling (also known as tool-calling) is a method by which LLMs can reliably connect and interact with external tools or APIs. The application provides the LLM with a set of tools, and the model intelligently decides which tool to invoke for a specific user query to complete a given task. Function calling gives the LLM the power to interact with external information sources like APIs, databases, or knowledge bases.

## Architecture and Core Mechanism

The function-calling workflow consists of 5 steps:

### Step 1: Initial Request with Tool Definitions
The application provides the available tools to the LLM in JSON format with fields: `name`, `description`, and the number and types of arguments needed.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }
        }
    }
]
```

### Step 2: Function/Tool Decision
The model reasons whether the query warrants the use of any available tool. If it does, it returns a JSON specifying the name of the function and the parameters to pass.

### Step 3: Application-side Execution
The LLM waits while the application processes the request and executes the function.

### Step 4: Result Concatenation
The output of the function is formatted and passed back to the LLM along with the original user query.

### Step 5: Final Generation
The LLM reasons about the function's outputs and the query, returning a grounded and well-informed response.

## Implementation Workflow (LangChain-style)

```
User Query + Tool Definitions
         |
         v
  LLM Reasoning (tool selection + arg extraction)
         |
         v
  Application-side Function Execution
         |
         v
  Tool Output + Original Query
         |
         v
  Final Informed LLM Response
```

## Tool Definition

Tools are defined as structured specifications containing:
- **name**: Unique identifier for the tool
- **description**: Natural language description of what the tool does
- **parameters**: JSON schema describing required/optional arguments
- **return_type**: Expected type of the function's return value

```python
tool_spec = ToolSpec(
    name="get_stock_price",
    description="Gives stock prices of some stocks",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Stock ticker symbol"}
        },
        "required": ["name"]
    }
)
```

## Tool Decision (LLM Reasoning)

The LLM analyzes the user query and decides:
1. Whether a tool needs to be called at all
2. Which specific tool to invoke
3. What arguments to pass to that tool

```python
tool_call = ToolCall(
    name="get_stock_price",
    arguments={"name": "TCS"}
)
```

## Tool Execution

The application executes the tool with the LLM-reasoned arguments. This step happens outside the LLM and may involve:
- Calling external APIs
- Querying databases
- Running Python functions
- Accessing knowledge bases

## Result Concatenation

The tool output is formatted and combined with the original query context before being fed back to the LLM.

```python
tool_output = ToolResult(
    tool_name="get_stock_price",
    output=3718.0,
    success=True
)
```

## Final Generation

The LLM synthesizes the tool output with the original query to produce a grounded response.

## Applications of Function Calling

### 1. Customer Support and Conversational AI
Customer-facing chatbots with function calling can resolve issues by accessing backend APIs like `get_order_status()`, `check_delivery()`, enabling accurate responses.

### 2. Travel Planning
A chatbot equipped with tools can search for hotels near a location, check vacancy status, and interact with backend APIs to book the hotel.

### 3. HR and Business Operations
A chatbot can resolve employee queries like "How many leaves can I take" or "What are the working hours" by interacting with backend APIs.

### 4. Automated SQL Queries
An LLM can generate SQL queries for a user query and interact with functions like `check_query()` (ensuring read-only) and `execute_query()` to safely query databases.

## Advantages

### Real-Time Data Access
Function calling eliminates the fundamental problem of LLMs giving responses based on stale training data. Tools provide access to current information.

### Reducing Hallucinations
Models tend to not admit when they don't know something, causing failures in downstream tasks. Function calling tackles this by grounding responses in actual tool outputs.

### Extends LLM Capabilities
LLMs are notoriously bad at arithmetic. By equipping them with a calculator tool, the LLM can call the calculator with operands and operators, increasing factual correctness.

## Limitations

### Token Consumption and Cost
When the number of functions and tools is high, the JSON schema grows large, increasing token count and cost for the user.

### Cost and Latency
Deciding which tool to call, executing it, and then performing final generation adds latency, making it unsuitable for applications requiring low latency.

### Security and Safety Risks
In sectors like healthcare and defense, using an LLM to call functions can have real-world consequences. Errors can cause financial loss, data corruption, or security breaches.

## Network Type
Generative AI with Tool Use and Function Calling

## Training

```bash
tool-use-train --model-dir ./artifacts/models --n-tools 5 --technique tool-selection
```

## Serving API

```bash
uvicorn tool_use_and_functional_calling.api:app --host 0.0.0.0 --port 8015
```

### Endpoints
- `GET /` - Service info with registered tools
- `GET /health` - Health check
- `POST /tools/register` - Register a new tool
- `POST /invoke` - Invoke tool use pipeline end-to-end
- `POST /tools/execute` - Execute a specific tool
- `GET /tools` - List all registered tools
- `GET /workflows` - List predefined workflows
- `GET /stats` - Model statistics
- `GET /metrics` - Prometheus metrics

### Supported Workflows
- `stock-price` - Get stock price via tool call
- `weather-lookup` - Weather information via tool
- `calculator` - Arithmetic via calculator tool
- `order-status` - Order status via backend API
- `sql-query` - Safe SQL query generation and execution

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
