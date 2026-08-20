# Code Generation

Generative AI code generation uses large language models to write, complete, and debug computer software automatically from natural language prompts or existing code context. It helps developers build features faster, handle repetitive boilerplate tasks, and modernize legacy systems.

## Network Type
Generative AI Code Generation with Transformer-based Language Model

## Architecture

### Key Components

1. **CodeTokenizer**: Tokenizes code and natural language into token IDs, handling special tokens like `<PAD>`, `<UNK>`, `<EOS>`, `<NL>`, `<INDENT>`, `<DEDENT>`, `<COMMENT>`.
2. **BaseCodeModel**: Transformer-based language model that learns code patterns and syntax.
3. **CodeCompletionModel**: Predicts and auto-completes lines or full functions given a code prefix.
4. **TextToCodeModel**: Translates plain English descriptions into functional code blocks or scripts.
5. **RefactoringModel**: Upgrades older software frameworks, improves readability, and translates code between languages.
6. **TestingAndDebuggingModel**: Scans for bugs, identifies security vulnerabilities, and auto-generates unit tests.

## Core Capabilities

### Code Completion
Predicts and auto-completes lines or full functions in real time as you type. The `CodeCompletionModel` takes a code prefix and generates the most likely continuation based on learned patterns.

### Text-to-Code
Translates plain English descriptions into functional code blocks or scripts across multiple languages. The `TextToCodeModel` takes natural language input and generates syntactically correct code.

### Refactoring & Modernization
Upgrades older software frameworks, improves readability, and translates code from one language to another. The `RefactoringModel` takes legacy code and generates modernized versions.

### Testing & Debugging
Scans for bugs, identifies security vulnerabilities, and auto-generates unit tests. The `TestingAndDebuggingModel` provides:
- Bug scanning with probability estimation
- Confidence scoring
- Suggested fixes
- Automatic unit test generation

## How It Works

Generative AI code generation works by:

1. **Training on Code Corpora**: The model is trained on large datasets of code across multiple languages and frameworks.
2. **Learning Patterns**: The transformer architecture learns syntax, semantics, and common patterns.
3. **Context Understanding**: Given a prompt or partial code, the model understands the context and generates appropriate continuations.
4. **Iterative Refinement**: Generated code can be refined through additional prompting or feedback.

## Applications

- **Code Completion**: IDE integrations that suggest code as developers type
- **Text-to-Code**: Converting requirements or pseudocode into working implementations
- **Refactoring**: Modernizing legacy codebases to newer frameworks or languages
- **Testing**: Automatically generating unit tests and test suites
- **Debugging**: Identifying bugs and suggesting fixes
- **Documentation**: Generating code comments and documentation

## Advantages

- **Speed**: Build features faster by automating repetitive boilerplate tasks
- **Productivity**: Reduce manual coding effort for common patterns
- **Modernization**: Easily upgrade legacy systems to modern frameworks
- **Accessibility**: Enable non-experts to generate code through natural language
- **Consistency**: Maintain consistent code style and patterns across projects

## Limitations

- **Accuracy**: Generated code may contain subtle bugs or security vulnerabilities
- **Context Limits**: Models have limited context windows for large codebases
- **Domain Specificity**: May require fine-tuning for specialized domains
- **Dependency**: Generated code may rely on libraries or patterns not present in the codebase

## Training

```bash
code_generation-train --model-dir ./artifacts/models --n-samples 500 --n-iterations 100 --d-model 256 --n-layers 2
```

## Serving API

```bash
uvicorn code_generation.api:app --host 0.0.0.0 --port 8015
```

### Endpoints
- `GET /` - Service info with capabilities
- `GET /health` - Health check
- `POST /complete` - Code completion given a prefix
- `POST /text-to-code` - Generate code from natural language description
- `POST /refactor` - Refactor/modernize existing code
- `POST /scan-bugs` - Scan code for bugs and vulnerabilities
- `POST /generate-tests` - Auto-generate unit tests for code
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Input Format

### Code Completion
- `code_prefix`: string (max 500 chars)
- `max_new_tokens`: int (1-200)

### Text-to-Code
- `description`: string (max 500 chars)
- `max_new_tokens`: int (1-200)

### Refactoring
- `old_code`: string (max 500 chars)
- `target_language`: string (default: "modern_python")

### Bug Scanning
- `code`: string (max 500 chars)

### Unit Test Generation
- `code`: string (max 500 chars)
- `max_new_tokens`: int (1-200)

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
