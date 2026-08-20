# Text Generation

Text generation is a core generative AI task that produces new text sequences from a given prompt or seed context. It uses transformer-based autoregressive models to predict the next token in a sequence, enabling applications like story generation, code generation, dialogue systems, and content creation.

## Network Type
Transformer-based Autoregressive Generation

## Architecture

### Key Components

1. **TextTokenizer**: Tokenizes text into token IDs with a vocabulary, supporting special tokens like `<PAD>`, `<UNK>`, `<EOS>`, and `<BOS>`.
2. **BaseTextModel**: Transformer-based backbone with multi-head self-attention and feed-forward layers.
3. **SamplingStrategy**: Controls generation diversity via temperature, top-k, and top-p (nucleus) sampling.
4. **TextGenerationModel**: Main model that orchestrates tokenization, forward passes, and sampling.

### Core Technologies

- **Transformer Blocks**: Multi-head attention captures long-range dependencies in text.
- **Positional Encoding**: Injects sequence position information since transformers have no inherent order.
- **Autoregressive Generation**: Generates one token at a time, conditioning on all previously generated tokens.
- **Sampling Strategies**:
  - **Temperature**: Controls randomness (lower = more deterministic, higher = more creative).
  - **Top-K**: Restricts sampling to the K most likely next tokens.
  - **Top-P (Nucleus)**: Dynamically selects from the smallest set of tokens whose cumulative probability exceeds P.

## How Text Generation Works

1. **Tokenize Prompt**: Convert input text into token IDs using the vocabulary.
2. **Forward Pass**: Feed token IDs through the transformer model to get logits for the next token.
3. **Apply Sampling**: Use temperature, top-k, and top-p to sample the next token from the distribution.
4. **Repeat**: Append the sampled token and repeat until `max_new_tokens` or `<EOS>` is reached.

## Applications

- **Creative Writing**: Generate stories, poems, and articles from brief prompts.
- **Dialogue Systems**: Power chatbots and conversational agents.
- **Code Generation**: Translate natural language descriptions into functional code.
- **Content Summarization**: Generate concise summaries of long documents.
- **Data-to-Text**: Convert structured data into natural language narratives.

## Training

```bash
text_generation-train --model-dir ./artifacts/models --n-samples 500 --temperature 0.8 --top-k 50 --top-p 0.9
```

## Serving API

```bash
uvicorn text_generation.api:app --host 0.0.0.0 --port 8015
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /generate` - Generate text from a prompt
- `POST /evaluate` - Evaluate generated text against reference
- `GET /stats` - Model statistics
- `GET /metrics` - Prometheus metrics

### Generation Parameters
- `prompt`: Input text seed for generation
- `max_new_tokens`: Maximum number of tokens to generate
- `temperature`: Sampling temperature (0.1 - 2.0)
- `top_k`: Top-k sampling parameter
- `top_p`: Nucleus sampling threshold

## Dependencies
- Python >= 3.11
- NumPy, FastAPI, Pydantic
- mlops-shared
