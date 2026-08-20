# Attention Mechanism

The attention mechanism allows models to focus on the most important parts of input data by assigning different weights to different elements. Forms the core of models like Transformers and BERT.

## Types Covered
- **Soft Attention**: Differentiable using softmax (standard for NLP/transformers)
- **Hard Attention**: Non-differentiable, selects specific input positions via sampling
- **Self-Attention**: Each element attends to other aspects in the same sequence
- **Multi-Head Attention**: Multiple parallel attention heads from different subspaces
- **Additive Attention**: Uses feed-forward NN to calculate attention scores

## Architecture
1. **Input Encoding**: RNN/LSTM/GRU/Transformer hidden states
2. **Query, Key, Value**: Linear transformations of input embeddings
3. **Similarity Computation**:
   - Dot Product: `Score(s,i) = h_s · y_i`
   - General: `Score(s,i) = h_s^T W y_i`
   - Concat: `Score(s,i) = v^T tanh(W[h_s; y_i])`
4. **Softmax**: `alpha(s,i) = softmax(Score(s,i))`
5. **Weighted Sum**: `c_t = Σ α(s,i) · V_i`
6. **Context Vector**: summarizes relevant input information
7. **Integration**: decoder uses context vector + hidden state

## Training
```bash
attention_mechanism-train --model-dir ./artifacts/models --n-iterations 200 --attention-type multi_head
```

## Serving API
```bash
uvicorn attention_mechanism.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Sequence prediction with attention weights
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Input Format
- `input_sequence`: 2D array (seq_len, input_dim) of embedded sequence tokens

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
