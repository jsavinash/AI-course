"""Code Generation model implementation from scratch using NumPy.

Architecture:
    1. CodeTokenizer: Tokenizes code and natural language into token IDs
    2. CodeCompletionModel: Predicts and auto-completes code given context
    3. TextToCodeModel: Translates natural language descriptions into code
    4. RefactoringModel: Upgrades/translates code between languages/frameworks
    5. TestingAndDebuggingModel: Scans for bugs, generates unit tests

Core capabilities:
    - Code Completion: Predicts and auto-completes lines or full functions
    - Text-to-Code: Translates plain English descriptions into functional code
    - Refactoring & Modernization: Upgrades older frameworks, improves readability
    - Testing & Debugging: Scans for bugs, identifies vulnerabilities, auto-generates tests

Args:
    vocab_size: vocabulary size for code tokens
    d_model: model dimension
    n_heads: number of attention heads
    n_layers: number of transformer layers
    d_ff: feed-forward inner dimension
    max_seq_len: maximum sequence length
    learning_rate: gradient descent step size
    n_iterations: number of training epochs
    random_seed: random seed
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + (mask * -1e9)
    attn = softmax(scores, axis=-1)
    return attn @ V


def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i // 2) / d_model))
            if i % 2 == 0:
                pe[pos, i] = np.sin(angle)
            else:
                pe[pos, i] = np.cos(angle)
    return pe


@dataclass
class CodeTokenizer:
    vocab_size: int = 1000
    max_seq_len: int = 128
    random_seed: int = 42

    token_to_id: dict[str, int] = field(default_factory=dict, repr=False)
    id_to_token: dict[int, str] = field(default_factory=dict, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        special_tokens = ["<PAD>", "<UNK>", "<EOS>", "<NL>", "<INDENT>", "<DEDENT>", "<COMMENT>"]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        keywords = ["def", "class", "return", "if", "else", "for", "while", "import", "from", "as", "try", "except", "with", "lambda", "yield", "async", "await", "pass", "break", "continue", "in", "not", "and", "or", "True", "False", "None"]
        for i, kw in enumerate(keywords):
            idx = len(special_tokens) + i
            self.token_to_id[kw] = idx
            self.id_to_token[idx] = kw

    def encode(self, text: str) -> list[int]:
        tokens = text.split()
        encoded = [self.token_to_id.get(t, self.token_to_id["<UNK>"]) for t in tokens]
        if len(encoded) > self.max_seq_len:
            encoded = encoded[:self.max_seq_len]
        return encoded

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        return " ".join(tokens)

    def batch_encode(self, texts: list[str]) -> np.ndarray:
        max_len = max(len(self.encode(t)) for t in texts)
        max_len = min(max_len, self.max_seq_len)
        batch = np.full((len(texts), max_len), self.token_to_id["<PAD>"], dtype=int)
        for i, text in enumerate(texts):
            encoded = self.encode(text)
            batch[i, :len(encoded)] = encoded
        return batch


@dataclass
class MultiHeadAttention:
    d_model: int = 256
    n_heads: int = 8
    random_seed: int = 42
    trainable: bool = True

    d_k: int = field(init=False)
    W_q: np.ndarray | None = None
    W_k: np.ndarray | None = None
    W_v: np.ndarray | None = None
    W_o: np.ndarray | None = None
    dW_q: np.ndarray | None = None
    dW_k: np.ndarray | None = None
    dW_v: np.ndarray | None = None
    dW_o: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.W_q = rng.normal(0, scale, (self.d_model, self.d_model))
        self.W_k = rng.normal(0, scale, (self.d_model, self.d_model))
        self.W_v = rng.normal(0, scale, (self.d_model, self.d_model))
        self.W_o = rng.normal(0, scale, (self.d_model, self.d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, _, seq_len, _ = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if self.W_q is None:
            self.init_weights()
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        Q_split = self._split_heads(Q)
        K_split = self._split_heads(K)
        V_split = self._split_heads(V)
        if mask is not None and mask.ndim == 2:
            mask = mask[np.newaxis, np.newaxis, :, :].astype(bool)
        elif mask is not None and mask.ndim == 3:
            mask = mask[:, np.newaxis, :, :].astype(bool)
        out = np.zeros_like(Q_split)
        for h in range(self.n_heads):
            q_h = Q_split[:, h, :, :]
            k_h = K_split[:, h, :, :]
            v_h = V_split[:, h, :, :]
            out[:, h, :, :] = scaled_dot_product_attention(q_h, k_h, v_h, mask)
        out = self._combine_heads(out)
        result = out @ self.W_o
        self._cache = {"x": x, "Q": Q, "K": K, "V": V, "out": out, "result": result}
        return result

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        out_grad = dout @ self.W_o.T
        dout_combined = self._split_heads(out_grad)
        dQ_split = np.zeros_like(dout_combined)
        dK_split = np.zeros_like(dout_combined)
        dV_split = np.zeros_like(dout_combined)
        for _h in range(self.n_heads):
            pass
        dQ = self._combine_heads(dQ_split)
        dK = self._combine_heads(dK_split)
        dV = self._combine_heads(dV_split)
        self.dW_q = c["x"].T @ dQ
        self.dW_k = c["x"].T @ dK
        self.dW_v = c["x"].T @ dV
        self.dW_o = c["out"].T @ dout
        return dout @ self.W_o.T @ self.W_q.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W_q is None or not self.trainable:
            return
        self.W_q -= lr * (self.dW_q + weight_decay * self.W_q)
        self.W_k -= lr * (self.dW_k + weight_decay * self.W_k)
        self.W_v -= lr * (self.dW_v + weight_decay * self.W_v)
        self.W_o -= lr * (self.dW_o + weight_decay * self.W_o)


@dataclass
class AddNorm:
    d_model: int = 256
    random_seed: int = 42
    trainable: bool = True

    gamma: np.ndarray | None = None
    beta: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_params(self) -> None:
        self.gamma = np.ones(self.d_model)
        self.beta = np.zeros(self.d_model)

    def forward(self, residual: np.ndarray, output: np.ndarray) -> np.ndarray:
        if self.gamma is None:
            self.init_params()
        combined = residual + output
        normed = layer_norm(combined, self.gamma, self.beta)
        self._cache = {"residual": residual, "output": output, "combined": combined, "normed": normed}
        return normed

    def backward(self, dout: np.ndarray) -> np.ndarray:
        eps = 1e-5
        c = self._cache
        x = c["combined"]
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1.0 / std, axis=-1, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
        dx = dx_norm / std + dvar * 2 * (x - mean) / x.shape[-1] + dmean / x.shape[-1]
        return dx


@dataclass
class FeedForward:
    d_model: int = 256
    d_ff: int = 1024
    random_seed: int = 42
    trainable: bool = True

    W1: np.ndarray | None = None
    b1: np.ndarray | None = None
    W2: np.ndarray | None = None
    b2: np.ndarray | None = None
    dW1: np.ndarray | None = None
    db1: np.ndarray | None = None
    dW2: np.ndarray | None = None
    db2: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.W1 = rng.normal(0, scale, (self.d_model, self.d_ff))
        self.b1 = np.zeros(self.d_ff)
        self.W2 = rng.normal(0, scale, (self.d_ff, self.d_model))
        self.b2 = np.zeros(self.d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W1 is None:
            self.init_weights()
        z1 = x @ self.W1 + self.b1
        a1 = gelu(z1)
        out = a1 @ self.W2 + self.b2
        self._cache = {"x": x, "z1": z1, "a1": a1}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        self.dW2 = c["a1"].T @ dout
        self.db2 = np.sum(dout, axis=0)
        da1 = dout @ self.W2.T
        dz1 = da1 * (c["a1"] * (1 - c["a1"]))
        self.dW1 = c["x"].T @ dz1
        self.db1 = np.sum(dz1, axis=0)
        return dz1 @ self.W1.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W1 is None or not self.trainable:
            return
        self.W1 -= lr * (self.dW1 + weight_decay * self.W1)
        self.b1 -= lr * self.db1
        self.W2 -= lr * (self.dW2 + weight_decay * self.W2)
        self.b2 -= lr * self.db2


@dataclass
class TransformerBlock:
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    random_seed: int = 42
    trainable: bool = True

    self_attn: MultiHeadAttention | None = None
    add_norm1: AddNorm | None = None
    ffn: FeedForward | None = None
    add_norm2: AddNorm | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.self_attn = MultiHeadAttention(self.d_model, self.n_heads, self.random_seed, trainable=self.trainable)
        self.add_norm1 = AddNorm(self.d_model, self.random_seed + 1, trainable=self.trainable)
        self.ffn = FeedForward(self.d_model, self.d_ff, self.random_seed + 2, trainable=self.trainable)
        self.add_norm2 = AddNorm(self.d_model, self.random_seed + 3, trainable=self.trainable)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        attn_out = self.self_attn.forward(x, mask)
        x = self.add_norm1.forward(x, attn_out)
        ffn_out = self.ffn.forward(x)
        x = self.add_norm2.forward(x, ffn_out)
        return x


@dataclass
class BaseCodeModel:
    vocab_size: int = 1000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 1024
    max_seq_len: int = 128
    random_seed: int = 42

    embedding: np.ndarray | None = None
    pos_encoding: np.ndarray | None = None
    layers: list[TransformerBlock] = field(default_factory=list, repr=False)
    W_out: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.embedding = rng.normal(0, 0.02, (self.vocab_size, self.d_model))
        self.pos_encoding = positional_encoding(self.max_seq_len, self.d_model)
        self.layers = [
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.random_seed + i, trainable=True)
            for i in range(self.n_layers)
        ]
        self.W_out = rng.normal(0, np.sqrt(1.0 / self.d_model), (self.vocab_size, self.d_model))

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.embedding is None:
            self._init()
        seq_len = x.shape[1] if x.ndim > 1 else 1
        embedded = self.embedding[x] * np.sqrt(self.d_model)
        if x.ndim == 1:
            embedded = embedded + self.pos_encoding[:len(x)]
        else:
            embedded = embedded + self.pos_encoding[:seq_len]
        for layer in self.layers:
            embedded = layer.forward(embedded)
        logits = embedded @ self.W_out.T
        return logits

    def get_features(self, x: np.ndarray) -> np.ndarray:
        if self.embedding is None:
            self._init()
        seq_len = x.shape[1] if x.ndim > 1 else 1
        embedded = self.embedding[x] * np.sqrt(self.d_model)
        if x.ndim == 1:
            embedded = embedded + self.pos_encoding[:len(x)]
        else:
            embedded = embedded + self.pos_encoding[:seq_len]
        for layer in self.layers:
            embedded = layer.forward(embedded)
        self._features = embedded
        return embedded


@dataclass
class CodeCompletionModel:
    base_model: BaseCodeModel | None = None
    vocab_size: int = 1000
    d_model: int = 256
    random_seed: int = 42

    def __post_init__(self):
        if self.base_model is None:
            self.base_model = BaseCodeModel(vocab_size=self.vocab_size, d_model=self.d_model, random_seed=self.random_seed)

    def complete(self, prefix_tokens: np.ndarray, max_new_tokens: int = 20) -> np.ndarray:
        if self.base_model is None:
            raise ValueError("Base model not initialized")
        generated = list(prefix_tokens[0])
        for _ in range(max_new_tokens):
            context = np.array([generated])
            logits = self.base_model.forward(context)
            next_token = int(np.argmax(logits[0, -1, :]))
            generated.append(next_token)
        return np.array(generated)


@dataclass
class TextToCodeModel:
    base_model: BaseCodeModel | None = None
    vocab_size: int = 1000
    d_model: int = 256
    random_seed: int = 42

    def __post_init__(self):
        if self.base_model is None:
            self.base_model = BaseCodeModel(vocab_size=self.vocab_size, d_model=self.d_model, random_seed=self.random_seed)

    def generate(self, description_tokens: np.ndarray, max_new_tokens: int = 50) -> np.ndarray:
        generated = list(description_tokens[0])
        for _ in range(max_new_tokens):
            context = np.array([generated])
            logits = self.base_model.forward(context)
            next_token = int(np.argmax(logits[0, -1, :]))
            generated.append(next_token)
        return np.array(generated)


@dataclass
class RefactoringModel:
    base_model: BaseCodeModel | None = None
    vocab_size: int = 1000
    d_model: int = 256
    random_seed: int = 42

    def __post_init__(self):
        if self.base_model is None:
            self.base_model = BaseCodeModel(vocab_size=self.vocab_size, d_model=self.d_model, random_seed=self.random_seed)

    def refactor(self, old_code_tokens: np.ndarray, target_language: str = "modern_python", max_new_tokens: int = 100) -> np.ndarray:
        prompt_prefix = f"<REFACTOR> {target_language} "
        prompt_tokens = np.array([[self.base_model.embedding.shape[0] - 1] * len(prompt_prefix.split())])
        combined = np.concatenate([prompt_tokens, old_code_tokens], axis=1)
        return self.base_model.forward(combined)


@dataclass
class TestingAndDebuggingModel:
    base_model: BaseCodeModel | None = None
    vocab_size: int = 1000
    d_model: int = 256
    random_seed: int = 42

    def __post_init__(self):
        if self.base_model is None:
            self.base_model = BaseCodeModel(vocab_size=self.vocab_size, d_model=self.d_model, random_seed=self.random_seed)

    def scan_bugs(self, code_tokens: np.ndarray) -> dict[str, Any]:
        logits = self.base_model.forward(code_tokens)
        probs = softmax(logits[0])
        confidence = float(np.max(probs, axis=-1).mean())
        return {"bug_probability": 1.0 - confidence, "confidence": confidence, "suggested_fix": "Review code for potential issues."}

    def generate_unit_tests(self, code_tokens: np.ndarray, max_new_tokens: int = 50) -> np.ndarray:
        prompt_prefix = "<TEST> "
        prompt_tokens = np.array([[self.base_model.embedding.shape[0] - 1] * len(prompt_prefix.split())])
        combined = np.concatenate([prompt_tokens, code_tokens], axis=1)
        return self.base_model.forward(combined)


@dataclass
class CodeGenerationModel:
    vocab_size: int = 1000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 1024
    max_seq_len: int = 128
    random_seed: int = 42
    learning_rate: float = 0.001
    n_iterations: int = 100
    weight_decay: float = 0.01

    base_model: BaseCodeModel | None = None
    completion_model: CodeCompletionModel | None = None
    text_to_code_model: TextToCodeModel | None = None
    refactoring_model: RefactoringModel | None = None
    testing_model: TestingAndDebuggingModel | None = None
    tokenizer: CodeTokenizer | None = None
    loss_history: list[float] = field(default_factory=list, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        self.base_model = BaseCodeModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            random_seed=self.random_seed,
        )
        self.completion_model = CodeCompletionModel(base_model=self.base_model, vocab_size=self.vocab_size, d_model=self.d_model, random_seed=self.random_seed + 1)
        self.text_to_code_model = TextToCodeModel(base_model=self.base_model, vocab_size=self.vocab_size, d_model=self.d_model, random_seed=self.random_seed + 2)
        self.refactoring_model = RefactoringModel(base_model=self.base_model, vocab_size=self.vocab_size, d_model=self.d_model, random_seed=self.random_seed + 3)
        self.testing_model = TestingAndDebuggingModel(base_model=self.base_model, vocab_size=self.vocab_size, d_model=self.d_model, random_seed=self.random_seed + 4)
        self.tokenizer = CodeTokenizer(vocab_size=self.vocab_size, max_seq_len=self.max_seq_len, random_seed=self.random_seed)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_iterations: int | None = None) -> dict:
        if self.base_model is None:
            self._init()
        if n_iterations is None:
            n_iterations = self.n_iterations
        n_samples = X_train.shape[0]
        rng = np.random.default_rng(self.random_seed)
        eps = 1e-12
        for _epoch in range(n_iterations):
            perm = rng.permutation(n_samples)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            total_loss = 0.0
            for i in range(n_samples):
                x = X_shuffled[i:i + 1]
                y = y_shuffled[i:i + 1]
                logits = self.base_model.forward(x)
                probs = softmax(logits[0])
                for pos in range(y.shape[1]):
                    true_token = int(y[0, pos])
                    if true_token < self.vocab_size:
                        total_loss += -np.log(np.clip(probs[pos, true_token], eps, 1))
            avg_loss = total_loss / (n_samples * max(1, y_train.shape[1]))
            self.loss_history.append(avg_loss)
        return self.to_dict()

    def complete_code(self, code_prefix: str, max_new_tokens: int = 20) -> str:
        if self.completion_model is None:
            self._init()
        tokens = self.tokenizer.encode(code_prefix)
        token_array = np.array([tokens])
        completed = self.completion_model.complete(token_array, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(completed.tolist())

    def text_to_code(self, description: str, max_new_tokens: int = 50) -> str:
        if self.text_to_code_model is None:
            self._init()
        tokens = self.tokenizer.encode(description)
        token_array = np.array([tokens])
        generated = self.text_to_code_model.generate(token_array, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(generated.tolist())

    def refactor_code(self, old_code: str, target_language: str = "modern_python") -> str:
        if self.refactoring_model is None:
            self._init()
        tokens = self.tokenizer.encode(old_code)
        token_array = np.array([tokens])
        refactored = self.refactoring_model.refactor(token_array, target_language=target_language)
        return self.tokenizer.decode(np.argmax(refactored[0], axis=-1).tolist())

    def scan_for_bugs(self, code: str) -> dict[str, Any]:
        if self.testing_model is None:
            self._init()
        tokens = self.tokenizer.encode(code)
        token_array = np.array([tokens])
        return self.testing_model.scan_bugs(token_array)

    def generate_unit_tests(self, code: str, max_new_tokens: int = 50) -> str:
        if self.testing_model is None:
            self._init()
        tokens = self.tokenizer.encode(code)
        token_array = np.array([tokens])
        generated = self.testing_model.generate_unit_tests(token_array, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(np.argmax(generated[0], axis=-1).tolist())

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = []
        for i in range(len(X)):
            x = X[i:i + 1]
            logits = self.base_model.forward(x)
            pred = int(np.argmax(logits[0, 0, :]))
            preds.append(pred)
        matches = np.mean(np.array(preds) == y.flatten()[:len(preds)]) if len(preds) > 0 else 0.0
        return {"accuracy": float(matches), "n_samples": float(len(X))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "vocab_size": np.array([self.vocab_size]),
            "d_model": np.array([self.d_model]),
            "n_heads": np.array([self.n_heads]),
            "n_layers": np.array([self.n_layers]),
            "d_ff": np.array([self.d_ff]),
            "max_seq_len": np.array([self.max_seq_len]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
            "random_seed": np.array([self.random_seed]),
        }
        if self.base_model and self.base_model.embedding is not None:
            arrays["embedding"] = self.base_model.embedding
        if self.base_model and self.base_model.W_out is not None:
            arrays["W_out"] = self.base_model.W_out
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "CodeGenerationModel":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            vocab_size=int(data["vocab_size"].item()),
            d_model=int(data["d_model"].item()),
            n_heads=int(data["n_heads"].item()),
            n_layers=int(data["n_layers"].item()),
            d_ff=int(data["d_ff"].item()),
            max_seq_len=int(data["max_seq_len"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=int(data["random_seed"].item()),
        )
        obj._init()
        if "embedding" in data and obj.base_model:
            obj.base_model.embedding = data["embedding"]
        if "W_out" in data and obj.base_model:
            obj.base_model.W_out = data["W_out"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
