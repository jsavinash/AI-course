#!/usr/bin/env python3
"""Generate standalone HTML documentation for all neural-network apps."""

import os
from pathlib import Path

BASE_DIR = Path("/Users/avi/Documents/ai/apps/neural-networks")
OUTPUT_DIR = BASE_DIR / "docs"

CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: #e0e0e0;
  background: #0f1115;
}
header {
  background: linear-gradient(135deg, #1a1d23 0%, #252a33 100%);
  border-bottom: 1px solid #333;
  padding: 2rem 1rem;
  text-align: center;
}
header h1 { font-size: 1.75rem; color: #fff; margin-bottom: 0.25rem; }
header p { color: #9aa; font-size: 0.95rem; }
main { max-width: 900px; margin: 0 auto; padding: 2rem 1rem; }
.section {
  background: #1a1d23;
  border: 1px solid #2a2e36;
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}
.section h2 {
  font-size: 1.25rem;
  color: #fff;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.section-subtitle { color: #9aa; font-size: 0.9rem; margin-bottom: 1rem; }
.math-block {
  background: #111;
  border: 1px solid #2a2e36;
  border-radius: 6px;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  overflow-x: auto;
  font-family: "SF Mono", Monaco, monospace;
  font-size: 0.9rem;
  color: #cdd6f4;
}
.derivation p { margin-top: 0.5rem; color: #bbc; }
.code-block-wrapper { position: relative; margin: 0.75rem 0; }
.code-block {
  background: #111;
  border: 1px solid #2a2e36;
  border-radius: 6px;
  padding: 1rem;
  overflow-x: auto;
  font-family: "SF Mono", Monaco, monospace;
  font-size: 0.85rem;
  line-height: 1.5;
  color: #cdd6f4;
}
.copy-btn {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  background: #333;
  color: #fff;
  border: none;
  border-radius: 4px;
  padding: 0.25rem 0.5rem;
  cursor: pointer;
  font-size: 0.8rem;
}
.copy-btn:hover { background: #444; }
.api-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
.api-table th, .api-table td {
  text-align: left;
  padding: 0.5rem;
  border-bottom: 1px solid #2a2e36;
  color: #cdd6f4;
}
.api-table th { color: #fff; font-weight: 600; }
.muted { color: #9aa; font-size: 0.9rem; }
footer {
  text-align: center;
  padding: 2rem 1rem;
  color: #9aa;
  font-size: 0.85rem;
  border-top: 1px solid #2a2e36;
}
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>__TITLE__ - AI App Documentation</title>
  <style>
__CSS__
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <p>__SUBTITLE__</p>
  </header>
  <main>
__CONTENT__
  </main>
  <footer>
    <p>Generated documentation for <strong>__APP_NAME__</strong></p>
  </footer>
</body>
</html>
"""


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def code_block(source, lang="python"):
    escaped = esc(source)
    return '<div class="code-block-wrapper">\n  <button class="copy-btn" onclick="navigator.clipboard.writeText(this.parentElement.querySelector(\'code\').innerText)">Copy</button>\n  <pre class="code-block"><code class="language-' + lang + '">' + escaped + '</code></pre>\n</div>'


def math_block(latex):
    return '<div class="math-block">$$' + latex + '$$</div>'


def generate_diffusion():
    title = "Diffusion Image Generation"
    subtitle = "Generates images by systematically removing noise from a random starting state"
    app_name = "diffusion"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>diffusion</code> (package: <code>diffusion_image_generation</code>)</p>
      <p><strong>Purpose:</strong> Implements a Denoising Diffusion Probabilistic Model (DDPM) for synthetic image generation. The model learns to reverse a Gaussian noising process, iteratively denoising random noise into structured 8x8 grayscale images.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>attention-generative/</code>, depending on the shared <code>ai-core</code> workspace package for logging, validation, metrics, model registry, and FastAPI observability middleware.</p>
      <p><strong>Key dependencies:</strong> <code>ai-core</code>, <code>numpy</code>, <code>fastapi</code>, <code>pydantic</code>, <code>prometheus-client</code>, <code>matplotlib</code>.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">DDPM equations, forward/reverse processes, and training objective</p>
      """ + math_block("q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1 - \\beta_t}x_{t-1}, \\beta_t I)") + """
      """ + math_block("p_\\theta(x_{t-1} | x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t), \\Sigma_\\theta(x_t, t))") + """
      """ + math_block("\\mathcal{L}_{simple} = \\mathbb{E}_{t, x_0, \\epsilon} \\left[ \\| \\epsilon - \\epsilon_\\theta(\\sqrt{\\bar{\\alpha}_t}x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\epsilon, t) \\|^2 \\right]") + """
      """ + math_block("\\bar{\\alpha}_t = \\prod_{s=1}^{t} (1 - \\beta_s)") + """
      <div class="derivation">
        <p><strong>Why DDPM?</strong> The forward process is a fixed Markov chain that gradually adds noise. The reverse process is learned by a neural network (SimpleCNN) that predicts the noise added at each step. Training minimizes the MSE between actual and predicted noise. This formulation is stable, requires no adversarial training, and produces high-fidelity samples with a simple training objective.</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">Data flow, state management, and execution lifecycle</p>
      <p>The app follows a standard MLOps lifecycle: synthetic data generation -> validation -> training -> evaluation -> model registry -> FastAPI serving.</p>
      <ul>
        <li><strong>Data flow:</strong> <code>data.py</code> generates 8x8 grayscale synthetic images with additive Gaussian noise.</li>
        <li><strong>Model:</strong> <code>DiffusionModel</code> (dataclass) encapsulates the noise schedule, forward/reverse sampling, and a <code>SimpleCNN</code> denoiser from <code>ai-core</code>.</li>
        <li><strong>State:</strong> Managed via module-level globals in <code>api.py</code> loaded during FastAPI <code>lifespan</code>.</li>
        <li><strong>Training path:</strong> <code>train.py:train()</code> loads data, validates, instantiates <code>DiffusionModel</code>, calls <code>fit()</code>, evaluates, saves artifacts, and registers with <code>ModelRegistry</code> (optionally MLflow).</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>

      <h3>model.py - DiffusionModel</h3>
      <p>The core model. Key methods:</p>
      """ + code_block('''@dataclass
class DiffusionModel:
    def _init_noise_schedule(self) -> None:
        """Initialize linear noise schedule betas, alphas, and cumulative products."""
        self.betas = np.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def _q_sample(self, x_0: np.ndarray, t: int, noise: np.ndarray | None = None) -> np.ndarray:
        """Sample from q(x_t | x_0): add noise to data at timestep t."""
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t]
        return sqrt_alpha_t * x_0 + sqrt_one_minus_t * noise

    def fit(self, X: np.ndarray, n_iterations: int | None = None) -> "DiffusionModel":
        """Train: for each sample, sample a random t, add noise, predict noise with SimpleCNN."""
        self.model = SimpleCNN(...)
        for _epoch in range(n_iterations):
            for i in range(n_samples):
                x_0 = X_img[i:i + 1]
                t = rng.integers(0, self.timesteps)
                noise = rng.normal(0, 1, size=x_0.shape)
                x_t = self._q_sample(x_0, t, noise)
                pred = self.model.predict_proba(x_t)
                loss = np.mean((noise - pred) ** 2)
            self.loss_history.append(total_loss / n_samples)''', lang="python") + """

      <h3>api.py - FastAPI lifespan and endpoints</h3>
      <p>The API initializes model, metrics, validator, and drift detector in an async context manager. Key pattern: lazy model loading with fallback to baseline training if no persisted model exists.</p>
      """ + code_block('''@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("diffusion_image_generation", port=METRICS_PORT)
    app.state.metrics = _metrics
    _validator = DataValidator(create_diffusion_image_generation_schema())
    _drift_detector = DriftDetector(feature_names=..., psi_threshold=DRIFT_THRESHOLD)
    _model, _model_version = _load_model()
    _reference_data = _load_reference_data()
    yield
    logger.info("Shutting down diffusion-image-generation API")''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from diffusion_image_generation.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_samples=500,
    n_filters=8,
    timesteps=100,
    learning_rate=0.01,
    n_iterations=200,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m diffusion.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>API</h3>
      """ + code_block('''import requests
response = requests.post("http://localhost:8000/predict", json={"timesteps_to_run": 100})
print(response.json())''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "attention-generative_diffusion.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_gan():
    title = "GAN Image Generation"
    subtitle = "Creates new synthetic images using adversarial competition"
    app_name = "gan"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>gan</code> (package: <code>gan_image_generation</code>)</p>
      <p><strong>Purpose:</strong> Implements a Generative Adversarial Network (GAN) that learns to generate 8x8 grayscale images by training a generator and discriminator in a minimax game.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>attention-generative/</code>, sharing the <code>ai-core</code> workspace for observability, validation, and model registry.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">GAN minimax objective and adversarial training</p>
      """ + math_block("\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{data}}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z}[\\log(1 - D(G(z)))]") + """
      """ + math_block("\\mathcal{L}_{D} = -\\frac{1}{m} \\sum_{i=1}^{m} [\\log D(x^{(i)}) + \\log(1 - D(G(z^{(i)})))]") + """
      """ + math_block("\\mathcal{L}_{G} = -\\frac{1}{m} \\sum_{i=1}^{m} \\log D(G(z^{(i)}))") + """
      <div class="derivation">
        <p><strong>Why GAN?</strong> The discriminator D maximizes its ability to distinguish real from fake. The generator G minimizes the probability of being detected. At Nash equilibrium, G reproduces the true data distribution. The implementation uses binary cross-entropy with sigmoid outputs and manual backpropagation through both networks.</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">Two-network competition and MLOps lifecycle</p>
      <ul>
        <li><strong>Generator:</strong> Dense(latent_dim -> hidden_dim, ReLU) -> Dense(hidden_dim -> n_pixels, sigmoid). Takes a 16-dim latent vector and outputs an 8x8 image.</li>
        <li><strong>Discriminator:</strong> Dense(n_pixels -> hidden_dim, ReLU) -> Dense(hidden_dim -> 1, sigmoid). Outputs a real/fake probability.</li>
        <li><strong>Training:</strong> Alternating updates - first update discriminator on real and fake batches, then update generator by backpropagating through the discriminator.</li>
        <li><strong>Loss tracking:</strong> Separate _gen_loss_history and _disc_loss_history lists enable per-component monitoring.</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>

      <h3>model.py - GAN class</h3>
      """ + code_block('''@dataclass
class GAN:
    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        # Generator weights
        self._gen_W1 = rng.normal(0, np.sqrt(1.0 / self.latent_dim), (self.latent_dim, self.hidden_dim))
        self._gen_W2 = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.n_features))
        # Discriminator weights
        self._disc_W1 = rng.normal(0, np.sqrt(1.0 / self.n_features), (self.n_features, self.hidden_dim))
        self._disc_W2 = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, 1))

    def fit(self, X_real: np.ndarray, n_iterations: int | None = None) -> "GAN":
        for _epoch in range(n_iterations):
            z = rng.normal(0, 1, size=(n_samples, self.latent_dim))
            gen_img, gen_cache = self._generator_forward(z)
            real_prob, real_cache = self._discriminator_forward(X_real)
            fake_prob, fake_cache = self._discriminator_forward(gen_img)

            # Update discriminator
            disc_loss = -np.mean(np.log(real_prob + eps) + np.log(1 - fake_prob + eps))
            self._disc_W1 -= lr * (grad_real["dW_disc1"] + grad_fake["dW_disc1"] + wd * self._disc_W1)

            # Update generator (backprop through discriminator)
            gen_loss = -np.mean(np.log(fake_prob + eps))
            d_fake_logits_g = -(1 - fake_prob) / n_samples
            grad_fake_g = self._discriminator_backward(d_fake_logits_g, fake_cache)
            d_gen_img = grad_fake_g["dX"]
            d_gen_logits = d_gen_img * sigmoid_derivative(sigmoid(gen_cache["logits"]))
            gen_grads = self._generator_backward(d_gen_logits, gen_cache)''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from gan_image_generation.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_samples=500,
    latent_dim=16,
    hidden_dim=32,
    learning_rate=0.01,
    n_iterations=200,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m gan.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>API</h3>
      """ + code_block('''import requests
response = requests.post("http://localhost:8002/predict", json={"latent_vector": [0.1]*16})
print(response.json())''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "attention-generative_gan.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_transformers():
    title = "Transformer Language Modeling"
    subtitle = "Next-token prediction using self-attention mechanisms"
    app_name = "transformers"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>transformers</code> (package: <code>transformer_language_modeling</code>)</p>
      <p><strong>Purpose:</strong> Implements a compact Transformer language model with multi-head self-attention for next-token prediction on synthetic token sequences.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>attention-generative/</code>, leveraging <code>ai-core</code> for MLOps infrastructure.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">Self-attention, multi-head attention, and training loss</p>
      """ + math_block("\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V") + """
      """ + math_block("\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h)W^O") + """
      """ + math_block("\\mathcal{L} = -\\sum_{t=1}^{T} \\log P(w_t | w_{<t}; \\theta)") + """
      <div class="derivation">
        <p><strong>Why Transformer?</strong> Self-attention captures long-range dependencies without recurrence. The scaled dot-product attention prevents gradient vanishing by dividing by sqrt(d_k). Causal masking ensures autoregressive training. Layer normalization and residual connections stabilize deep stacks.</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">Encoder-decoder blocks, attention, and feed-forward networks</p>
      <ul>
        <li><strong>Input:</strong> Token IDs (batch, seq_len=16) embedded via learned token + positional embeddings.</li>
        <li><strong>Attention:</strong> SimpleAttention computes Q, K, V, applies causal mask, and returns attended output.</li>
        <li><strong>Norms:</strong> Three LayerNorm instances stabilize training.</li>
        <li><strong>Feed-forward:</strong> Two-layer MLP with ReLU.</li>
        <li><strong>Output:</strong> Dense projection to vocab_size=100, softmax for next-token probabilities.</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>
      <h3>model.py - SimpleAttention and TransformerLanguageModel</h3>
      """ + code_block('''class SimpleAttention:
    def forward(self, X: np.ndarray) -> np.ndarray:
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores[:, causal_mask] = -1e9
        attn = softmax(scores)
        out = attn @ V
        return out @ self.W_o

class TransformerLanguageModel:
    def _forward(self, tokens: np.ndarray):
        emb = self.token_embedding[tokens] + self.position_embedding[np.arange(self.seq_len)]
        normed = self.ln1.forward(emb)
        attn_out = self.attn.forward(normed)
        h = emb + attn_out
        ff_in = self.ln2.forward(h)
        ff_hidden = np.maximum(0, ff_in @ self.W_ff1 + self.b_ff1)
        ff_out = ff_hidden @ self.W_ff2 + self.b_ff2
        h2 = h + ff_out
        logits = self.ln3.forward(h2) @ self.W_out + self.b_out
        return logits, cache''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from transformer_language_modeling.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_samples=500,
    d_model=32,
    num_heads=4,
    hidden_dim=64,
    learning_rate=0.05,
    n_iterations=300,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m transformers.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>API</h3>
      """ + code_block('''import requests
response = requests.post("http://localhost:8001/predict", json={"tokens": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]})
print(response.json())''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "attention-generative_transformers.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_vae():
    title = "VAE Data Generation"
    subtitle = "Generates new data variations by sampling from a learned probabilistic latent space"
    app_name = "vae"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>vae</code> (package: <code>vae_data_generation</code>)</p>
      <p><strong>Purpose:</strong> Implements a Variational Autoencoder (VAE) that learns a probabilistic latent space and can generate new data variations by sampling from it.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>attention-generative/</code>, sharing <code>ai-core</code> for MLOps.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">ELBO, encoder/decoder, and KL regularization</p>
      """ + math_block("q_\\phi(z|x) = \\mathcal{N}(\\mu_\\phi(x), \\sigma_\\phi^2(x))") + """
      """ + math_block("\\mathcal{L} = \\underbrace{\\mathbb{E}_{z \\sim q_\\phi}[\\log p_\\theta(x|z)]}_{\\text{Reconstruction}} - \\underbrace{D_{KL}(q_\\phi(z|x) \\| p(z))}_{\\text{Regularization}}") + """
      """ + math_block("D_{KL} = \\frac{1}{2} \\sum_{j=1}^{J} \\left(1 + \\log(\\sigma_j^2) - \\mu_j^2 - \\sigma_j^2\\right)") + """
      <div class="derivation">
        <p><strong>Why VAE?</strong> The reparameterization trick z = mu + std * eps enables backpropagation through stochastic samples. The KL divergence regularizes the latent space to match a standard normal prior, making it possible to generate new data by sampling z ~ N(0,1).</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">Encoder-decoder with reparameterization and dual loss</p>
      <ul>
        <li><strong>Encoder:</strong> Maps input to mu and log_var. Samples z = mu + exp(0.5*log_var) * eps.</li>
        <li><strong>Decoder:</strong> Reconstructs x_recon = sigmoid(decoder(z)).</li>
        <li><strong>Loss:</strong> recon_loss + kl_loss. Reconstruction is MSE; KL is analytical.</li>
        <li><strong>Training:</strong> Manual backprop with gradient clipping.</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>
      <h3>model.py - VAE class</h3>
      """ + code_block('''@dataclass
class VAE:
    def _encode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        h = relu(X @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar
        eps = np.random.default_rng(self.random_seed).normal(0, 1, size=mu.shape)
        std = np.exp(0.5 * log_var)
        z = mu + std * eps  # reparameterization trick
        return mu, log_var, z, cache

    def fit(self, X: np.ndarray, n_iterations: int | None = None) -> "VAE":
        for _epoch in range(n_iterations):
            for i in range(n_samples):
                mu, log_var, z, cache = self._encode(x_i)
                x_recon, cache = self._decode(z, cache)
                recon_loss = np.mean((x_i - x_recon) ** 2)
                kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
                total_loss = recon_loss + kl_loss
                # manual backward pass with gradient clipping''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from vae_data_generation.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_samples=500,
    latent_dim=16,
    hidden_dim=64,
    learning_rate=0.01,
    n_iterations=300,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m vae.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>Generation</h3>
      """ + code_block('''from vae_data_generation.model import VAE
model = VAE.load("vae_model.npz")
samples = model.generate(n_samples=10)''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "attention-generative_vae.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_gnn():
    title = "GNN Social Networks"
    subtitle = "Graph Neural Networks that optimize directly on graph structures for social network analysis"
    app_name = "gnn-social-networks"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>gnn-social-networks</code> (package: <code>gnn_social_networks</code>)</p>
      <p><strong>Purpose:</strong> Implements a Graph Convolutional Network (GCN) for node classification on social network graph data.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>graph-physics-informed/</code>, sharing <code>ai-core</code> for MLOps.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">Graph convolution and message passing</p>
      """ + math_block("H^{(l+1)} = \\sigma(\\tilde{A}_{norm} H^{(l)} W^{(l)})") + """
      """ + math_block("\\tilde{A}_{norm} = D^{-1/2} (A + I) D^{-1/2}") + """
      <div class="derivation">
        <p><strong>Why GCN?</strong> Standard convolutions don't apply to irregular graphs. GCNs aggregate neighbor features via normalized adjacency multiplication, enabling end-to-end learning on graph-structured data. The symmetric normalization preserves node degree scale.</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">Two GCN layers + readout for node classification</p>
      <ul>
        <li><strong>Input:</strong> Node features (n_nodes, n_features) and adjacency matrix (n_nodes, n_nodes).</li>
        <li><strong>GCNLayer:</strong> Computes A_norm @ H @ W. Caches H and A_norm for backward.</li>
        <li><strong>Backward:</strong> dW = H.T @ (A_norm.T @ dout), dH = A_norm @ dout @ W.T.</li>
        <li><strong>Readout:</strong> Final dense layer + softmax for multi-class node classification.</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>
      <h3>model.py - GCNLayer and GNNSocialNetworks</h3>
      """ + code_block('''def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """A_norm = D^{-1/2} (A + I) D^{-1/2}"""
    A_eye = A + np.eye(A.shape[0])
    D = np.diag(np.sum(A_eye, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
    return D_inv_sqrt @ A_eye @ D_inv_sqrt

class GCNLayer:
    def forward(self, H: np.ndarray, A_norm: np.ndarray) -> np.ndarray:
        out = A_norm @ H @ self.W
        self._cache = {"H": H, "A_norm": A_norm}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        H = self._cache["H"]
        A_norm = self._cache["A_norm"]
        self.dW = H.T @ (A_norm.T @ dout)
        dH = A_norm @ dout @ self.W.T
        return dH''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from gnn_social_networks.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_nodes=20,
    hidden_dim=16,
    learning_rate=0.05,
    n_iterations=200,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m gnn_social_networks.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>API</h3>
      """ + code_block('''import requests
response = requests.post("http://localhost:8009/predict", json={
    "features": [0.1]*32,
    "adjacency_row": [0.0]*20
})
print(response.json())''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "graph-physics-informed_gnn-social-networks.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_pinn():
    title = "PINN Heat Equation"
    subtitle = "Physics-Informed Neural Network for solving the heat equation"
    app_name = "pinn-heat-equation"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>pinn-heat-equation</code> (package: <code>pinn_heat_equation</code>)</p>
      <p><strong>Purpose:</strong> Implements a Physics-Informed Neural Network (PINN) that solves the heat equation u_t = alpha * u_xx while respecting physical laws as soft constraints.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>graph-physics-informed/</code>, sharing <code>ai-core</code> for MLOps.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">PINN loss decomposition and PDE residual</p>
      """ + math_block("\\mathcal{L}_{total} = \\mathcal{L}_{data} + \\lambda \\mathcal{L}_{pde}") + """
      """ + math_block("\\mathcal{L}_{data} = \\frac{1}{N} \\sum_{i=1}^{N} |u_\\theta(x_i, t_i) - u_i|^2") + """
      """ + math_block("\\mathcal{L}_{pde} = \\frac{1}{N_f} \\sum_{i=1}^{N_f} \\left| \\frac{\\partial u_\\theta}{\\partial t} - \\alpha \\frac{\\partial^2 u_\\theta}{\\partial x^2} \\right|^2") + """
      <div class="derivation">
        <p><strong>Why PINN?</strong> By embedding the PDE as a soft constraint via automatic differentiation (here approximated by finite differences), the network learns a solution that respects physical laws even in unlabeled regions of the domain. This is particularly valuable when data is sparse or expensive to obtain.</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">MLP with physics residual computation</p>
      <ul>
        <li><strong>Network:</strong> Fully connected MLP with tanh activations. Input: (x, t) coordinates. Output: temperature u(x, t).</li>
        <li><strong>Physics residual:</strong> Computed via central finite differences: du_dt - alpha * d2u_dx2. Epsilon 1e-5 perturbations approximate derivatives.</li>
        <li><strong>Loss:</strong> Combined data MSE + physics MSE. Gradient clipping prevents instability from the residual term.</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>
      <h3>model.py - PINNHeatEquation</h3>
      """ + code_block('''def _compute_physics_residual(self, X: np.ndarray, u_pred: np.ndarray) -> np.ndarray:
    """Heat equation residual: du/dt - alpha * d2u/dx2 via finite differences."""
    eps = 1e-5
    X_x_plus = X.copy(); X_x_plus[:, 0] += eps
    X_x_minus = X.copy(); X_x_minus[:, 0] -= eps
    X_t_plus = X.copy(); X_t_plus[:, 1] += eps
    X_t_minus = X.copy(); X_t_minus[:, 1] -= eps

    u_x_plus, _ = self._forward(X_x_plus)
    u_x_minus, _ = self._forward(X_x_minus)
    u_t_plus, _ = self._forward(X_t_plus)
    u_t_minus, _ = self._forward(X_t_minus)

    du_dt = (u_t_plus - u_t_minus) / (2 * eps)
    d2u_dx2 = (u_x_plus - 2 * u_pred + u_x_minus) / (eps ** 2)
    residual = du_dt - self.alpha * d2u_dx2
    return residual''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from pinn_heat_equation.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_samples=200,
    alpha=0.01,
    hidden_dim=32,
    n_layers=2,
    learning_rate=0.01,
    n_iterations=500,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m pinn_heat_equation.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>Prediction</h3>
      """ + code_block('''import numpy as np
from pinn_heat_equation.model import PINNHeatEquation
model = PINNHeatEquation.load("pinn_model.npz")
X = np.array([[0.5, 0.1]])  # x=0.5, t=0.1
u = model.predict(X)
residual = model.predict_proba(X)
print(f"Temperature: {u}, Residual: {residual}")''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "graph-physics-informed_pinn-heat-equation.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_snn():
    title = "SNN Image Classification"
    subtitle = "Spiking neural networks using neuromorphic computing with discrete spikes"
    app_name = "snn-image-classification"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>snn-image-classification</code> (package: <code>snn_image_classification</code>)</p>
      <p><strong>Purpose:</strong> Implements a Spiking Neural Network (SNN) using Leaky Integrate-and-Fire (LIF) neurons for image classification. Neurons communicate via discrete spike events.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>graph-physics-informed/</code>, sharing <code>ai-core</code> for MLOps.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">LIF neuron dynamics and spike-based computation</p>
      """ + math_block("\\tau_m \\frac{dV_m}{dt} = -(V_m - V_{rest}) + R_m I(t)") + """
      """ + math_block("\\text{if } V_m \\geq V_{th}: \\text{ emit spike}, V_m \\leftarrow V_{reset}") + """
      """ + math_block("S(t) = \\sum_{i} \\delta(t - t_i)") + """
      """ + math_block("\\tau_s \\frac{dS}{dt} = -S") + """
      <div class="derivation">
        <p><strong>Why SNN?</strong> Event-driven computation is energy-efficient and biologically plausible. The surrogate gradient (sigmoid derivative of the spike) enables training despite the non-differentiable threshold operation.</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">Temporal coding with LIF neurons</p>
      <ul>
        <li><strong>Input encoding:</strong> Rate coding - pixel intensity maps to input current.</li>
        <li><strong>LIFNeuron:</strong> Membrane potential V = leak_rate * V + (1 - leak_rate) * (V_rest + I_in). Spike when V >= threshold, then reset.</li>
        <li><strong>Temporal pooling:</strong> Spike trains averaged over timesteps to produce continuous features for the readout layer.</li>
        <li><strong>Surrogate gradient:</strong> sigmoid_derivative(spikes) approximates the gradient of the step function.</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>
      <h3>model.py - LIFNeuron and SNNImageClassification</h3>
      """ + code_block('''class LIFNeuron:
    def forward(self, x: np.ndarray, n_timesteps: int = 10) -> np.ndarray:
        membrane = np.full((batch_size, self.n_neurons), self.v_rest)
        for t in range(n_timesteps):
            I_in = x @ self.W + self.b
            membrane = self.leak_rate * membrane + (1 - self.leak_rate) * (self.v_rest + I_in)
            new_spikes = (membrane >= self.threshold).astype(np.float32)
            membrane = np.where(new_spikes > 0, self.reset_voltage, membrane)
            spike_trains[:, :, t] = new_spikes
        return spike_trains

    def backward(self, dout: np.ndarray) -> np.ndarray:
        surrogate = sigmoid_derivative(spikes_t)
        grad_spikes = grad_out_t * surrogate
        self.dW += x.T @ grad_spikes / n_timesteps
        self.db += np.sum(grad_spikes, axis=0) / n_timesteps''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from snn_image_classification.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_samples=500,
    hidden_dim=128,
    learning_rate=0.01,
    n_iterations=200,
    n_timesteps=10,
    threshold=1.0,
    leak_rate=0.9,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m snn_image_classification.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>Prediction</h3>
      """ + code_block('''import numpy as np
from snn_image_classification.model import SNNImageClassification
model = SNNImageClassification.load("snn_model.npz")
X = np.random.rand(1, 64)
probs = model.predict_proba(X)
print("Predicted class:", np.argmax(probs))''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "graph-physics-informed_snn-image-classification.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_autoencoder():
    title = "Autoencoders Dimensionality Reduction"
    subtitle = "Autoencoder networks that compress input into a latent code and reconstruct it"
    app_name = "autoencoders-dimensionality-reduction"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>autoencoders-dimensionality-reduction</code> (package: <code>autoencoders_dimensionality_reduction</code>)</p>
      <p><strong>Purpose:</strong> Implements an autoencoder for dimensionality reduction and denoising. Compresses input into a lower-dimensional latent code and reconstructs it.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>unsupervised/</code>, sharing <code>ai-core</code> for MLOps.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">Encoder-decoder reconstruction and bottleneck</p>
      """ + math_block("z = f(x) = \\sigma(W_e x + b_e) \\quad \\text{(encoder)}") + """
      """ + math_block("\\hat{x} = g(z) = \\sigma(W_d z + b_d) \\quad \\text{(decoder)}") + """
      """ + math_block("\\mathcal{L} = \\|x - \\hat{x}\\|^2 + \\lambda (\\|W_e\\|^2 + \\|W_d\\|^2)") + """
      <div class="derivation">
        <p><strong>Why autoencoder?</strong> The bottleneck forces the network to learn compressed, meaningful representations. L2 regularization prevents overfitting. Optionally, noise can be injected during training (denoising autoencoder) to improve robustness.</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">Symmetric encoder-decoder with bottleneck</p>
      <ul>
        <li><strong>Encoder:</strong> Dense(n_features -> hidden_dim, ReLU) -> Dense(hidden_dim -> latent_dim, ReLU)</li>
        <li><strong>Decoder:</strong> Dense(latent_dim -> hidden_dim, ReLU) -> Dense(hidden_dim -> n_features, sigmoid)</li>
        <li><strong>Loss:</strong> MSE between input and reconstruction, plus L2 regularization on weights.</li>
        <li><strong>Denoising:</strong> Optional noise_rate zeroes out random input features during training.</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>
      <h3>model.py - Autoencoder</h3>
      """ + code_block('''@dataclass
class Autoencoder:
    def _forward(self, X: np.ndarray, dropout: bool = True) -> tuple[np.ndarray, dict]:
        h1 = relu(X @ self._W1 + self._b1)
        h2 = relu(h1 @ self._W2 + self._b2)
        h3 = relu(h2 @ self._W3 + self._b3)
        out = sigmoid(h3 @ self._W4 + self._b4)
        return out, cache

    def fit(self, X: np.ndarray, n_iterations: int | None = None) -> "Autoencoder":
        for _epoch in range(n_iterations):
            X_shuffled = X[rng.permutation(n_samples)]
            for i in range(n_samples):
                if self.noise_rate > 0:
                    noise_mask = rng.random(x_i.shape) > self.noise_rate
                    x_noisy = x_i * noise_mask
                else:
                    x_noisy = x_i
                out, cache = self._forward(x_noisy)
                loss = np.mean((x_i - out) ** 2)
                # manual backprop with gradient clipping''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from autoencoders_dimensionality_reduction.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_samples=500,
    latent_dim=8,
    hidden_dim=16,
    learning_rate=0.01,
    n_iterations=300,
    noise_rate=0.1,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m autoencoders_dimensionality_reduction.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>Encoding and Reconstruction</h3>
      """ + code_block('''from autoencoders_dimensionality_reduction.model import Autoencoder
model = Autoencoder.load("autoencoders_model.npz")
latent = model.encode(X)
recon = model.decode(latent)
anomaly_scores = model.predict_proba(X)''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "unsupervised_autoencoders-dimensionality-reduction.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_dbn():
    title = "Deep Belief Networks"
    subtitle = "Generative graphical models composed of multiple layers of latent variables"
    app_name = "deep-belief-networks"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>deep-belief-networks</code> (package: <code>deep_belief_networks</code>)</p>
      <p><strong>Purpose:</strong> Implements a Deep Belief Network (DBN) - a stack of Restricted Boltzmann Machines (RBMs) greedily pre-trained for unsupervised feature learning.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>unsupervised/</code>, sharing <code>ai-core</code> for MLOps.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">RBM energy, contrastive divergence, and greedy pre-training</p>
      """ + math_block("E(v, h) = -\\sum_i a_i v_i - \\sum_j b_j h_j - \\sum_{i,j} v_i h_j W_{ij}") + """
      """ + math_block("P(h_j=1|v) = \\sigma(b_j + \\sum_i W_{ij} v_i)") + """
      """ + math_block("\\Delta W_{ij} = \\epsilon (\\langle v_i h_j \\rangle_{data} - \\langle v_i h_j \\rangle_{recon})") + """
      <div class="derivation">
        <p><strong>Why DBN?</strong> Each RBM is trained greedily layer-by-layer. The hidden activations of one RBM become the visible inputs of the next. This pre-training initializes deep networks in a region of weight space conducive to further supervised fine-tuning.</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">Stacked RBMs with Contrastive Divergence</p>
      <ul>
        <li><strong>RBM:</strong> Bipartite graph with binary visible and hidden units. No intra-layer connections.</li>
        <li><strong>Training:</strong> Contrastive Divergence (CD-k). Sample h ~ P(h|v), then reconstruct v' ~ P(v|h), then h' ~ P(h|v'). Update weights with v*h - v'*h'.</li>
        <li><strong>Stacking:</strong> After training RBM1, its hidden probabilities become the visible inputs of RBM2, and so on.</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>
      <h3>model.py - RBM and DeepBeliefNetwork</h3>
      """ + code_block('''class RBM:
    def fit(self, X: np.ndarray, n_epochs: int = 50) -> "RBM":
        for _epoch in range(n_epochs):
            for i in range(n_samples):
                v = X_shuffled[i:i + 1]
                h_prob, h_sample = self._sample_h(v, rng)
                for _ in range(self.n_cd_steps - 1):
                    v_prob, v_sample = self._sample_v(h_sample, rng)
                    h_prob, h_sample = self._sample_h(v_prob, rng)
                v_k_prob, v_k_sample = self._sample_v(h_sample, rng)
                h_k_prob, _ = self._sample_h(v_k_sample, rng)
                self.dW = np.outer(v[0], h_prob[0]) - np.outer(v_k_prob[0], h_k_prob[0])
                self.W -= self.learning_rate * (self.dW + self.weight_decay * self.W)

class DeepBeliefNetwork:
    def fit(self, X: np.ndarray, n_epochs: int | None = None) -> "DeepBeliefNetwork":
        data = X.copy()
        for layer_idx, hidden_dim in enumerate(self.hidden_dims):
            rbm = RBM(n_visible=data.shape[1], n_hidden=hidden_dim, ...)
            rbm.fit(data, n_epochs=n_epochs)
            self.layers.append(rbm)
            data = rbm.transform(data)''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from deep_belief_networks.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_samples=500,
    hidden_dims=[16, 8],
    learning_rate=0.05,
    n_cd_steps=1,
    n_epochs=100,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m deep_belief_networks.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>Feature Extraction</h3>
      """ + code_block('''from deep_belief_networks.model import DeepBeliefNetwork
model = DeepBeliefNetwork.load("dbn_model.npz")
latent = model.transform(X)
recon = model.reconstruct(X)''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "unsupervised_deep-belief-networks.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_rbm():
    title = "Restricted Boltzmann Machines"
    subtitle = "Stochastic neural networks that learn probability distributions over binary inputs"
    app_name = "restricted-boltzmann-machines"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>restricted-boltzmann-machines</code> (package: <code>rbm_feature_learning</code>)</p>
      <p><strong>Purpose:</strong> Implements a single-layer Restricted Boltzmann Machine (RBM) for unsupervised feature learning from binary inputs.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>unsupervised/</code>, sharing <code>ai-core</code> for MLOps.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">RBM energy function and Contrastive Divergence</p>
      """ + math_block("E(v, h) = -\\sum_i a_i v_i - \\sum_j b_j h_j - \\sum_{i,j} v_i h_j W_{ij}") + """
      """ + math_block("P(h_j=1|v) = \\sigma(b_j + \\sum_i W_{ij} v_i)") + """
      """ + math_block("\\Delta W_{ij} = \\epsilon (\\langle v_i h_j \\rangle_{data} - \\langle v_i h_j \\rangle_{recon})") + """
      <div class="derivation">
        <p><strong>Why RBM?</strong> RBMs learn a joint probability distribution over visible and hidden units. The bipartite structure (no intra-layer connections) makes Gibbs sampling efficient. CD-k approximates the intractable log-likelihood gradient with a short Gibbs chain.</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">Bipartite graph with stochastic binary units</p>
      <ul>
        <li><strong>Architecture:</strong> Fully connected bipartite graph between visible (n_features) and hidden (n_hidden) units.</li>
        <li><strong>Sampling:</strong> Bernoulli sampling from sigmoid probabilities for both directions.</li>
        <li><strong>Training:</strong> CD-k: positive phase from data, negative phase from reconstructed samples.</li>
        <li><strong>Output:</strong> Hidden activations serve as feature embeddings; reconstruction gives a denoised output.</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>
      <h3>model.py - RBM</h3>
      """ + code_block('''class RBM:
    def _sample_h(self, v: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        probs = _sigmoid(v @ self.W + self.c)
        samples = _bernoulli_sample(probs, rng)
        return probs, samples

    def fit(self, X: np.ndarray, n_epochs: int | None = None) -> "RBM":
        for _epoch in range(n_epochs):
            for i in range(n_samples):
                v = X_shuffled[i:i + 1]
                h_prob, h_sample = self._sample_h(v, rng)
                for _ in range(self.n_cd_steps - 1):
                    v_prob, v_sample = self._sample_v(h_sample, rng)
                    h_prob, h_sample = self._sample_h(v_prob, rng)
                v_k_prob, v_k_sample = self._sample_v(h_sample, rng)
                h_k_prob, _ = self._sample_h(v_k_sample, rng)
                dW = np.outer(v[0], h_prob[0]) - np.outer(v_k_prob[0], h_k_prob[0])
                self.W -= self.learning_rate * (dW + self.weight_decay * self.W)''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from rbm_feature_learning.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_samples=500,
    n_hidden=16,
    learning_rate=0.05,
    n_cd_steps=1,
    n_epochs=100,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m restricted_boltzmann_machines.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>Feature Extraction</h3>
      """ + code_block('''from rbm_feature_learning.model import RBM
model = RBM.load("rbm_model.npz")
latent = model.transform(X_binary)
recon = model.reconstruct(X_binary)''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "unsupervised_restricted-boltzmann-machines.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_som():
    title = "Self-Organizing Maps"
    subtitle = "Unsupervised networks that produce a low-dimensional discretized representation of the input space"
    app_name = "self-organizing-maps"

    content = """
    <section class="section">
      <h2><span>1</span> Overview &amp; Monorepo Context</h2>
      <p class="section-subtitle">Name, purpose, and role within the broader monorepo</p>
      <p><strong>Name:</strong> <code>self-organizing-maps</code> (package: <code>self_organizing_maps</code>)</p>
      <p><strong>Purpose:</strong> Implements a Self-Organizing Map (SOM) for unsupervised clustering and visualization. Projects high-dimensional data onto a 2D grid while preserving topological relationships.</p>
      <p><strong>Monorepo role:</strong> Lives under <code>unsupervised/</code>, sharing <code>ai-core</code> for MLOps.</p>
    </section>

    <section class="section">
      <h2><span>2</span> The Mathematical Foundation</h2>
      <p class="section-subtitle">Competitive learning and Gaussian neighborhood adaptation</p>
      """ + math_block("c(x) = \\arg\\min_i \\|x - w_i\\|") + """
      """ + math_block("w_i(t+1) = w_i(t) + \\alpha(t) \\cdot h_{ci}(t) \\cdot (x(t) - w_i(t))") + """
      """ + math_block("h_{ci}(t) = \\exp\\left(-\\frac{\\|r_c - r_i\\|^2}{2\\sigma(t)^2}\\right)") + """
      <div class="derivation">
        <p><strong>Why SOM?</strong> The Best Matching Unit (BMU) competition creates a topologically ordered map. The Gaussian neighborhood function h_ci ensures that not only the BMU but also its neighbors adapt, preserving input space topology on the 2D grid.</p>
      </div>
    </section>

    <section class="section">
      <h2><span>3</span> Core Logic &amp; Architecture</h2>
      <p class="section-subtitle">2D grid of neurons with competitive learning</p>
      <ul>
        <li><strong>Weights:</strong> Each neuron on a grid_height x grid_width grid holds a weight vector of dimension n_features.</li>
        <li><strong>BMU search:</strong> Euclidean distance to all neurons; the closest wins.</li>
        <li><strong>Neighborhood:</strong> Gaussian decay based on grid distance from BMU.</li>
        <li><strong>Decay:</strong> Both learning rate alpha(t) and neighborhood radius sigma(t) decay linearly over iterations.</li>
      </ul>
    </section>

    <section class="section">
      <h2><span>4</span> Detailed Code Walkthrough</h2>
      <p class="section-subtitle">Key files, classes, and critical logic</p>
      <h3>model.py - SelfOrganizingMap</h3>
      """ + code_block('''class SelfOrganizingMap:
    def _find_bmu(self, x: np.ndarray) -> tuple[int, int]:
        distances = np.sqrt(np.sum((self.weights - x) ** 2, axis=1))
        bmu_idx = np.argmin(distances)
        row = bmu_idx // self.grid_width
        col = bmu_idx % self.grid_width
        return int(row), int(col)

    def _neighborhood(self, bmu_row: int, bmu_col: int, sigma: float) -> np.ndarray:
        neighborhood = np.zeros((self.grid_height, self.grid_width))
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                dist = np.sqrt((r - bmu_row) ** 2 + (c - bmu_col) ** 2)
                neighborhood[r, c] = np.exp(-dist ** 2 / (2 * sigma ** 2 + 1e-8))
        return neighborhood.flatten()

    def fit(self, X: np.ndarray, n_iterations: int | None = None) -> "SelfOrganizingMap":
        for iteration in range(n_iterations):
            t = iteration / n_iterations
            lr = self.learning_rate * (1 - t)
            sigma = self.sigma * (1 - t)
            for _i in range(n_samples):
                x = X[rng.integers(0, n_samples)]
                bmu_row, bmu_col = self._find_bmu(x)
                neighborhood = self._neighborhood(bmu_row, bmu_col, sigma)
                for n in range(self.n_neurons):
                    delta = lr * neighborhood[n] * (x - self.weights[n])
                    self.weights[n] += delta''', lang="python") + """
    </section>

    <section class="section">
      <h2><span>5</span> Examples &amp; Usage</h2>
      <p class="section-subtitle">Initialize, run, and consume the codebase</p>
      <h3>Training</h3>
      """ + code_block('''from pathlib import Path
from self_organizing_maps.train import train

metrics = train(
    model_dir=Path("./artifacts/models"),
    n_samples=500,
    grid_height=5,
    grid_width=5,
    learning_rate=0.5,
    n_iterations=300,
    sigma=2.0,
)
print(metrics)''', lang="python") + """
      <h3>CLI</h3>
      """ + code_block("uv run python -m self_organizing_maps.train --model-dir ./artifacts/models", lang="bash") + """
      <h3>Clustering</h3>
      """ + code_block('''import numpy as np
from self_organizing_maps.model import SelfOrganizingMap
model = SelfOrganizingMap.load("som_model.npz")
bmu_coords = model.predict(X)
quantization_error = model.predict_proba(X)''', lang="python") + """
    </section>
    """

    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__APP_NAME__", app_name).replace("__CSS__", CSS).replace("__CONTENT__", content)
    out_file = OUTPUT_DIR / "unsupervised_self-organizing-maps.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated: {out_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_diffusion()
    generate_gan()
    generate_transformers()
    generate_vae()
    generate_gnn()
    generate_pinn()
    generate_snn()
    generate_autoencoder()
    generate_dbn()
    generate_rbm()
    generate_som()
    print(f"\nTotal: 11 HTML files generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
