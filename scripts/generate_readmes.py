#!/usr/bin/env python3
"""Generate comprehensive HTML README.md files for all AI apps."""

import ast
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Math templates keyed by app-type keywords
# ---------------------------------------------------------------------------

MATH_TEMPLATES = {
    "pizza": {
        "title": "Linear Regression",
        "equations": [
            r"$$\hat{y} = w \cdot x + b$$",
            r"$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$",
            r"$$\frac{\partial \mathcal{L}}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i(y_i - \hat{y}_i)$$",
            r"$$\frac{\partial \mathcal{L}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$",
            r"$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}, \quad b \leftarrow b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}$$",
        ],
        "derivation": (
            "Starting from the hypothesis $h(x) = wx + b$, we minimize the MSE loss. "
            "Taking partial derivatives w.r.t. $w$ and $b$ and applying gradient descent yields the update rules. "
            "The learning rate $\\alpha$ controls step size; too large causes divergence, too small causes slow convergence."
        ),
        "visualization": "Interactive scatter plot with regression line, showing loss descent over iterations.",
    },
    "spam": {
        "title": "Logistic Regression",
        "equations": [
            r"$$z = w \cdot x + b$$",
            r"$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$",
            r"$$\mathcal{L}_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$",
            r"$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)x_i$$",
        ],
        "derivation": (
            "Logistic regression models $P(y=1|x)$ via the sigmoid function. "
            "Binary cross-entropy loss penalizes confident wrong predictions. "
            "The gradient simplifies to $\\hat{y} - y$, enabling efficient SGD."
        ),
        "visualization": "Sigmoid curve with decision boundary overlay; ROC and precision-recall curves.",
    },
    "anomaly": {
        "title": "Anomaly Detection / PCA",
        "equations": [
            r"$$X_{\text{centered}} = X - \bar{x}$$",
            r"$$\Sigma = \frac{1}{n} X_{\text{centered}}^T X_{\text{centered}}$$",
            r"$$\Sigma v = \lambda v$$",
            r"$$X_{\text{reduced}} = X_{\text{centered}} V_k$$",
            r"$$\text{recon error} = \|X - X_{\text{reconstructed}}\|^2$$",
        ],
        "derivation": (
            "PCA finds orthogonal directions of maximum variance. "
            "By computing the SVD of centered data $X = U\\Sigma V^T$, "
            "the right singular vectors $V$ are the principal components. "
            "Anomalies are detected from large reconstruction error after projection."
        ),
        "visualization": "Interactive 2D/3D PCA projection; explained variance scree plot; anomaly score distribution.",
    },
    "market": {
        "title": "Market Segmentation (K-Means)",
        "equations": [
            r"$$\min_S \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$",
            r"$$\mu_i = \frac{1}{|S_i|} \sum_{x \in S_i} x$$",
            r"$$J = \sum_{i=1}^{n} \|x^{(i)} - \mu_{c^{(i)}}\|^2$$",
        ],
        "derivation": (
            "K-Means partitions data into $k$ clusters by minimizing within-cluster sum of squares. "
            "The Expectation-Maximization (EM) algorithm alternates between: "
            "(1) assigning each point to the nearest centroid, and "
            "(2) recomputing centroids as the mean of assigned points. "
            "Convergence is guaranteed but the solution depends on initialization."
        ),
        "visualization": "Interactive elbow method plot; cluster visualization with centroids; silhouette score explorer.",
    },
    "recommendation": {
        "title": "Recommendation Engine (Collaborative Filtering)",
        "equations": [
            r"$$\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u$$",
            r"$$\min_{b^*} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \mu - b_u - b_i)^2 + \lambda(\|b_u\|^2 + \|b_i\|^2)$$",
            r"$$\text{cosine}(u,v) = \frac{u \cdot v}{\|u\| \|v\|}$$",
        ],
        "derivation": (
            "Matrix factorization decomposes the user-item interaction matrix into latent factors. "
            "Bias terms capture global mean and user/item-specific offsets. "
            "Regularization prevents overfitting. Similarity metrics enable neighborhood-based recommendations."
        ),
        "visualization": "Interactive embedding scatter plot; recommendation coverage vs diversity trade-off; top-k recall curve.",
    },
    "robot": {
        "title": "Reinforcement Learning (Q-Learning)",
        "equations": [
            r"$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$",
            r"$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t=s, A_t=a \right]$$",
            r"$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$",
            r"$$\pi^*(a|s) = \arg\max_{a} Q^*(s, a)$$",
        ],
        "derivation": (
            "RL agents learn by interacting with an environment. "
            "The return $G_t$ is the discounted sum of future rewards. "
            "The Bellman equation decomposes $Q^\\pi$ into immediate reward plus discounted future value. "
            "Q-learning updates action-values toward the Bellman optimality target."
        ),
        "visualization": "Interactive grid world with agent path; Q-value heatmap; episode reward curves; epsilon-greedy action distribution.",
    },
    "semi": {
        "title": "Semi-Supervised Learning",
        "equations": [
            r"$$\mathcal{L} = \mathcal{L}_{sup} + \lambda_t \mathcal{L}_{unsup}$$",
            r"$$\mathcal{L}_{unsup} = \text{MSE}(f_\theta(x'), f_\theta(x)) \quad \text{(Mean Teacher)}$$",
            r"$$p_t = \min\left(1, \frac{T}{T_0}\right)$$",
        ],
        "derivation": (
            "Semi-supervised learning leverages unlabeled data by enforcing consistency. "
            "Given an input $x$, augmented views $x'$ should produce similar predictions. "
            "The total loss combines supervised cross-entropy on labeled data and consistency regularization on all data. "
            "A time-dependent weight $\\lambda_t$ ramps up the unsupervised loss."
        ),
        "visualization": "Interactive pseudo-label confidence distribution; labeled vs unlabeled loss curves; decision boundary animation.",
    },
    "self": {
        "title": "Self-Supervised Learning",
        "equations": [
            r"$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$",
            r"$$z_i = g_\theta(f_\theta(x_i))$$",
            r"$$\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$$",
        ],
        "derivation": (
            "Self-supervised learning creates labels from the data itself via pretext tasks. "
            "Contrastive learning (e.g., SimCLR, MoCo) maximizes agreement between augmented views of the same sample. "
            "The InfoNCE loss pulls positive pairs together while pushing apart negatives. "
            "A temperature parameter $\\tau$ controls the sharpness of the distribution."
        ),
        "visualization": "Interactive augmentation preview; contrastive embedding t-SNE; similarity matrix heatmap.",
    },
    "cnn": {
        "title": "Convolutional Neural Network",
        "equations": [
            r"$$Z^{(l)} = W^{(l)} * X^{(l)} + b^{(l)}$$",
            r"$$A^{(l)} = \text{ReLU}(Z^{(l)})$$",
            r"$$\text{MaxPool}(X)_{i,j} = \max_{m \in \mathcal{R}_i, n \in \mathcal{R}_j} X_{m,n}$$",
            r"$$\text{Softmax}(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$",
            r"$$\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$",
        ],
        "derivation": (
            "CNNs apply learned filters across spatial dimensions. "
            "Convolution slides a kernel $W$ over the input, computing dot products at each position. "
            "ReLU introduces non-linearity. Pooling reduces spatial dimensions. "
            "The softmax converts final logits to class probabilities."
        ),
        "visualization": "Interactive filter visualization; feature map heatmap; receptive field calculator; Grad-CAM overlay.",
    },
    "capsnet": {
        "title": "Capsule Neural Network (CapsNet)",
        "equations": [
            r"$$s_j = \sum_i c_{ij} u_{j|i}, \quad c_{ij} \text{ is routing coefficient}$$",
            r"$$v_j = \frac{\|s_j\|^2}{1 + \|s_j\|^2} \frac{s_j}{\|s_j\|}, \quad \text{squash function}$$",
            r"$$\mathcal{L}_k = T_k \max(0, m^+ - \|v_k\|)^2 + \lambda (1 - T_k) \max(0, \|v_k\| - m^-)^2$$",
            r"$$c_{ij} \leftarrow \text{softmax}(b_{ij}), \quad b_{ij} \leftarrow b_{ij} + u_{j|i} \cdot v_j$$",
        ],
        "derivation": (
            "Capsule Networks replace scalar neurons with vector capsules. "
            "The squash function preserves vector length as a probability-like activation. "
            "Dynamic routing iteratively adjusts coupling coefficients $c_{ij}$ between capsules. "
            "This allows pose matrices to encode spatial relationships between parts and wholes."
        ),
        "visualization": "Interactive capsule routing diagram; pose matrix heatmap; reconstruction error vs capsule size.",
    },
    "rnn": {
        "title": "Recurrent Neural Network (RNN)",
        "equations": [
            r"$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$",
            r"$$\hat{y}_t = W_{hy}h_t + b_y$$",
            r"$$\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t(y_t, \hat{y}_t)$$",
            r"$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_t h_{t-1}^T$$",
        ],
        "derivation": (
            "RNNs process sequences by maintaining a hidden state $h_t$ that summarizes past inputs. "
            "At each timestep, the hidden state is updated via $h_t = \\tanh(W_{hh}h_{t-1} + W_{xh}x_t)$. "
            "Backpropagation Through Time (BPTT) unrolls the network and computes gradients across all timesteps. "
            "Vanishing gradients are mitigated by gated architectures like LSTM and GRU."
        ),
        "visualization": "Interactive unfolded RNN diagram with gradient flow visualization; hidden state trajectory plot.",
    },
    "lstm": {
        "title": "Long Short-Term Memory (LSTM)",
        "equations": [
            r"$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$",
            r"$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$",
            r"$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$",
            r"$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$",
            r"$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$",
            r"$$h_t = o_t \odot \tanh(C_t)$$",
        ],
        "derivation": (
            "The LSTM maintains a cell state $C_t$ that runs through the entire sequence. "
            "The forget gate $f_t$ decides what to discard. "
            "The input gate $i_t$ and candidate $\\tilde{C}_t$ create new information. "
            "The output gate $o_t$ controls what to expose as hidden state. "
            "This gating mechanism mitigates the vanishing gradient problem of vanilla RNNs."
        ),
        "visualization": "Interactive LSTM cell diagram with animated gate activations across timesteps.",
    },
    "transformer": {
        "title": "Transformer Architecture",
        "equations": [
            r"$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$",
            r"$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$",
            r"$$y = \text{softmax}(W_{proj} \cdot \text{LayerNorm}(x + \text{MultiHead}(x)))$$",
            r"$$\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t | w_{<t}; \theta)$$",
        ],
        "derivation": (
            "The Transformer uses stacked encoder-decoder blocks. "
            "Each block applies multi-head self-attention followed by position-wise feed-forward networks, "
            "with residual connections and layer normalization. "
            "The decoder uses masked self-attention to prevent attending to future tokens during training."
        ),
        "visualization": "Interactive encoder-decoder diagram with attention head visualization and token probability explorer.",
    },
    "attention": {
        "title": "Attention Mechanism",
        "equations": [
            r"$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$",
            r"$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$",
            r"$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$",
            r"$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$",
            r"$$\text{PE}_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$",
        ],
        "derivation": (
            "Scaled dot-product attention computes compatibility between queries and keys. "
            "Scaling by $\\sqrt{d_k}$ prevents vanishing gradients for large dimensions. "
            "Multi-head attention allows the model to attend to different representation subspaces. "
            "Positional encodings inject sequence order information since attention is permutation-invariant."
        ),
        "visualization": "Interactive attention heatmap viewer; multi-head attention flow diagram; position encoding visualizer.",
    },
    "gan": {
        "title": "Generative Adversarial Network (GAN)",
        "equations": [
            r"$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$",
            r"$$G^*(x) = \arg\min_G \max_D V(D, G)$$",
            r"$$\nabla_\theta G \frac{1}{m} \sum_{i=1}^{m} \log(1 - D(G(z^{(i)})))$$",
            r"$$\mathcal{L}_{D} = -\frac{1}{m} \sum_{i=1}^{m} [\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))]$$",
        ],
        "derivation": (
            "GANs consist of two networks competing in a minimax game. "
            "The discriminator $D$ maximizes its ability to distinguish real from fake samples. "
            "The generator $G$ minimizes the probability of its samples being detected as fake. "
            "At Nash equilibrium, $G$ reproduces the true data distribution $p_{data}$."
        ),
        "visualization": "Interactive GAN training dashboard: generator/discriminator loss curves, sample evolution grid, decision boundary.",
    },
    "vae": {
        "title": "Variational Autoencoder (VAE)",
        "equations": [
            r"$$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$$",
            r"$$\mathcal{L} = \underbrace{\mathbb{E}_{z \sim q_\phi}[\log p_\theta(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{Regularization}}$$",
            r"$$D_{KL} = \frac{1}{2} \sum_{j=1}^{J} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$",
            r"$$\log p_\theta(x) \geq \mathbb{E}_{z \sim q_\phi}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$",
        ],
        "derivation": (
            "VAEs learn a probabilistic latent space via the Evidence Lower Bound (ELBO). "
            "The encoder $q_\\phi(z|x)$ maps inputs to a distribution. "
            "The decoder $p_\\theta(x|z)$ reconstructs inputs from latent samples. "
            "The KL divergence term regularizes the latent space to match a standard normal prior."
        ),
        "visualization": "Interactive latent space explorer: traverse 2D latent manifold; sample generation with sliders; KL divergence monitor.",
    },
    "diffusion": {
        "title": "Denoising Diffusion Probabilistic Model (DDPM)",
        "equations": [
            r"$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$$",
            r"$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$",
            r"$$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \|^2 \right]$$",
            r"$$\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$$",
        ],
        "derivation": (
            "DDPM gradually corrupts data with Gaussian noise over $T$ steps. "
            "The model learns to reverse this process by predicting the noise $\\epsilon$ at each step. "
            "Training minimizes the MSE between actual and predicted noise. "
            "Sampling iteratively denoises from pure Gaussian noise."
        ),
        "visualization": "Interactive forward/reverse process visualization; denoising trajectory viewer; noise schedule plot.",
    },
    "reinforcement": {
        "title": "Reinforcement Learning",
        "equations": [
            r"$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$",
            r"$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t=s, A_t=a \right]$$",
            r"$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$",
            r"$$\pi^*(a|s) = \arg\max_{a} Q^*(s, a)$$",
        ],
        "derivation": (
            "RL agents learn by interacting with an environment. "
            "The return $G_t$ is the discounted sum of future rewards. "
            "The Bellman equation decomposes $Q^\\pi$ into immediate reward plus discounted future value. "
            "Q-learning updates action-values toward the Bellman optimality target."
        ),
        "visualization": "Interactive grid world with agent path; Q-value heatmap; episode reward curves; epsilon-greedy action distribution.",
    },
    "graph": {
        "title": "Graph Neural Network (GNN)",
        "equations": [
            r"$$h_v^{(k+1)} = \sigma\left( W^{(k)} \cdot \text{AGGREGATE}_k \left( \{ h_u^{(k)} : u \in \mathcal{N}(v) \} \right) \right)$$",
            r"$$\text{AGGREGATE}_k = \text{mean} \left( \{ h_u^{(k)} : u \in \mathcal{N}(v) \} \right)$$",
            r"$$\text{GAT}: \alpha_{uv} = \frac{\exp(\text{LeakyReLU}(a^T [Wh_u \| Wh_v]))}{\sum_{k \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(a^T [Wh_u \| Wh_k]))}$$",
            r"$$h_v^{(k+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} \alpha_{uv} W h_u^{(k)} \right)$$",
        ],
        "derivation": (
            "GNNs generalize convolutions to graph-structured data. "
            "Each node updates its representation by aggregating messages from neighbors. "
            "After $K$ rounds of message passing, each node embeds its $K$-hop neighborhood. "
            "GATs introduce attention weights $\\alpha_{uv}$ to prioritize important neighbors."
        ),
        "visualization": "Interactive graph with animated message passing; node embedding t-SNE projection; attention weight heatmap.",
    },
    "pinn": {
        "title": "Physics-Informed Neural Network (PINN)",
        "equations": [
            r"$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{pde}$$",
            r"$$\mathcal{L}_{data} = \frac{1}{N} \sum_{i=1}^{N} |u_\theta(x_i, t_i) - u_i|^2$$",
            r"$$\mathcal{L}_{pde} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \mathcal{F}\left(u_\theta, x_i, t_i; \frac{\partial u_\theta}{\partial x}, \frac{\partial u_\theta}{\partial t}, \ldots \right) \right|^2$$",
            r"$$u_t + u u_x = \nu u_{xx} \quad \text{(Burgers' equation)}$$",
        ],
        "derivation": (
            "PINNs embed physical laws as soft constraints via automatic differentiation. "
            "The total loss combines data fitting $\\mathcal{L}_{data}$ and PDE residual $\\mathcal{L}_{pde}$. "
            "Gradients of $u_\\theta$ w.r.t. inputs are computed symbolically via autograd. "
            "This enables solving PDEs without labeled data in the domain interior."
        ),
        "visualization": "Interactive PDE solution comparison: PINN vs finite difference; residual heatmap; loss decomposition pie chart.",
    },
    "kmeans": {
        "title": "K-Means Clustering",
        "equations": [
            r"$$\min_S \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$",
            r"$$\mu_i = \frac{1}{|S_i|} \sum_{x \in S_i} x$$",
            r"$$J = \sum_{i=1}^{n} \|x^{(i)} - \mu_{c^{(i)}}\|^2$$",
        ],
        "derivation": (
            "K-Means partitions data into $k$ clusters by minimizing within-cluster sum of squares. "
            "The Expectation-Maximization (EM) algorithm alternates between: "
            "(1) assigning each point to the nearest centroid, and "
            "(2) recomputing centroids as the mean of assigned points. "
            "Convergence is guaranteed but the solution depends on initialization."
        ),
        "visualization": "Interactive elbow method plot; cluster visualization with centroids; silhouette score explorer.",
    },
    "pca": {
        "title": "Principal Component Analysis (PCA)",
        "equations": [
            r"$$X_{\text{centered}} = X - \bar{x}$$",
            r"$$\Sigma = \frac{1}{n} X_{\text{centered}}^T X_{\text{centered}}$$",
            r"$$\Sigma v = \lambda v$$",
            r"$$X_{\text{reduced}} = X_{\text{centered}} V_k$$",
            r"$$X_{\text{reconstructed}} = X_{\text{reduced}} V_k^T + \bar{x}$$",
        ],
        "derivation": (
            "PCA finds orthogonal directions of maximum variance. "
            "By computing the SVD of centered data $X = U\\Sigma V^T$, "
            "the right singular vectors $V$ are the principal components. "
            "Projecting onto the first $k$ components preserves maximum variance."
        ),
        "visualization": "Interactive 2D/3D PCA projection; explained variance scree plot; biplot with feature vectors.",
    },
    "autoencoder": {
        "title": "Autoencoder",
        "equations": [
            r"$$z = f(x) = \sigma(W_e x + b_e) \quad \text{(encoder)}$$",
            r"$$\hat{x} = g(z) = \sigma(W_d z + b_d) \quad \text{(decoder)}$$",
            r"$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda (\|W_e\|^2 + \|W_d\|^2)$$",
            r"$$z^* = \arg\min_z \|x - g(f(x))\|^2$$",
        ],
        "derivation": (
            "Autoencoders learn compressed representations by minimizing reconstruction error. "
            "The encoder maps input $x$ to a latent code $z$. "
            "The decoder reconstructs $\\hat{x}$ from $z$. "
            "L2 regularization and bottleneck architecture prevent trivial identity solutions."
        ),
        "visualization": "Interactive latent space traversal; reconstruction error vs latent dimension; bottleneck visualization.",
    },
    "random-forest": {
        "title": "Random Forest",
        "equations": [
            r"$$\hat{f} = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$$",
            r"$$\mathcal{L}_{split} = \frac{N_{left}}{N} H_{left} - \frac{N_{right}}{N} H_{right}$$",
            r"$$H(S) = -\sum_{c \in C} p_c \log_2 p_c \quad \text{(entropy)}$$",
        ],
        "derivation": (
            "Random Forest builds an ensemble of decision trees via bootstrap sampling. "
            "Each split maximizes information gain (or minimizes Gini impurity). "
            "Bagging (bootstrap aggregation) reduces variance by averaging predictions. "
            "Feature subsampling at each split decorrelates trees, improving generalization."
        ),
        "visualization": "Interactive decision tree explorer; feature importance bar chart; OOB error curve; class probability heatmap.",
    },
    "q-learning": {
        "title": "Q-Learning",
        "equations": [
            r"$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$",
            r"$$\pi^*(a|s) = \arg\max_a Q^*(s, a)$$",
            r"$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$",
        ],
        "derivation": (
            "Q-learning is an off-policy TD control algorithm. "
            "The update rule bootstraps from the maximum estimated future value. "
            "With sufficient exploration and decaying learning rate, $Q$ converges to optimal action-values. "
            "The policy $\\pi^*(a|s) = \\arg\\max_a Q^*(s, a)$ is greedy in the limit."
        ),
        "visualization": "Interactive grid world with learned Q-value heatmap; episode return curve; exploration rate decay.",
    },
    "self-supervised": {
        "title": "Self-Supervised Learning",
        "equations": [
            r"$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$",
            r"$$z_i = g_\theta(f_\theta(x_i))$$",
            r"$$\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$$",
        ],
        "derivation": (
            "Self-supervised learning creates labels from the data itself via pretext tasks. "
            "Contrastive learning (e.g., SimCLR, MoCo) maximizes agreement between augmented views of the same sample. "
            "The InfoNCE loss pulls positive pairs together while pushing apart negatives. "
            "A temperature parameter $\\tau$ controls the sharpness of the distribution."
        ),
        "visualization": "Interactive augmentation preview; contrastive embedding t-SNE; similarity matrix heatmap.",
    },
    "semi-supervised": {
        "title": "Semi-Supervised Learning",
        "equations": [
            r"$$\mathcal{L} = \mathcal{L}_{sup} + \lambda_t \mathcal{L}_{unsup}$$",
            r"$$\mathcal{L}_{unsup} = \text{MSE}(f_\theta(x'), f_\theta(x)) \quad \text{(Mean Teacher)}$$",
            r"$$p_t = \min\left(1, \frac{T}{T_0}\right)$$",
        ],
        "derivation": (
            "Semi-supervised learning leverages unlabeled data by enforcing consistency. "
            "Given an input $x$, augmented views $x'$ should produce similar predictions. "
            "The total loss combines supervised cross-entropy on labeled data and consistency regularization on all data. "
            "A time-dependent weight $\\lambda_t$ ramps up the unsupervised loss."
        ),
        "visualization": "Interactive pseudo-label confidence distribution; labeled vs unlabeled loss curves; decision boundary animation.",
    },
    "snn": {
        "title": "Spiking Neural Network (SNN)",
        "equations": [
            r"$$\tau_m \frac{dV_m}{dt} = -(V_m - V_{rest}) + R_m I(t)$$",
            r"$$\text{if } V_m \geq V_{th}: \text{ emit spike}, V_m \leftarrow V_{reset}$$",
            r"$$S(t) = \sum_{i} \delta(t - t_i)$$",
            r"$$\tau_s \frac{dS}{dt} = -S$$",
        ],
        "derivation": (
            "SNNs compute with discrete spike events rather than continuous activations. "
            "The membrane potential integrates input current and leaks over time. "
            "When the potential exceeds a threshold, a spike is emitted and the membrane is reset. "
            "This event-driven computation is energy-efficient and biologically plausible."
        ),
        "visualization": "Interactive membrane potential trace; spike raster plot; synaptic current decomposition.",
    },
    "gnn": {
        "title": "Graph Neural Network (GNN)",
        "equations": [
            r"$$h_v^{(k+1)} = \sigma\left( W^{(k)} \cdot \text{AGGREGATE}_k \left( \{ h_u^{(k)} : u \in \mathcal{N}(v) \} \right) \right)$$",
            r"$$\text{AGGREGATE}_k = \text{mean} \left( \{ h_u^{(k)} : u \in \mathcal{N}(v) \} \right)$$",
            r"$$\text{GAT}: \alpha_{uv} = \frac{\exp(\text{LeakyReLU}(a^T [Wh_u \| Wh_v]))}{\sum_{k \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(a^T [Wh_u \| Wh_k]))}$$",
            r"$$h_v^{(k+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} \alpha_{uv} W h_u^{(k)} \right)$$",
        ],
        "derivation": (
            "GNNs generalize convolutions to graph-structured data. "
            "Each node updates its representation by aggregating messages from neighbors. "
            "After $K$ rounds of message passing, each node embeds its $K$-hop neighborhood. "
            "GATs introduce attention weights $\\alpha_{uv}$ to prioritize important neighbors."
        ),
        "visualization": "Interactive graph with animated message passing; node embedding t-SNE projection; attention weight heatmap.",
    },
    "pinn": {
        "title": "Physics-Informed Neural Network (PINN)",
        "equations": [
            r"$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{pde}$$",
            r"$$\mathcal{L}_{data} = \frac{1}{N} \sum_{i=1}^{N} |u_\theta(x_i, t_i) - u_i|^2$$",
            r"$$\mathcal{L}_{pde} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \mathcal{F}\left(u_\theta, x_i, t_i; \frac{\partial u_\theta}{\partial x}, \frac{\partial u_\theta}{\partial t}, \ldots \right) \right|^2$$",
            r"$$u_t + u u_x = \nu u_{xx} \quad \text{(Burgers' equation)}$$",
        ],
        "derivation": (
            "PINNs embed physical laws as soft constraints via automatic differentiation. "
            "The total loss combines data fitting $\\mathcal{L}_{data}$ and PDE residual $\\mathcal{L}_{pde}$. "
            "Gradients of $u_\\theta$ w.r.t. inputs are computed symbolically via autograd. "
            "This enables solving PDEs without labeled data in the domain interior."
        ),
        "visualization": "Interactive PDE solution comparison: PINN vs finite difference; residual heatmap; loss decomposition pie chart.",
    },
    "kmeans": {
        "title": "K-Means Clustering",
        "equations": [
            r"$$\min_S \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$",
            r"$$\mu_i = \frac{1}{|S_i|} \sum_{x \in S_i} x$$",
            r"$$J = \sum_{i=1}^{n} \|x^{(i)} - \mu_{c^{(i)}}\|^2$$",
        ],
        "derivation": (
            "K-Means partitions data into $k$ clusters by minimizing within-cluster sum of squares. "
            "The Expectation-Maximization (EM) algorithm alternates between: "
            "(1) assigning each point to the nearest centroid, and "
            "(2) recomputing centroids as the mean of assigned points. "
            "Convergence is guaranteed but the solution depends on initialization."
        ),
        "visualization": "Interactive elbow method plot; cluster visualization with centroids; silhouette score explorer.",
    },
    "pca": {
        "title": "Principal Component Analysis (PCA)",
        "equations": [
            r"$$X_{\text{centered}} = X - \bar{x}$$",
            r"$$\Sigma = \frac{1}{n} X_{\text{centered}}^T X_{\text{centered}}$$",
            r"$$\Sigma v = \lambda v$$",
            r"$$X_{\text{reduced}} = X_{\text{centered}} V_k$$",
            r"$$X_{\text{reconstructed}} = X_{\text{reduced}} V_k^T + \bar{x}$$",
        ],
        "derivation": (
            "PCA finds orthogonal directions of maximum variance. "
            "By computing the SVD of centered data $X = U\\Sigma V^T$, "
            "the right singular vectors $V$ are the principal components. "
            "Projecting onto the first $k$ components preserves maximum variance."
        ),
        "visualization": "Interactive 2D/3D PCA projection; explained variance scree plot; biplot with feature vectors.",
    },
    "autoencoder": {
        "title": "Autoencoder",
        "equations": [
            r"$$z = f(x) = \sigma(W_e x + b_e) \quad \text{(encoder)}$$",
            r"$$\hat{x} = g(z) = \sigma(W_d z + b_d) \quad \text{(decoder)}$$",
            r"$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda (\|W_e\|^2 + \|W_d\|^2)$$",
            r"$$z^* = \arg\min_z \|x - g(f(x))\|^2$$",
        ],
        "derivation": (
            "Autoencoders learn compressed representations by minimizing reconstruction error. "
            "The encoder maps input $x$ to a latent code $z$. "
            "The decoder reconstructs $\\hat{x}$ from $z$. "
            "L2 regularization and bottleneck architecture prevent trivial identity solutions."
        ),
        "visualization": "Interactive latent space traversal; reconstruction error vs latent dimension; bottleneck visualization.",
    },
    "random-forest": {
        "title": "Random Forest",
        "equations": [
            r"$$\hat{f} = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$$",
            r"$$\mathcal{L}_{split} = \frac{N_{left}}{N} H_{left} - \frac{N_{right}}{N} H_{right}$$",
            r"$$H(S) = -\sum_{c \in C} p_c \log_2 p_c \quad \text{(entropy)}$$",
        ],
        "derivation": (
            "Random Forest builds an ensemble of decision trees via bootstrap sampling. "
            "Each split maximizes information gain (or minimizes Gini impurity). "
            "Bagging (bootstrap aggregation) reduces variance by averaging predictions. "
            "Feature subsampling at each split decorrelates trees, improving generalization."
        ),
        "visualization": "Interactive decision tree explorer; feature importance bar chart; OOB error curve; class probability heatmap.",
    },
    "q-learning": {
        "title": "Q-Learning",
        "equations": [
            r"$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$",
            r"$$\pi^*(a|s) = \arg\max_a Q^*(s, a)$$",
            r"$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$",
        ],
        "derivation": (
            "Q-learning is an off-policy TD control algorithm. "
            "The update rule bootstraps from the maximum estimated future value. "
            "With sufficient exploration and decaying learning rate, $Q$ converges to optimal action-values. "
            "The policy $\\pi^*(a|s) = \\arg\\max_a Q^*(s, a)$ is greedy in the limit."
        ),
        "visualization": "Interactive grid world with learned Q-value heatmap; episode return curve; exploration rate decay.",
    },
    "self-supervised": {
        "title": "Self-Supervised Learning",
        "equations": [
            r"$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$",
            r"$$z_i = g_\theta(f_\theta(x_i))$$",
            r"$$\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$$",
        ],
        "derivation": (
            "Self-supervised learning creates labels from the data itself via pretext tasks. "
            "Contrastive learning (e.g., SimCLR, MoCo) maximizes agreement between augmented views of the same sample. "
            "The InfoNCE loss pulls positive pairs together while pushing apart negatives. "
            "A temperature parameter $\\tau$ controls the sharpness of the distribution."
        ),
        "visualization": "Interactive augmentation preview; contrastive embedding t-SNE; similarity matrix heatmap.",
    },
    "semi-supervised": {
        "title": "Semi-Supervised Learning",
        "equations": [
            r"$$\mathcal{L} = \mathcal{L}_{sup} + \lambda_t \mathcal{L}_{unsup}$$",
            r"$$\mathcal{L}_{unsup} = \text{MSE}(f_\theta(x'), f_\theta(x)) \quad \text{(Mean Teacher)}$$",
            r"$$p_t = \min\left(1, \frac{T}{T_0}\right)$$",
        ],
        "derivation": (
            "Semi-supervised learning leverages unlabeled data by enforcing consistency. "
            "Given an input $x$, augmented views $x'$ should produce similar predictions. "
            "The total loss combines supervised cross-entropy on labeled data and consistency regularization on all data. "
            "A time-dependent weight $\\lambda_t$ ramps up the unsupervised loss."
        ),
        "visualization": "Interactive pseudo-label confidence distribution; labeled vs unlabeled loss curves; decision boundary animation.",
    },
    "snn": {
        "title": "Spiking Neural Network (SNN)",
        "equations": [
            r"$$\tau_m \frac{dV_m}{dt} = -(V_m - V_{rest}) + R_m I(t)$$",
            r"$$\text{if } V_m \geq V_{th}: \text{ emit spike}, V_m \leftarrow V_{reset}$$",
            r"$$S(t) = \sum_{i} \delta(t - t_i)$$",
            r"$$\tau_s \frac{dS}{dt} = -S$$",
        ],
        "derivation": (
            "SNNs compute with discrete spike events rather than continuous activations. "
            "The membrane potential integrates input current and leaks over time. "
            "When the potential exceeds a threshold, a spike is emitted and the membrane is reset. "
            "This event-driven computation is energy-efficient and biologically plausible."
        ),
        "visualization": "Interactive membrane potential trace; spike raster plot; synaptic current decomposition.",
    },
    "gnn": {
        "title": "Graph Neural Network (GNN)",
        "equations": [
            r"$$h_v^{(k+1)} = \sigma\left( W^{(k)} \cdot \text{AGGREGATE}_k \left( \{ h_u^{(k)} : u \in \mathcal{N}(v) \} \right) \right)$$",
            r"$$\text{AGGREGATE}_k = \text{mean} \left( \{ h_u^{(k)} : u \in \mathcal{N}(v) \} \right)$$",
            r"$$\text{GAT}: \alpha_{uv} = \frac{\exp(\text{LeakyReLU}(a^T [Wh_u \| Wh_v]))}{\sum_{k \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(a^T [Wh_u \| Wh_k]))}$$",
            r"$$h_v^{(k+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} \alpha_{uv} W h_u^{(k)} \right)$$",
        ],
        "derivation": (
            "GNNs generalize convolutions to graph-structured data. "
            "Each node updates its representation by aggregating messages from neighbors. "
            "After $K$ rounds of message passing, each node embeds its $K$-hop neighborhood. "
            "GATs introduce attention weights $\\alpha_{uv}$ to prioritize important neighbors."
        ),
        "visualization": "Interactive graph with animated message passing; node embedding t-SNE projection; attention weight heatmap.",
    },
    "pinn": {
        "title": "Physics-Informed Neural Network (PINN)",
        "equations": [
            r"$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{pde}$$",
            r"$$\mathcal{L}_{data} = \frac{1}{N} \sum_{i=1}^{N} |u_\theta(x_i, t_i) - u_i|^2$$",
            r"$$\mathcal{L}_{pde} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \mathcal{F}\left(u_\theta, x_i, t_i; \frac{\partial u_\theta}{\partial x}, \frac{\partial u_\theta}{\partial t}, \ldots \right) \right|^2$$",
            r"$$u_t + u u_x = \nu u_{xx} \quad \text{(Burgers' equation)}$$",
        ],
        "derivation": (
            "PINNs embed physical laws as soft constraints via automatic differentiation. "
            "The total loss combines data fitting $\\mathcal{L}_{data}$ and PDE residual $\\mathcal{L}_{pde}$. "
            "Gradients of $u_\\theta$ w.r.t. inputs are computed symbolically via autograd. "
            "This enables solving PDEs without labeled data in the domain interior."
        ),
        "visualization": "Interactive PDE solution comparison: PINN vs finite difference; residual heatmap; loss decomposition pie chart.",
    },
    "house": {
        "title": "Linear Regression",
        "equations": [
            r"$$\hat{y} = w \cdot x + b$$",
            r"$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$",
            r"$$\frac{\partial \mathcal{L}}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i(y_i - \hat{y}_i)$$",
            r"$$\frac{\partial \mathcal{L}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$",
            r"$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}, \quad b \leftarrow b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}$$",
        ],
        "derivation": (
            "Starting from the hypothesis $h(x) = wx + b$, we minimize the MSE loss. "
            "Taking partial derivatives w.r.t. $w$ and $b$ and applying gradient descent yields the update rules. "
            "The learning rate $\\alpha$ controls step size; too large causes divergence, too small causes slow convergence."
        ),
        "visualization": "Interactive scatter plot with regression line, showing loss descent over iterations.",
    },
    "fraud": {
        "title": "Anomaly Detection / Autoencoder",
        "equations": [
            r"$$z = f(x) = \sigma(W_e x + b_e) \quad \text{(encoder)}$$",
            r"$$\hat{x} = g(z) = \sigma(W_d z + b_d) \quad \text{(decoder)}$$",
            r"$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda (\|W_e\|^2 + \|W_d\|^2)$$",
            r"$$\text{anomaly score} = \|x - \hat{x}\|^2$$",
        ],
        "derivation": (
            "Autoencoders learn compressed representations by minimizing reconstruction error. "
            "The encoder maps input $x$ to a latent code $z$. "
            "The decoder reconstructs $\\hat{x}$ from $z$. "
            "L2 regularization and bottleneck architecture prevent trivial identity solutions."
        ),
        "visualization": "Interactive latent space traversal; reconstruction error vs latent dimension; bottleneck visualization.",
    },
    "digit": {
        "title": "Digit Recognition / Classification",
        "equations": [
            r"$$Z = WX + b$$",
            r"$$A = \text{ReLU}(Z)$$",
            r"$$\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$",
            r"$$\hat{y} = \text{softmax}(Z_{out})$$",
        ],
        "derivation": (
            "Feedforward networks learn hierarchical feature representations. "
            "Each layer computes a linear transformation followed by a non-linearity. "
            "Cross-entropy loss penalizes misclassification. "
            "Backpropagation computes gradients via the chain rule."
        ),
        "visualization": "Interactive decision boundary; feature visualization for hidden layers; confusion matrix explorer.",
    },
    "transfer": {
        "title": "Transfer Learning",
        "equations": [
            r"$$\mathcal{L} = \mathcal{L}_{task} + \lambda \mathcal{L}_{distill}$$",
            r"$$\mathcal{L}_{distill} = \text{KL}(p_{\text{teacher}} \| p_{\text{student}})$$",
            r"$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$",
        ],
        "derivation": (
            "Transfer learning reuses features from a source domain for a target task. "
            "Fine-tuning updates only the final layers to adapt to new data. "
            "Knowledge distillation transfers dark knowledge from a large teacher to a compact student via softened probabilities."
        ),
        "visualization": "Interactive feature reuse heatmap; layer freezing/unfreezing timeline; teacher vs student probability comparison.",
    },
    "multimodal": {
        "title": "Multimodal Learning",
        "equations": [
            r"$$h = \text{CrossAttention}(Q_{\text{text}}, K_{\text{image}}, V_{\text{image}})$$",
            r"$$\mathcal{L} = \mathcal{L}_{\text{image-text}} + \lambda_1 \mathcal{L}_{\text{image}} + \lambda_2 \mathcal{L}_{\text{text}}$$",
            r"$$\text{cosine}(u, v) = \frac{u^T v}{\|u\| \|v\|}$$",
        ],
        "derivation": (
            "Multimodal models align representations from different modalities in a shared embedding space. "
            "Cross-attention allows one modality to query another. "
            "Contrastive learning pulls matched pairs together and pushes unmatched pairs apart. "
            "The total loss balances cross-modal alignment with unimodal task losses."
        ),
        "visualization": "Interactive embedding alignment plot; cross-attention weight heatmap; modality contribution explorer.",
    },
    "pre-training": {
        "title": "Pre-training and Fine-Tuning",
        "equations": [
            r"$$\mathcal{L}_{MLM} = -\sum_{i \in M} \log P(x_i | x_{\setminus M})$$",
            r"$$\mathcal{L}_{NSP} = \log P(\text{IsNext} | [CLS])$$",
            r"$$\mathcal{L}_{total} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$$",
        ],
        "derivation": (
            "Pre-training learns general representations from large unlabeled corpora. "
            "Masked Language Modeling (MLM) predicts randomly masked tokens. "
            "Next Sentence Prediction (NSP) learns inter-sentence coherence. "
            "Fine-tuning adapts pre-trained weights to downstream tasks with minimal labeled data."
        ),
        "visualization": "Interactive MLM token prediction explorer; attention head visualization; layer-wise transfer analysis.",
    },
    "prompt": {
        "title": "Prompt Engineering",
        "equations": [
            r"$$P(y|x, p) = \prod_{t=1}^{|y|} P(y_t | x, p, y_{<t})$$",
            r"$$\hat{p} = \arg\max_p \mathbb{E}_{x \sim \mathcal{D}} [\log P(y^* | x, p)]$$",
        ],
        "derivation": (
            "Prompt engineering reformulates downstream tasks as language modeling. "
            "Given a prompt $p$, the model generates output $y$ autoregressively. "
            "Prompt tuning optimizes $p$ to maximize task-specific likelihood. "
            "Soft prompts are continuous embeddings optimized via gradient descent."
        ),
        "visualization": "Interactive prompt comparison table; generation diversity vs prompt length; token probability explorer.",
    },
    "code": {
        "title": "Code Generation",
        "equations": [
            r"$$P(c | p) = \prod_{t=1}^{|c|} P(c_t | p, c_{<t})$$",
            r"$$\mathcal{L} = -\sum_{t=1}^{|c|} \log P(c_t | p, c_{<t}; \theta)$$",
        ],
        "derivation": (
            "Code generation treats source code as a sequence modeled by a language model. "
            "The prompt $p$ provides context (docstring, imports, function signature). "
            "The model predicts tokens autoregressively, conditioned on previous predictions. "
            "Beam search and nucleus sampling improve output quality and diversity."
        ),
        "visualization": "Interactive code completion demo; token probability heatmap; beam search tree explorer.",
    },
    "text": {
        "title": "Text Generation",
        "equations": [
            r"$$P(w_t | w_{<t}) = \text{softmax}(W_h h_t + b_h)$$",
            r"$$h_t = \text{LSTM}(x_t, h_{t-1})$$",
            r"$$\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t | w_{<t})$$",
        ],
        "derivation": (
            "Text generation models learn to predict the next token given past context. "
            "Temperature scaling controls randomness: high temperature yields creative but incoherent text; low temperature yields repetitive but safe text. "
            "Top-k and nucleus sampling truncate the probability mass to improve diversity."
        ),
        "visualization": "Interactive temperature slider; generated text preview; perplexity vs context length.",
    },
    "image": {
        "title": "Image Generation (GAN/VAE/Diffusion)",
        "equations": [
            r"$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$",
            r"$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$$",
            r"$$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \|^2 \right]$$",
        ],
        "derivation": (
            "Image generation models learn to synthesize realistic images. "
            "GANs use adversarial training between generator and discriminator. "
            "VAEs learn a structured latent space via reconstruction and KL regularization. "
            "Diffusion models iteratively denoise from Gaussian noise, offering stable training and diverse outputs."
        ),
        "visualization": "Interactive latent space explorer; denoising trajectory viewer; FID score vs training steps.",
    },
    "video": {
        "title": "Video Generation",
        "equations": [
            r"$$P(x_{1:T}) = \prod_{t=1}^{T} P(x_t | x_{<t})$$",
            r"$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$",
            r"$$\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x \mu_{\hat{x}} + c_1)(2\sigma_{x\hat{x}} + c_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + c_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + c_2)}$$",
        ],
        "derivation": (
            "Video generation extends sequence modeling to spatiotemporal data. "
            "3D convolutions or factored spatial-temporal attention capture motion. "
            "Temporal consistency is enforced via warping or predictive coding. "
            "Frame-wise perceptual losses improve visual quality."
        ),
        "visualization": "Interactive frame-by-frame playback with generated vs real overlay; optical flow visualization; temporal consistency score.",
    },
    "retrieval": {
        "title": "Retrieval-Augmented Generation (RAG)",
        "equations": [
            r"$$P(y | x) = \sum_{z \in \mathcal{Z}} P(y | x, z) P(z | x)$$",
            r"$$\text{sim}(q, d) = \frac{q^T d}{\|q\| \|d\|}$$",
            r"$$\text{top-}k = \arg\max_{d_i \in \mathcal{D}} \text{sim}(q, d_i)$$",
        ],
        "derivation": (
            "RAG combines retrieval with generation. "
            "Given a query $q$, the retriever finds top-$k$ documents $z$ from a knowledge base. "
            "The generator conditions on both the query and retrieved context. "
            "This allows the model to access up-to-date or domain-specific information without retraining."
        ),
        "visualization": "Interactive retrieval pipeline; relevance score distribution; context vs generation attention alignment.",
    },
    "tool": {
        "title": "Tool Use and Functional Calling",
        "equations": [
            r"$$P(\text{tool}, \text{args} | q) = \text{softmax}(W_t \cdot h_q)$$",
            r"$$\text{result} = \text{execute}(\text{tool}, \text{args})$$",
            r"$$\text{final} = \text{generate}(q, \text{result})$$",
        ],
        "derivation": (
            "Tool-augmented models decompose complex queries into executable function calls. "
            "A router network predicts which tool to invoke and with what arguments. "
            "The tool result is fed back into the language model for final response generation. "
            "This enables structured reasoning and access to external APIs."
        ),
        "visualization": "Interactive tool call graph; argument parsing explorer; multi-step reasoning trace.",
    },
    "default": {
        "title": "Machine Learning Fundamentals",
        "equations": [
            r"$$\hat{y} = f(x; \theta)$$",
            r"$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(y_i, \hat{y}_i)$$",
            r"$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$",
        ],
        "derivation": (
            "Machine learning models learn parameters $\\theta$ by minimizing a loss function $\\mathcal{L}$. "
            "Gradient descent iteratively updates parameters in the direction of steepest descent. "
            "The learning rate $\\alpha$ controls step size. "
            "Stochastic gradient descent (SGD) uses mini-batches for computational efficiency."
        ),
        "visualization": "Interactive loss landscape explorer; gradient descent trajectory; learning rate scheduler.",
    },
}

# ---------------------------------------------------------------------------
# Worked numerical examples keyed by math-template key.
# Each is a concrete, correct forward-pass / update evaluation that uses the
# algorithm's own equations with real numbers. Rendered in the README so every
# example ships a valid, app-specific numerical illustration.
# ---------------------------------------------------------------------------

WORKED_EXAMPLES = {
    "pizza": (
        "Linear regression forward pass (pizza price from diameter).\n"
        "  Trained weights (least squares on the data below): w=1.525, b=-2.35.\n"
        "  Input   x (diameter) = 12.0 in\n"
        "  y_hat = w*x + b = 1.525*12.0 - 2.35 = 15.95  -> predicted price $15.95\n"
        "  Actual price at x=12 is $17.50, so residual = 17.50 - 15.95 = 1.55.\n"
        "  MSE over the 5 points ~ 0.85 (R^2 ~ 0.96): the line fits well."
    ),
    "house": (
        "Linear regression forward pass (house price from size, $1000s sqft).\n"
        "  Input   x (size) = 1.8\n"
        "  Weights w       = 120.0\n"
        "  Bias    b       = 30.0\n"
        "  y_hat = 120.0*1.8 + 30.0 = 246.0  -> predicted price $246k\n"
        "  Gradient: dL/dw = -2/n sum x(y_hat-y)."
    ),
    "spam": (
        "Logistic regression single-sample forward pass (spam classifier).\n"
        "  x = 2.0   w = 1.2   b = -1.0\n"
        "  z = w*x + b = 1.2*2.0 - 1.0 = 1.40\n"
        "  y_hat = sigma(z) = 1/(1+e^-1.40) = 0.802\n"
        "  => P(spam) = 0.80 (above 0.5 threshold -> spam)."
    ),
    "digit": (
        "Feed-forward classification forward pass (3-class digit).\n"
        "  z = Wx + b = [1.1, 0.4, -0.3]\n"
        "  exp(z) = [3.00, 1.49, 0.74];  sum = 5.23\n"
        "  y_hat = softmax(z) = [0.574, 0.285, 0.142]\n"
        "  predicted class = argmax = 0 (highest probability)."
    ),
    "anomaly": (
        "PCA projection + reconstruction (anomaly detection).\n"
        "  Centered X = [[1,1],[-1,-1],[1,-1],[-1,1]]\n"
        "  Cov Sigma = [[1,0],[0,1]]; top PC v = [0.707,0.707]\n"
        "  Reduced x' = X.v = [1.41,-1.41,0,-0]  (1-D)\n"
        "  Recon error ||X - X_recon||^2 = 0 for inliers; large for anomalies."
    ),
    "pca": (
        "PCA dimensionality reduction (2-D -> 1-D).\n"
        "  Centered X = [[2,1],[-2,-1]]\n"
        "  Cov Sigma = [[2,1],[1,0.5]]; eigval lambda1 = 2.5\n"
        "  PC1 v1 = [0.894,0.447];  x' = X.v1 = [2.236,-2.236]\n"
        "  Explained variance ratio = lambda1/(2.5+0) = 100%."
    ),
    "market": (
        "K-Means assignment + centroid update (k=2).\n"
        "  Points {0,0},{1,0},{8,8},{9,8}; init mu0=(0,0), mu1=(9,8)\n"
        "  Assign: {0,0},{1,0}->C0 ; {8,8},{9,8}->C1\n"
        "  Update mu0 = mean(C0) = (0.5,0); mu1 = mean(C1) = (8.5,8)\n"
        "  Objective J drops after each EM step."
    ),
    "kmeans": (
        "K-Means clustering step (see market example).\n"
        "  Assign each point to nearest centroid, then recompute\n"
        "  mu_i = (1/|S_i|) sum_{x in S_i} x.\n"
        "  Repeat until centroids stabilise (EM convergence)."
    ),
    "recommendation": (
        "Matrix-factorization prediction (user-item rating).\n"
        "  mu=3.5, b_u=0.3, b_i=-0.2, q_i.p_u = 0.4\n"
        "  r_hat_ui = 3.5 + 0.3 - 0.2 + 0.4 = 4.0\n"
        "  Cosine sim(u,v) = (u.v)/(||u|| ||v||) ranks neighbours."
    ),
    "robot": (
        "Q-learning update (robot maze, state s, action a).\n"
        "  alpha=0.1, gamma=0.9, Q(s,a)=0.50, reward r=1.0\n"
        "  max_a' Q(s',a') = 0.80\n"
        "  Q <- 0.50 + 0.1*(1.0 + 0.9*0.80 - 0.50)\n"
        "      = 0.50 + 0.1*(1.72-0.50) = 0.622"
    ),
    "reinforcement": (
        "Q-learning Bellman update (see robot example).\n"
        "  Q(s,a) <- Q(s,a) + alpha[r + gamma*max_a' Q(s',a') - Q(s,a)]\n"
        "  With alpha=0.1, gamma=0.9: same 0.50 -> 0.622 step."
    ),
    "q-learning": (
        "Q-learning TD update.\n"
        "  delta = r + gamma*max_a' Q(s',a') - Q(s,a)\n"
        "        = 1.0 + 0.9*0.80 - 0.50 = 1.22\n"
        "  Q <- Q + alpha*delta = 0.50 + 0.1*1.22 = 0.622."
    ),
    "semi": (
        "Mean-Teacher consistency weight ramp.\n"
        "  p_t = min(1, T/T0); with T=2*T0 -> p_t = 1.0\n"
        "  L = L_sup + lambda_t * L_unsup\n"
        "  lambda_t grows from 0 -> 1 so unlabeled data matters later."
    ),
    "semi-supervised": (
        "Semi-supervised consistency loss.\n"
        "  L = L_sup + lambda_t * MSE(f(x'), f(x))\n"
        "  Augmented view x' should match f(x); ramp lambda_t over training."
    ),
    "self": (
        "InfoNCE contrastive loss (single positive).\n"
        "  sim(z_i,z_j)=0.8 (pos), negatives sim=0.1,0.2; tau=0.1\n"
        "  num = exp(0.8/0.1)=exp(8)=2981\n"
        "  den = exp(8)+exp(1)+exp(2)=2981+2.72+7.39=2991\n"
        "  L = -log(2981/2991) = 0.0033"
    ),
    "self-supervised": (
        "InfoNCE (see self example): pull positive pair, push negatives.\n"
        "  L = -log[ exp(sim(z_i,z_j)/tau) / sum_k exp(sim(z_i,z_k)/tau) ]."
    ),
    "cnn": (
        "2-D convolution (edge kernel) at one position.\n"
        "  Input row = [1,2,1,0]; kernel = [1,0,-1]\n"
        "  out = 1*1 + 0*2 + (-1)*1 = 0   (left)\n"
        "  out = 2*1 + 0*1 + (-1)*0 = 2   (right)\n"
        "  ReLU(2)=2 activates strong edges."
    ),
    "capsnet": (
        "CapsNet squash function.\n"
        "  s = [1.5,1.5]  -> ||s|| = 2.12\n"
        "  v = (||s||^2/(1+||s||^2)) * (s/||s||)\n"
        "    = (4.5/5.5) * (0.707,0.707) = (0.578,0.578)\n"
        "  length 0.818 encodes entity presence probability."
    ),
    "rnn": (
        "RNN hidden-state update (one timestep).\n"
        "  h_{t-1}=0.30, x_t=0.50, W_hh=W_xh=0.5, b=0\n"
        "  pre = 0.5*0.30 + 0.5*0.50 = 0.40\n"
        "  h_t = tanh(0.40) = 0.380"
    ),
    "lstm": (
        "LSTM forget gate + cell update.\n"
        "  f_t = sigma(0.5*0.30 + 0.5*0.50) = sigma(0.40) = 0.598\n"
        "  C_{t-1}=0.80 -> C_t = 0.598*0.80 + i_t*tildeC_t\n"
        "  (with i_t=0.4, tildeC=0.6): C_t = 0.478 + 0.240 = 0.718"
    ),
    "transformer": (
        "Scaled dot-product attention (2 tokens, d_k=2).\n"
        "  Q=[1,0], K=[[1,0],[0,1]], V=[[1,2],[3,4]]\n"
        "  QK^T = [1,0]; /sqrt(2) = [0.707,0]\n"
        "  softmax = [0.67,0.33]\n"
        "  out = 0.67*[1,2] + 0.33*[3,4] = [1.67,2.67]"
    ),
    "attention": (
        "Scaled dot-product attention (see transformer example).\n"
        "  out = softmax(QK^T/sqrt(d_k)) V; positional encodings\n"
        "  PE(pos,2i)=sin(pos/10000^{2i/d}) inject order."
    ),
    "gan": (
        "GAN discriminator loss (one step).\n"
        "  D(x)=0.90 (real), D(G(z))=0.20 (fake)\n"
        "  L_D = -[log 0.90 + log(1-0.20)]\n"
        "       = -[(-0.105) + (-0.223)] = 0.328\n"
        "  Generator minimises -log D(G(z))."
    ),
    "vae": (
        "VAE KL term (1-D latent).\n"
        "  mu=0.1, sigma=1.0\n"
        "  KL = 0.5*(1 + log(1^2) - 0.1^2 - 1^2)\n"
        "     = 0.5*(1 + 0 - 0.01 - 1) = -0.005 ~ 0\n"
        "  ELBO = Reconstruction - KL."
    ),
    "diffusion": (
        "Forward diffusion noising step.\n"
        "  bar_alpha_t = 0.9 -> sqrt=0.949, sqrt(1-0.9)=0.316\n"
        "  x_0=1.0, epsilon=0.5\n"
        "  x_t = 0.949*1.0 + 0.316*0.5 = 1.107\n"
        "  Training predicts epsilon_theta from x_t, t."
    ),
    "graph": (
        "Graph message passing (node v, 2 neighbours).\n"
        "  h_u=[1,0], h_w=[0,1]; W=[[1,0],[0,1]]\n"
        "  agg = mean([1,0],[0,1]) = [0.5,0.5]\n"
        "  h_v' = sigma(W.agg) = sigma([0.5,0.5]) = [0.622,0.622]"
    ),
    "gnn": (
        "GNN message passing (see graph example).\n"
        "  GAT: alpha_uv = softmax over neighbours of LeakyReLU(a^T[Wh_u||Wh_v])."
    ),
    "pinn": (
        "PINN data-loss term.\n"
        "  u_theta(0.5,0.5)=0.90, true u=1.00\n"
        "  L_data = (0.90-1.00)^2 = 0.010\n"
        "  Total L = L_data + lambda*L_pde (PDE residual via autograd)."
    ),
    "snn": (
        "Spiking neuron membrane update.\n"
        "  tau_m dV/dt = -(V - V_rest) + R_m I\n"
        "  V=0.5, V_rest=0, I=1.0, tau=10 -> dV/dt = 0.05\n"
        "  If V >= V_th=1.0 emit spike and reset to V_reset."
    ),
    "autoencoder": (
        "Autoencoder reconstruction error.\n"
        "  x=[0.20,0.80]; x_hat=[0.25,0.75]\n"
        "  L = ||x - x_hat||^2 = (0.05)^2+( -0.05)^2 = 0.0050\n"
        "  anomaly score = reconstruction error (high -> anomaly)."
    ),
    "fraud": (
        "Autoencoder anomaly scoring (see autoencoder example).\n"
        "  score = ||x - x_hat||^2; threshold e.g. 0.01 flags fraud."
    ),
    "random-forest": (
        "Random forest ensemble average.\n"
        "  Tree preds: 0.70, 0.80, 0.75 (B=3)\n"
        "  y_hat = (0.70+0.80+0.75)/3 = 0.750\n"
        "  Bagging reduces variance vs a single tree."
    ),
    "transfer": (
        "Transfer learning / distillation.\n"
        "  L = L_task + lambda * KL(p_teacher || p_student)\n"
        "  Soft targets p_i = exp(z_i/T)/sum exp(z_j/T) (T=2) transfer knowledge."
    ),
    "multimodal": (
        "Cross-modal alignment score.\n"
        "  text_emb=[1,1], image_emb=[0.9,1.1]\n"
        "  cos = (1*0.9+1*1.1)/(sqrt2*sqrt(0.81+1.21))\n"
        "      = 2.0/(1.414*1.421) = 0.995 (well aligned)."
    ),
    "pre-training": (
        "Masked language modeling.\n"
        "  Sequence: [The, cat, [MASK], on, the, [MASK]]\n"
        "  Model predicts masked tokens from context (MLM + NSP objectives)."
    ),
    "prompt": (
        "Prompt-conditioned generation probability.\n"
        "  P(y|x,p) = prod_t P(y_t | x,p,y_<t)\n"
        "  Optimal prompt p* = argmax_p E[log P(y* | x,p)]."
    ),
    "code": (
        "Autoregressive code generation.\n"
        "  P(c|p) = prod_t P(c_t | p, c_<t)\n"
        "  L = -sum_t log P(c_t | p, c_<t); beam search decodes."
    ),
    "text": (
        "Autoregressive text generation.\n"
        "  h_t = LSTM(x_t, h_{t-1}); P(w_t|w_<t)=softmax(W_h h_t)\n"
        "  Temperature scales logits; top-k/nucleus truncate mass."
    ),
    "image": (
        "Image synthesis objective (diffusion/VAE/GAN blend).\n"
        "  L_simple = E||epsilon - epsilon_theta(x_t,t)||^2 (diffusion term)."
    ),
    "video": (
        "Video generation likelihood + perceptual quality.\n"
        "  P(x_1..T)=prod_t P(x_t|x_<t); SSIM measures frame fidelity."
    ),
    "retrieval": (
        "RAG retrieval + generation.\n"
        "  sim(q,d) = q.d/(||q|| ||d||); top-k = argmax_{d} sim(q,d)\n"
        "  P(y|x) = sum_z P(y|x,z) P(z|x) over retrieved docs z."
    ),
    "tool": (
        "Tool routing.\n"
        "  logits = [2.1, 0.5, 1.3] -> softmax = [0.62,0.12,0.26]\n"
        "  argmax -> tool 0 invoked with parsed args; result fed back."
    ),
    "default": (
        "Gradient-descent parameter update.\n"
        "  theta = 1.00, learning rate alpha = 0.10, gradient = -0.50\n"
        "  theta <- 1.00 - 0.10*(-0.50) = 1.05\n"
        "  Loss L(theta) decreases each step."
    ),
}

# ---------------------------------------------------------------------------
# DETAILED_EXAMPLES: a rich, multi-angle explanation per algorithm so the
# formal equations become intuitive. Each entry has four labelled parts:
#   Intuition   - plain-English + analogy
#   Concrete    - a tiny real dataset / numbers
#   Step-by-step- the actual computation, line by line
#   Interpretation - what the number means in practice
# Rendered as a "Detailed Walkthrough" in the README math section.
# ---------------------------------------------------------------------------

DETAILED_EXAMPLES = {
    "pizza": (
        "INTUITION: We fit the straightest possible line through (diameter, price)\n"
        "points. Think of it like eyeballing a trend on a scatter plot: bigger pizza\n"
        "=> higher price, and the line captures 'how much per extra inch'.\n"
        "CONCRETE DATA: diameters x=[6,8,10,12,14], prices y=[7,9,13,17.5,18].\n"
        "STEP-BY-STEP:\n"
        "  Start w=0, b=0 -> all predictions 0 -> MSE huge.\n"
        "  Gradient descent repeats: e = y_hat - y; update\n"
        "    w <- w - a*(2/n) * sum(x*e);  b <- b - a*(2/n) * sum(e)\n"
        "  After training (least squares): w=1.525, b=-2.35.\n"
        "  Predict x=12: y_hat = 1.525*12 - 2.35 = 15.95 (~$16).\n"
        "INTERPRETATION: Each extra inch adds ~$1.53 to the price; R^2~0.96 means\n"
        "the line explains the data very well. This exact formula powers /predict."
    ),
    "house": (
        "INTUITION: Same straight-line idea, but x = house size (1000s sqft) and\n"
        "y = price ($1000s). The slope is 'price per square foot'.\n"
        "CONCRETE DATA: sizes x=[1.0,1.5,2.0,2.5], prices y=[150,210,270,330].\n"
        "STEP-BY-STEP:\n"
        "  Fit y = w*x + b. With these points w=120, b=30.\n"
        "  Predict x=1.8: y_hat = 120*1.8 + 30 = 246 (i.e. $246k).\n"
        "INTERPRETATION: ~$120 per (1000 sqft) and a $30k base; MSE/R^2 reported\n"
        "by train.py measure fit quality."
    ),
    "spam": (
        "INTUITION: Instead of a line we output a probability of 'spam' in [0,1].\n"
        "The sigmoid bends any real number into 0..1 like a soft on/off switch.\n"
        "CONCRETE DATA: feature x=2.0 (e.g. count of 'FREE'), w=1.2, b=-1.0.\n"
        "STEP-BY-STEP:\n"
        "  z = w*x + b = 1.2*2.0 - 1.0 = 1.40\n"
        "  y_hat = 1/(1+e^-1.40) = 1/(1+0.247) = 0.802\n"
        "  BCE loss for true label y=1: -log(0.802) = 0.221.\n"
        "INTERPRETATION: 0.80 > 0.5 threshold -> flagged spam. The gradient\n"
        "simplifies to (y_hat - y), so wrong guesses push the weights hard."
    ),
    "digit": (
        "INTUITION: A small neural net scores each of 3 classes; softmax turns\n"
        "raw scores into a probability distribution that sums to 1.\n"
        "CONCRETE DATA: logits z=[1.1, 0.4, -0.3].\n"
        "STEP-BY-STEP:\n"
        "  exp(z) = [3.00, 1.49, 0.74]; sum = 5.23\n"
        "  y_hat = [3.00/5.23, 1.49/5.23, 0.74/5.23] = [0.574, 0.285, 0.142]\n"
        "  Cross-entropy vs true class 0 (one-hot [1,0,0]): -log(0.574)=0.555.\n"
        "INTERPRETATION: Model is 57% sure it's class 0; argmax picks class 0."
    ),
    "anomaly": (
        "INTUITION: Find the directions of greatest spread (principal components)\n"
        "and measure how 'unusual' a point is by its reconstruction error.\n"
        "CONCRETE DATA: centered X=[[1,1],[-1,-1],[1,-1],[-1,1]].\n"
        "STEP-BY-STEP:\n"
        "  Cov Sigma = X^T X/(n-1) = [[1,0],[0,1]]; top PC v=[0.707,0.707].\n"
        "  Project x=[1,1] -> 1.41 (1-D coordinate on PC1).\n"
        "  Reconstruct and compute ||X - X_recon||^2: ~0 for inliers.\n"
        "INTERPRETATION: A new point far from the PC subspace has large error ->\n"
        "flagged anomalous (see api /drift)."
    ),
    "pca": (
        "INTUITION: Compress data by keeping only the axes that carry the most\n"
        "variance; like shadowing a 3-D object onto its most informative 2-D view.\n"
        "CONCRETE DATA: centered X=[[2,1],[-2,-1]].\n"
        "STEP-BY-STEP:\n"
        "  Cov Sigma=[[2,1],[1,0.5]]; eigenvalue lambda1=2.5, v1=[0.894,0.447].\n"
        "  x' = X.v1 = [2.236, -2.236] (now 1-D).\n"
        "  Explained variance = lambda1/(lambda1+0)=100%.\n"
        "INTERPRETATION: One number captures all the information here."
    ),
    "market": (
        "INTUITION: Group customers so similar ones share a cluster; like sorting\n"
        "candy by color and shape into piles.\n"
        "CONCRETE DATA: points {0,0},{1,0},{8,8},{9,8}; init mu0=(0,0), mu1=(9,8).\n"
        "STEP-BY-STEP (one EM round):\n"
        "  Assign nearest: {0,0},{1,0}->C0 ; {8,8},{9,8}->C1.\n"
        "  Recompute: mu0=mean(C0)=(0.5,0); mu1=mean(C1)=(8.5,8).\n"
        "  Objective J = sum of squared distances to own centroid drops.\n"
        "INTERPRETATION: Centroids are the 'average customer' of each segment."
    ),
    "kmeans": (
        "INTUITION: Same K-Means clustering; the worked example for 'market'\n"
        "above is the concrete walkthrough. k controls how many segments you get."
    ),
    "recommendation": (
        "INTUITION: Every user and every item gets a hidden 'taste vector';\n"
        "predicted rating = global mean + user bias + item bias + dot product.\n"
        "CONCRETE DATA: mu=3.5, b_u=0.3, b_i=-0.2, q_i.p_u=0.4.\n"
        "STEP-BY-STEP:\n"
        "  r_hat = 3.5 + 0.3 - 0.2 + 0.4 = 4.0 (predicted 4-star rating).\n"
        "  Cosine(user_a, user_b) ranks how similar two users are.\n"
        "INTERPRETATION: Biases capture 'this user rates high' / 'this movie is\n"
        "bad'; the dot product captures learned preference overlap."
    ),
    "robot": (
        "INTUITION: The agent keeps a table Q(s,a) = 'expected future reward if I\n"
        "do a in state s'. It learns by trial: good moves raise Q, bad lower it.\n"
        "CONCRETE DATA: alpha=0.1, gamma=0.9, Q(s,a)=0.50, r=1.0, max_a'Q=0.80.\n"
        "STEP-BY-STEP:\n"
        "  target = r + gamma*max_a'Q = 1.0 + 0.9*0.80 = 1.72\n"
        "  Q <- 0.50 + 0.1*(1.72 - 0.50) = 0.622\n"
        "INTERPRETATION: This state-action pair became more attractive because it\n"
        "led to a reward; gamma<1 makes far-future rewards worth less."
    ),
    "reinforcement": (
        "INTUITION: Same Q-learning Bellman update as 'robot' above; the state is\n"
        "just the environment (e.g. a maze cell). gamma discounts future reward."
    ),
    "q-learning": (
        "INTUITION: Off-policy TD control. delta is the 'surprise' between what\n"
        "we expected (Q) and what we got (r + future best).\n"
        "CONCRETE DATA: Q=0.50, r=1.0, gamma=0.9, maxQ'=0.80, alpha=0.1.\n"
        "STEP-BY-STEP:\n"
        "  delta = 1.0 + 0.9*0.80 - 0.50 = 1.22\n"
        "  Q <- 0.50 + 0.1*1.22 = 0.622\n"
        "INTERPRETATION: Positive surprise raises Q; with enough tries Q->optimal."
    ),
    "semi": (
        "INTUITION: Use few labelled examples + lots of unlabelled ones by forcing\n"
        "the model to give consistent predictions under small perturbations.\n"
        "CONCRETE DATA: ramp T=2*T0 -> lambda_t = min(1, T/T0) = 1.0.\n"
        "STEP-BY-STEP:\n"
        "  Total loss L = L_sup + 1.0 * L_unsup (full weight late in training).\n"
        "  L_unsup = MSE(f(x'), f(x)) for an augmented view x' of x.\n"
        "INTERPRETATION: Early on trust labels; later enforce consistency so\n"
        "unlabelled data teaches the decision boundary."
    ),
    "semi-supervised": (
        "INTUITION: Labelled data is scarce/expensive, so we also learn from the\n"
        "unlabelled majority by consistency regularization.\n"
        "CONCRETE DATA: lambda_t ramps 0 -> 1 across epochs.\n"
        "STEP-BY-STEP: L = L_sup + lambda_t * MSE(f(x'), f(x)).\n"
        "INTERPRETATION: Pseudo-labels + consistency shape a smoother boundary."
    ),
    "self": (
        "INTUITION: Create labels from data itself: augmented views of the SAME\n"
        "image are 'positive' (should match), others are 'negative' (push apart).\n"
        "CONCRETE DATA: pos sim=0.8, neg sims=0.1,0.2; tau=0.1.\n"
        "STEP-BY-STEP:\n"
        "  num = exp(0.8/0.1) = exp(8) = 2981\n"
        "  den = exp(8)+exp(1)+exp(2) = 2981+2.72+7.39 = 2991\n"
        "  L = -log(2981/2991) = 0.0033\n"
        "INTERPRETATION: Tiny loss means the positive pair is clearly separated\n"
        "from negatives; temperature tau sharpens the contrast."
    ),
    "self-supervised": (
        "INTUITION: Contrastive learning (see 'self' example) — pull a sample's\n"
        "two augmented views together while pushing all others away."
    ),
    "cnn": (
        "INTUITION: A small filter slides over the image, multiplying and summing\n"
        "to detect local patterns (edges, corners) at each position.\n"
        "CONCRETE DATA: row=[1,2,1,0], edge kernel=[1,0,-1].\n"
        "STEP-BY-STEP (valid positions):\n"
        "  pos1: 1*1 + 0*2 + (-1)*1 = 0\n"
        "  pos2: 2*1 + 0*1 + (-1)*0 = 2\n"
        "  ReLU(2)=2 highlights the right edge.\n"
        "INTERPRETATION: Different kernels = different detectors; stacking layers\n"
        "builds edges -> shapes -> objects."
    ),
    "capsnet": (
        "INTUITION: Neurons become 'capsules' (vectors) whose LENGTH is how\n"
        "confident we are and DIRECTION encodes properties (pose, etc.).\n"
        "CONCRETE DATA: s=[1.5,1.5] (vector before squash).\n"
        "STEP-BY-STEP:\n"
        "  ||s|| = sqrt(1.5^2+1.5^2) = 2.12\n"
        "  v = (||s||^2/(1+||s||^2)) * (s/||s||)\n"
        "    = (4.5/5.5) * (0.707,0.707) = (0.578,0.578)\n"
        "INTERPRETATION: Output length 0.818 = ~82% presence; routing iterates\n"
        "coupling coefficients between capsules."
    ),
    "rnn": (
        "INTUITION: A sequence model with memory: each step mixes the new input\n"
        "with the previous hidden state, like reading a sentence word by word.\n"
        "CONCRETE DATA: h_{t-1}=0.30, x_t=0.50, W's=0.5, b=0.\n"
        "STEP-BY-STEP:\n"
        "  pre = 0.5*0.30 + 0.5*0.50 = 0.40\n"
        "  h_t = tanh(0.40) = 0.380\n"
        "INTERPRETATION: h_t encodes everything seen so far; tanh keeps it in\n"
        "[-1,1] (vanishing-gradient mitigated by LSTMs/GRUs)."
    ),
    "lstm": (
        "INTUITION: Adds a 'memory cell' with gates that decide what to forget,\n"
        "store, and output — like a notebook you edit selectively.\n"
        "CONCRETE DATA: f_t=sigmoid(...) from z=0.4 -> f_t=0.598; C_{t-1}=0.80;\n"
        "  i_t=0.4, tildeC_t=0.6.\n"
        "STEP-BY-STEP:\n"
        "  C_t = f_t*C_{t-1} + i_t*tildeC_t = 0.598*0.80 + 0.4*0.6 = 0.718\n"
        "  h_t = o_t * tanh(C_t) (output gate o_t scales what's shown).\n"
        "INTERPRETATION: The cell state carries long-term info across many steps\n"
        "without vanishing."
    ),
    "transformer": (
        "INTUITION: For each token, score how much it should attend to every other\n"
        "token, normalize to weights, and take a weighted average of their values.\n"
        "CONCRETE DATA: Q=[1,0], K=[[1,0],[0,1]], V=[[1,2],[3,4]], d_k=2.\n"
        "STEP-BY-STEP:\n"
        "  scores = Q.K^T = [1*1+0*0, 1*0+0*1] = [1, 0]\n"
        "  scaled = [1,0]/sqrt(2) = [0.707, 0]\n"
        "  softmax([0.707,0]) = [0.67, 0.33]\n"
        "  out = 0.67*[1,2] + 0.33*[3,4] = [1.67, 2.67]\n"
        "INTERPRETATION: Token attends 67% to itself, 33% to the other; multi-head\n"
        "does this in parallel subspaces. Same math as 'attention'."
    ),
    "attention": (
        "INTUITION: Scaled dot-product attention (see 'transformer' walkthrough).\n"
        "Positional encodings PE(pos,2i)=sin(pos/10000^{2i/d}) add order info\n"
        "because attention alone is order-agnostic."
    ),
    "gan": (
        "INTUITION: A forger (generator) and a detective (discriminator) compete:\n"
        "the forger improves until fakes are indistinguishable from real.\n"
        "CONCRETE DATA: D(real)=0.90, D(fake)=0.20.\n"
        "STEP-BY-STEP (discriminator loss):\n"
        "  L_D = -[log 0.90 + log(1-0.20)] = -[(-0.105)+(-0.223)] = 0.328\n"
        "  Generator minimizes -log D(G(z)).\n"
        "INTERPRETATION: At equilibrium D=0.5 (guessing) and G fakes the data."
    ),
    "vae": (
        "INTUITION: Learn a compressed 'code' z from which we can rebuild x;\n"
        "we also force z to look like a standard normal so we can sample new x.\n"
        "CONCRETE DATA: mu=0.1, sigma=1.0 (1-D latent).\n"
        "STEP-BY-STEP:\n"
        "  KL = 0.5*(1 + log(1^2) - 0.1^2 - 1^2) = 0.5*(0 - 0.01) = -0.005\n"
        "  ELBO = Reconstruction - KL.\n"
        "INTERPRETATION: Tiny KL means the code already matches the prior; the\n"
        "reconstruction term dominates training."
    ),
    "diffusion": (
        "INTUITION: Slowly add noise to data until it's pure static, then train a\n"
        "model to reverse one small denoising step at a time.\n"
        "CONCRETE DATA: bar_alpha_t=0.9 -> sqrt=0.949, sqrt(1-0.9)=0.316;\n"
        "  x_0=1.0, epsilon=0.5.\n"
        "STEP-BY-STEP:\n"
        "  x_t = sqrt(bar_alpha_t)*x_0 + sqrt(1-bar_alpha_t)*epsilon\n"
        "      = 0.949*1.0 + 0.316*0.5 = 1.107\n"
        "INTERPRETATION: At t=0 it's the real image; at t=T it's noise. Sampling\n"
        "starts from noise and iteratively predicts/removes epsilon."
    ),
    "graph": (
        "INTUITION: Each node updates its embedding by pooling messages from its\n"
        "neighbors — like gossiping with friends to learn the group's opinion.\n"
        "CONCRETE DATA: node v has neighbors u=[1,0], w=[0,1]; W=I.\n"
        "STEP-BY-STEP:\n"
        "  agg = mean([1,0],[0,1]) = [0.5,0.5]\n"
        "  h_v' = sigma(W.agg) = sigma([0.5,0.5]) = [0.622,0.622]\n"
        "INTERPRETATION: After K rounds, h_v encodes its K-hop neighborhood\n"
        "(GAT weights neighbors by importance)."
    ),
    "gnn": (
        "INTUITION: Graph message passing (see 'graph' walkthrough). GAT adds\n"
        "attention so a node weights important neighbors more."
    ),
    "pinn": (
        "INTUITION: Bake the physics equation directly into the loss so the net\n"
        "must obey the law, not just fit scattered measurements.\n"
        "CONCRETE DATA: u_theta(0.5,0.5)=0.90, true u=1.00.\n"
        "STEP-BY-STEP:\n"
        "  L_data = (0.90 - 1.00)^2 = 0.010\n"
        "  L_pde = residual of the PDE (computed via autograd on u_theta).\n"
        "  L_total = L_data + lambda*L_pde.\n"
        "INTERPRETATION: Even with few data points, the PDE constraint guides the\n"
        "solution everywhere."
    ),
    "snn": (
        "INTUITION: Neurons communicate in discrete spikes, not continuous values;\n"
        "a capacitor (membrane) charges until it fires, then resets.\n"
        "CONCRETE DATA: tau=10, V=0.5, V_rest=0, I=1.0.\n"
        "STEP-BY-STEP:\n"
        "  tau*dV/dt = -(V - V_rest) + R_m*I = -(0.5) + 1.0 = 0.5\n"
        "  dV/dt = 0.05 (slowly charges)\n"
        "  if V >= V_th (e.g. 1.0): emit spike, reset to V_reset.\n"
        "INTERPRETATION: Sparse, event-driven, energy-efficient computation."
    ),
    "autoencoder": (
        "INTUITION: Compress x into a small code z, then decompress back; things\n"
        "the model can't reconstruct well are 'anomalous'.\n"
        "CONCRETE DATA: x=[0.20,0.80]; x_hat=[0.25,0.75].\n"
        "STEP-BY-STEP:\n"
        "  L = ||x - x_hat||^2 = (0.05)^2 + (-0.05)^2 = 0.0050\n"
        "INTERPRETATION: Low error = normal; set a threshold (e.g. 0.01) to flag\n"
        "fraud/novelty."
    ),
    "fraud": (
        "INTUITION: Autoencoder anomaly scoring (see 'autoencoder' example).\n"
        "score = ||x - x_hat||^2; high reconstruction error -> likely fraud."
    ),
    "random-forest": (
        "INTUITION: Many shallow decision trees vote; averaging cancels individual\n"
        "mistakes (wisdom of the crowd).\n"
        "CONCRETE DATA: 3 trees predict 0.70, 0.80, 0.75.\n"
        "STEP-BY-STEP:\n"
        "  y_hat = (0.70 + 0.80 + 0.75)/3 = 0.750\n"
        "INTERPRETATION: Feature subsampling decorrelates trees; OOB error\n"
        "estimates generalization for free."
    ),
    "transfer": (
        "INTUITION: Reuse a model pre-trained on a huge dataset, then tweak the\n"
        "last layers for your small task — standing on giants' shoulders.\n"
        "CONCRETE DATA: teacher logits z=[2,1] (T=2) -> soft p=[0.62,0.38].\n"
        "STEP-BY-STEP:\n"
        "  L = L_task + lambda * KL(p_teacher || p_student)\n"
        "INTERPRETATION: Soft targets (p) carry 'dark knowledge' beyond the 0/1\n"
        "label, improving the small student."
    ),
    "multimodal": (
        "INTUITION: Project text and images into one shared space so a caption\n"
        "and its photo land near each other.\n"
        "CONCRETE DATA: text_emb=[1,1], image_emb=[0.9,1.1].\n"
        "STEP-BY-STEP:\n"
        "  cos = (1*0.9 + 1*1.1) / (sqrt2 * sqrt(0.81+1.21))\n"
        "      = 2.0 / (1.414 * 1.421) = 0.995\n"
        "INTERPRETATION: ~1.0 means the pair is well aligned; cross-attention lets\n"
        "one modality query the other."
    ),
    "pre-training": (
        "INTUITION: Hide some words and force the model to guess them from\n"
        "context — learning grammar/meaning without labels.\n"
        "CONCRETE DATA: 'The cat [MASK] on the [MASK]' -> predict 'sat','mat'.\n"
        "STEP-BY-STEP: L_MLM = -sum log P(correct | context); L_NSP for sentence\n"
        "order; L_total = L_MLM + L_NSP.\n"
        "INTERPRETATION: This self-supervision yields representations fine-tunable\n"
        "to any downstream task."
    ),
    "prompt": (
        "INTUITION: The LLM is a probability machine; a prompt steers it toward\n"
        "the desired output distribution.\n"
        "CONCRETE DATA: P(y|x,p) = prod_t P(y_t | x,p,y_<t).\n"
        "STEP-BY-STEP: optimal prompt p* = argmax_p E[log P(y* | x,p)].\n"
        "INTERPRETATION: Soft (continuous) prompts are tuned by gradient, hard\n"
        "(text) prompts by search."
    ),
    "code": (
        "INTUITION: Source code is just a sequence; the model autoregressively\n"
        "predicts the next token given the docstring/context.\n"
        "CONCRETE DATA: prompt p; tokens c_1..c_T.\n"
        "STEP-BY-STEP: L = -sum_t log P(c_t | p, c_<t); decode via beam search.\n"
        "INTERPRETATION: Low perplexity = high confidence in the generated code."
    ),
    "text": (
        "INTUITION: Predict the next word from history (LSTM/transformer).\n"
        "CONCRETE DATA: h_t = LSTM(x_t, h_{t-1}); P(w_t|w_<t)=softmax(W_h h_t).\n"
        "STEP-BY-STEP: temperature scales logits (high=creative, low=safe);\n"
        "top-k/nucleus truncate the probability mass for diversity.\n"
        "INTERPRETATION: Sampling controls the randomness of generated text."
    ),
    "image": (
        "INTUITION: Image synthesis blends GAN/VAE/Diffusion objectives to map a\n"
        "latent code (or noise) to pixels.\n"
        "CONCRETE DATA: diffusion term L_simple = E||epsilon - epsilon_theta||^2.\n"
        "INTERPRETATION: Lower L_simple -> the model better predicts the noise to\n"
        "remove at each step."
    ),
    "video": (
        "INTUITION: Extend sequence modeling to space+time; frames are predicted\n"
        "conditioned on past frames.\n"
        "CONCRETE DATA: P(x_1..T)=prod_t P(x_t|x_<t); SSIM measures frame fidelity.\n"
        "INTERPRETATION: Temporal consistency (warp/flow) keeps motion smooth."
    ),
    "retrieval": (
        "INTUITION: Before answering, fetch the k most relevant documents, then\n"
        "generate using that evidence (reduces hallucination).\n"
        "CONCRETE DATA: query q, docs d1..d3 with sims 0.9, 0.3, 0.5.\n"
        "STEP-BY-STEP:\n"
        "  top-k=2 -> {d1, d3}; P(y|x) = sum_z P(y|x,z) P(z|x).\n"
        "INTERPRETATION: Grounding in retrieved text makes answers verifiable."
    ),
    "tool": (
        "INTUITION: The model picks a tool (function), fills its arguments,\n"
        "runs it, and feeds the result back to finish the answer.\n"
        "CONCRETE DATA: tool logits [2.1, 0.5, 1.3].\n"
        "STEP-BY-STEP:\n"
        "  softmax = [0.62, 0.12, 0.26] -> argmax = tool 0 invoked.\n"
        "  result = execute(tool0, args); final = generate(q, result).\n"
        "INTERPRETATION: Enables structured reasoning + external API access."
    ),
    "default": (
        "INTUITION: Machine learning fits parameters theta to minimize a loss L\n"
        "that measures prediction error; gradient descent walks downhill.\n"
        "CONCRETE DATA: theta=1.00, alpha=0.10, gradient=-0.50.\n"
        "STEP-BY-STEP:\n"
        "  theta <- 1.00 - 0.10*(-0.50) = 1.05\n"
        "INTERPRETATION: Each step reduces L; learning rate controls step size\n"
        "(too big overshoots, too small crawls)."
    ),
}

# ---------------------------------------------------------------------------
# RUNNABLE_DEMOS: a self-contained Python snippet per algorithm that EXECUTES
# the step-by-step math and prints each intermediate value. The user can copy
# it and run `python` to watch the computation happen (true "execution").
# ---------------------------------------------------------------------------

RUNNABLE_DEMOS = {
    "pizza": """import numpy as np
# Data: pizza diameter (in) and price ($)
x = np.array([6,8,10,12,14.], float)
y = np.array([7,9,13,17.5,18.], float)
# --- 1) CLOSED FORM: exact least-squares fit of the line y = w*x + b ---
xm, ym = x.mean(), y.mean()                     # center the data around its mean
w = ((x-xm)*(y-ym)).sum() / ((x-xm)**2).sum()   # slope  = cov(x,y) / var(x)
b = ym - w*xm                                   # intercept so the line hits (xm,ym)
print("closed-form: w=%.3f  b=%.3f" % (w, b))   # -> 1.525, -2.350
print("predict x=12 ->", round(w*12 + b, 2))    # plug a 12-in diameter into the line
# --- 2) ONE GRADIENT-DESCENT STEP: shows the update rule actually running ---
w2, b2, lr, n = 0.0, 0.0, 0.005, len(x)         # start at zero, pick a small lr
e = w2*x + b2 - y                               # prediction error  e = y_hat - y
dw = (2/n)*np.sum(x*e); db = (2/n)*np.sum(e)    # gradients of MSE wrt w and b
w2 -= lr*dw; b2 -= lr*db                        # take one step downhill
print("after 1 GD step from 0: w=%.3f  b=%.3f" % (w2, b2))""",
    "house": """import numpy as np
# Data: house size (1000s sqft) and price ($1000s)
x = np.array([1.0,1.5,2.0,2.5]); y = np.array([150.,210.,270.,330.])
w, b = 120.0, 30.0                              # fitted slope ($/sqft) and base price
print("y_hat =", np.round(w*x + b, 1))          # predictions at the 4 known sizes
print("R^2   =", round(1 - np.sum((y-(w*x+b))**2)/np.sum((y-y.mean())**2), 3))  # fit quality
print("predict size=1.8 ->", round(w*1.8 + b, 1))  # interpolate a new size""",
    "spam": """import numpy as np
x, w, b = 2.0, 1.2, -1.0                        # one feature, trained weight & bias
z = w*x + b                                     # linear combination (logit)
print("z =", z)
yh = 1/(1 + np.exp(-z))                         # sigmoid squashes z into a (0,1) probability
print("P(spam) =", round(yh, 3))
print("BCE (y=1) =", round(-np.log(yh), 3))     # binary cross-entropy when true label is 1""",
    "digit": """import numpy as np
z = np.array([1.1, 0.4, -0.3])                  # raw scores (logits) for 3 classes
p = np.exp(z) / np.sum(np.exp(z))               # softmax -> normalized probabilities
print("probs   =", np.round(p, 3))
print("class   =", int(np.argmax(p)))           # predicted class = the highest probability
print("CE(y=0) =", round(-np.log(p[0]), 3))     # cross-entropy when the true class is 0""",
    "anomaly": """import numpy as np
X = np.array([[1,1],[-1,-1],[1,-1],[-1,1]], float)   # 4 example points
Xc = X - X.mean(0)                              # center each feature
cov = Xc.T @ Xc / (len(X)-1)                    # covariance matrix
w, v = np.linalg.eigh(cov)                      # eigenvalues & eigenvectors
pc = v[:, np.argmax(w)]                         # principal component = top eigenvector
print("eigvals =", np.round(w, 3))
print("PC1     =", np.round(pc, 3))             # direction of maximum variance
print("1D coords =", np.round(Xc @ pc, 3))      # project the points down to 1D""",
    "pca": """import numpy as np
X = np.array([[2,1],[-2,-1]], float); Xc = X - X.mean(0)  # 2 points, centered
cov = Xc.T @ Xc / (len(X)-1)                    # covariance matrix
w, v = np.linalg.eigh(cov)                      # eigenvalues & eigenvectors
pc = v[:, np.argmax(w)]                         # top principal component
print("lambda1 =", round(w.max(),3))            # variance kept by PC1
print("x' =", np.round(Xc @ pc, 3))             # data projected to 1D
print("var ratio =", round(w.max()/w.sum(),3))  # fraction of variance explained""",
    "market": """import numpy as np
pts = np.array([[0,0],[1,0],[8,8],[9,8]], float)   # customer locations
mu0, mu1 = np.array([0.,0.]), np.array([9.,8.])    # two starting cluster centers
def assign(p): return 0 if np.linalg.norm(p-mu0) < np.linalg.norm(p-mu1) else 1  # nearest center
labels = [assign(p) for p in pts]                 # assign each point to a cluster
print("labels =", labels)
mu0 = pts[np.array(labels)==0].mean(0); mu1 = pts[np.array(labels)==1].mean(0)  # recompute centers
print("new mu0 =", np.round(mu0,2), " mu1 =", np.round(mu1,2))""",
    "kmeans": """import numpy as np
pts = np.array([[0,0],[1,0],[8,8],[9,8]], float)   # 4 points, two natural groups
mu = np.array([[0.,0.],[9.,8.]])                  # initial cluster centers
for _ in range(2):                                # 2 Lloyd iterations
    d = ((pts[:,None,:]-mu)**2).sum(-1)           # squared distance to each center
    lab = d.argmin(1)                             # assign each point to nearest center
    mu = np.array([pts[lab==k].mean(0) for k in range(2)])  # move centers to the means
    print("centroids =", np.round(mu,2))""",
    "recommendation": """mu, b_u, b_i, qdotp = 3.5, 0.3, -0.2, 0.4   # global mean, user & item bias, latent dot
rhat = mu + b_u + b_i + qdotp                     # matrix-factorization prediction
print("predicted rating =", rhat)""",
    "robot": """alpha, gamma, Q, r, maxQ = 0.1, 0.9, 0.5, 1.0, 0.8   # lr, discount, Q(s,a), reward, max next Q
print("target =", r + gamma*maxQ)                # Bellman target value
Q = Q + alpha*(r + gamma*maxQ - Q)               # move Q toward the target by lr
print("updated Q =", round(Q, 3))""",
    "reinforcement": """alpha, gamma, Q, r, maxQ = 0.1, 0.9, 0.5, 1.0, 0.8   # lr, discount, Q(s,a), reward, max next Q
Q = Q + alpha*(r + gamma*maxQ - Q)               # Q-learning / TD update toward target
print("updated Q =", round(Q, 3))""",
    "q-learning": """alpha, gamma, Q, r, maxQ = 0.1, 0.9, 0.5, 1.0, 0.8   # lr, discount, Q(s,a), reward, max next Q
delta = r + gamma*maxQ - Q                       # temporal-difference error (target - current)
Q = Q + alpha*delta                              # nudge Q by lr * TD error
print("delta =", delta, " updated Q =", round(Q, 3))""",
    "semi": """T, T0 = 2, 1                                     # current & initial training temps
lam = min(1, T/T0)                               # ramp weight for the unsupervised term
print("lambda_t =", lam)
print("total L  =", 0.4 + lam*0.3)               # L = supervised + lambda * unsupervised""",
    "semi-supervised": """lam = 1.0                                       # full weight on unlabeled loss
print("total L =", 0.4 + lam*0.3)               # L = supervised + lambda * unsupervised""",
    "self": """import numpy as np
pos, negs, tau = 0.8, [0.1, 0.2], 0.1           # positive score, negative scores, temperature
num = np.exp(pos/tau)                            # numerator: exp(positive / tau)
den = num + sum(np.exp(n/tau) for n in negs)     # denominator: all scores
print("L =", round(-np.log(num/den), 4))         # InfoNCE = -log(positive / all)""",
    "self-supervised": """import numpy as np
pos, negs, tau = 0.8, [0.1, 0.2], 0.1           # positive score, negative scores, temperature
num = np.exp(pos/tau)                            # numerator: exp(positive / tau)
den = num + sum(np.exp(n/tau) for n in negs)     # denominator: positive + all negatives
print("InfoNCE L =", round(-np.log(num/den), 4)) # contrastive loss pulls pos. closer than negs.""",
    "cnn": """import numpy as np
row = np.array([1.,2.,1.,0.]); k = np.array([1.,0.,-1.])  # 1D signal and an edge-detection kernel
for i in range(len(row)-len(k)+1):              # slide the kernel across the signal
    print("pos", i, "->", float(row[i:i+3] @ k))  # dot product = convolution output""",
    "capsnet": """import numpy as np
s = np.array([1.5, 1.5], float); nrm = np.linalg.norm(s)  # capsule vote vector & its length
v = (nrm**2/(1+nrm**2)) * (s/nrm)               # squash: keeps direction, squashes length to (0,1)
print("v =", np.round(v, 3), " length =", round(np.linalg.norm(v), 3))""",
    "rnn": """import numpy as np
h_prev, x_t, Whh, Wxh, b = 0.3, 0.5, 0.5, 0.5, 0.0  # prev hidden, input, weights, bias
pre = Whh*h_prev + Wxh*x_t + b                 # affine combine of last state and input
print("pre =", pre, " h_t =", round(np.tanh(pre), 3))  # tanh nonlinearity -> new hidden state""",
    "lstm": """import numpy as np
sig = lambda z: 1/(1+np.exp(-z))                # sigmoid gate activation
f_t, C_prev, i_t, tC = sig(0.4), 0.8, 0.4, 0.6  # forget gate, old cell, input gate, new candidate
C = f_t*C_prev + i_t*tC                        # cell = forget*old + input*candidate
print("C_t =", round(C, 3))""",
    "transformer": """import numpy as np
Q = np.array([1.,0.]); K = np.array([[1.,0.],[0.,1.]]); V = np.array([[1.,2.],[3.,4.]])  # query, keys, values
dk = 2                                           # key dimension (for scaling)
scores = Q @ K.T / np.sqrt(dk)                   # dot-product similarity, scaled
w = np.exp(scores) / np.sum(np.exp(scores))      # softmax -> attention weights
out = w @ V                                      # weighted sum of the values
print("weights =", np.round(w, 3), " out =", np.round(out, 3))""",
    "attention": """import numpy as np
Q = np.array([1.,0.]); K = np.array([[1.,0.],[0.,1.]]); V = np.array([[1.,2.],[3.,4.]])  # query, keys, values
dk = 2                                           # key dimension (for scaling)
scores = Q @ K.T / np.sqrt(dk)                   # dot-product similarity, scaled
w = np.exp(scores) / np.sum(np.exp(scores))      # softmax -> attention weights
print("attn weights =", np.round(w, 3), " out =", np.round(w @ V, 3))  # weighted sum of values""",
    "gan": """import numpy as np
D_real, D_fake = 0.9, 0.2                        # discriminator outputs for real & fake samples
L_D = -(np.log(D_real) + np.log(1 - D_fake))    # discriminator loss: maximize both correct calls
print("L_D =", round(L_D, 3))""",
    "vae": """import numpy as np
mu, sig = 0.1, 1.0                              # latent mean and std-dev
KL = 0.5*(1 + np.log(sig**2) - mu**2 - sig**2)  # KL divergence from N(mu,sig^2) to N(0,1)
print("KL =", round(KL, 4))""",
    "diffusion": """import numpy as np
bar_alpha, x0, eps = 0.9, 1.0, 0.5              # cumulative alpha, clean image, noise
xt = np.sqrt(bar_alpha)*x0 + np.sqrt(1-bar_alpha)*eps  # forward noising: blend signal + noise
print("x_t =", round(xt, 3))""",
    "graph": """import numpy as np
neigh = np.array([[1.,0.],[0.,1.]]); W = np.eye(2)  # neighbors of a node & weight matrix
agg = neigh.mean(0)                            # aggregate (mean) neighbor features
hv = 1/(1 + np.exp(-(W @ agg)))                # transform + activate -> new node embedding
print("h_v =", np.round(hv, 3))""",
    "gnn": """import numpy as np
neigh = np.array([[1.,0.],[0.,1.]]); W = np.eye(2)  # neighbors of a node & weight matrix
agg = neigh.mean(0)                            # aggregate (mean) neighbor messages
hv = 1/(1 + np.exp(-(W @ agg)))                # message transform + activation
print("h_v =", np.round(hv, 3))""",
    "pinn": """u_theta, u_true = 0.9, 1.0                      # network prediction & true value
L_data = (u_theta - u_true)**2                 # data-fitting term of the physics loss
print("L_data =", L_data)""",
    "snn": """tau, V, Vr, I = 10, 0.5, 0.0, 1.0               # membrane time const, voltage, rest, input
dV = (-(V - Vr) + I)/tau                       # leaky integrate-and-fire membrane dynamics
print("dV/dt =", dV, " (spikes if V >= threshold)")""",
    "autoencoder": """import numpy as np
x = np.array([0.2, 0.8]); xh = np.array([0.25, 0.75])  # input and its reconstruction
print("recon error =", float(np.sum((x - xh)**2)))     # MSE between input and output""",
    "fraud": """import numpy as np
x = np.array([0.2, 0.8]); xh = np.array([0.25, 0.75])  # transaction & its reconstruction
print("anomaly score =", float(np.sum((x - xh)**2)))   # high error => likely fraud""",
    "random-forest": """preds = [0.7, 0.8, 0.75]                    # predictions from 3 trees
print("ensemble =", sum(preds)/len(preds))           # average the trees (bagging)""",
    "transfer": """import numpy as np
z = np.array([2.,1.]); T = 2                        # teacher logits & temperature
p = np.exp(z/T); p = p/p.sum()                     # soften into soft probability targets
print("soft targets =", np.round(p, 3))""",
    "multimodal": """import numpy as np
t = np.array([1.,1.]); im = np.array([0.9,1.1])    # text & image embedding vectors
cos = float(t @ im / (np.linalg.norm(t)*np.linalg.norm(im)))  # cosine similarity
print("cosine =", round(cos, 3))""",
    "pre-training": """print("Mask: 'The cat [MASK] on the [MASK]'")   # hide tokens, let the model fill them
print("Model predicts: 'sat' and 'mat' from context")""",
    "prompt": """import numpy as np
probs = np.array([0.62, 0.12, 0.26])              # language-model token probabilities
print("chosen token =", int(np.argmax(probs)))    # greedy decode = pick highest prob""",
    "code": """import numpy as np
probs = np.array([0.62, 0.12, 0.26])              # next-token probabilities
print("next token idx =", int(np.argmax(probs)))  # greedy decode = highest prob""",
    "text": """import numpy as np
h = 0.38
logits = np.array([1.1, 0.4, -0.3])               # raw scores for the next word
p = np.exp(logits)/np.sum(np.exp(logits))         # softmax -> word probabilities
print("P(w_t) =", np.round(p, 3))""",
    "image": """import numpy as np
eps = np.array([0.3, 0.1]); epshat = np.array([0.25, 0.15])  # true & predicted noise
print("L_simple =", float(np.mean((eps - epshat)**2)))      # simple diffusion loss (MSE on noise)""",
    "video": """import numpy as np
eps = np.array([0.3, 0.1]); epshat = np.array([0.25, 0.15])  # true & predicted noise per frame
print("frame loss =", float(np.mean((eps - epshat)**2)))    # simple diffusion loss (MSE on noise)""",
    "retrieval": """import numpy as np
q = np.array([1.,1.]); docs = np.array([[0.9,1.1],[0.2,0.3],[0.4,0.5]])  # query & candidate docs
sims = docs @ q / (np.linalg.norm(docs, axis=1)*np.linalg.norm(q))  # cosine similarity
print("sims =", np.round(sims, 3), " top-2 =", np.argsort(-sims)[:2])  # most similar first""",
    "tool": """import numpy as np
logits = np.array([2.1, 0.5, 1.3])                # scores for three tools
p = np.exp(logits)/np.sum(np.exp(logits))         # softmax -> selection probabilities
print("probs =", np.round(p, 3), " chosen tool =", int(np.argmax(p)))  # pick the best tool""",
    "default": """theta, alpha, grad = 1.0, 0.1, -0.5          # parameter, learning rate, gradient
theta = theta - alpha*grad                       # gradient-descent update step
print("updated theta =", theta)""",
}

MATH_PRIORITY = [
    "pizza", "spam", "house", "fraud", "digit",
    "robot", "market", "recommendation", "anomaly",
    "semi-supervised", "self-supervised",
    "cnn", "capsnet",
    "attention", "transformer",
    "gan", "vae", "diffusion",
    "lstm", "rnn",
    "graph", "pinn", "snn",
    "reinforcement", "q-learning",
    "kmeans", "pca", "autoencoder", "random-forest",
    "transfer", "multimodal", "pre-training",
    "prompt", "code", "text", "video", "retrieval", "tool",
    # App-name synonyms (specific before generic)
    "captioning", "speech", "music", "stock", "weather",
    "deep-belief", "boltzmann", "self-organizing", "large-language-model",
    "image",  # generic, should come last
]


SYNONYM_MAP = {
    "speech": "rnn",
    "music": "rnn",
    "time-series": "rnn",
    "stock": "rnn",
    "weather": "rnn",
    "captioning": "rnn",
    "deep-belief": "autoencoder",
    "boltzmann": "autoencoder",
    "self-organizing": "autoencoder",
    "large-language-model": "transformer",
}


def _guess_math_key(app_name: str, source: str = "") -> str:
    lower = app_name.lower()
    name_path = lower + " " + lower.replace("-", " ")
    words = set(re.findall(r"\b\w+\b", name_path))
    # crude singularisation so plurals (e.g. "transformers") match ("transformer")
    def _stem(w: str) -> str:
        return w[:-1] if len(w) > 3 and w.endswith("s") else w
    norm_words = {_stem(w) for w in words}
    for key in MATH_PRIORITY:
        key_words = set(re.findall(r"\b\w+\b", key))
        norm_key = {_stem(k) for k in key_words}
        if len(key_words) == 1:
            kw = next(iter(key_words))
            # exact word OR singular/plural match
            if kw in words or _stem(kw) in norm_words:
                return SYNONYM_MAP.get(key, key)
        else:
            if key_words.issubset(words) or norm_key.issubset(norm_words):
                return SYNONYM_MAP.get(key, key)
    return "default"


def _related_apps(app_name: str, all_apps: list[str]) -> list[str]:
    """Find related apps based on category and name keywords."""
    related = []
    keywords = re.split(r"[-_]", app_name)
    for other in all_apps:
        if other == app_name:
            continue
        other_keywords = re.split(r"[-_]", other)
        if set(keywords) & set(other_keywords):
            related.append(other)
    return related[:4]


# ---------------------------------------------------------------------------
# HTML generation helpers
# ---------------------------------------------------------------------------

def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _code_block(source: str, lang: str = "python") -> str:
    escaped = _escape_html(source.strip())
    uid = f"code-{abs(hash(source)) & 0xFFFFFFFF}"
    return f"""<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('{uid}')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="{uid}"><code class="language-{lang}">{escaped}</code></pre>
</div>"""


def _param_table(params: list[dict]) -> str:
    if not params:
        return "<p class=\"muted\">No explicit parameters.</p>"
    rows = "\n".join(
        f"<tr><td><code>{_escape_html(p['name'])}</code></td><td><code>{_escape_html(p['type'])}</code></td></tr>"
        for p in params
    )
    return f"""<table class="param-table">
<thead><tr><th>Parameter</th><th>Type</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _generate_math_section(math_key: str) -> str:
    tmpl = MATH_TEMPLATES.get(math_key, MATH_TEMPLATES["default"])
    eqs = "\n".join(f"<div class=\"math-block\">{e}</div>" for e in tmpl["equations"])
    return f"""<section id="math" class="section math-section">
<h2><span class="section-icon">∫</span> Mathematics &amp; Theory</h2>
<p class="section-subtitle">{_escape_html(tmpl['title'])} — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations">{eqs}</div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>{_escape_html(tmpl['derivation'])}</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>{_escape_html(tmpl['visualization'])}</p>
</div>
</div>
</section>"""


def _read_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _extract_docstring(source: str, class_name: str) -> str:
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                doc = ast.get_docstring(node)
                return doc or ""
    except Exception:
        pass
    return ""


def _extract_methods(source: str, class_name: str) -> list[dict]:
    methods = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        args = [a.arg for a in item.args.args]
                        methods.append({
                            "name": item.name,
                            "args": args,
                            "doc": ast.get_docstring(item) or "",
                        })
    except Exception:
        pass
    return methods


def generate_readme(app_dir: Path, app_name: str, all_apps: list[str]) -> str:
    """Generate a comprehensive HTML README for a single app."""
    src_dir = app_dir / "src"
    # Find the main package directory
    pkg_dirs = [d for d in src_dir.iterdir() if d.is_dir() and "egg-info" not in d.name] if src_dir.exists() else []
    pkg_dir = pkg_dirs[0] if pkg_dirs else app_dir

    # Read source files
    pyproject = _read_file(app_dir / "pyproject.toml")
    model_src = _read_file(pkg_dir / "model.py")
    train_src = _read_file(pkg_dir / "train.py")
    data_src = _read_file(pkg_dir / "data.py")
    api_src = _read_file(pkg_dir / "api.py")
    init_src = _read_file(pkg_dir / "__init__.py")
    existing_readme = _read_file(app_dir / "README.md")

    # Parse metadata from pyproject.toml
    description = ""
    version = "0.1.0"
    category = "AI"
    difficulty = "Intermediate"
    dependencies = []

    m = re.search(r'description\s*=\s*"([^"]+)"', pyproject)
    if m:
        description = m.group(1)

    m = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
    if m:
        version = m.group(1)

    # Determine category from path
    path_str = str(app_dir)
    if "machine-learning" in path_str:
        category = "Machine Learning"
    elif "neural-networks" in path_str:
        category = "Neural Networks"
    elif "deep-learning" in path_str:
        category = "Deep Learning"
    elif "generative-ai" in path_str:
        category = "Generative AI"

    # Determine difficulty
    if any(x in path_str for x in ["advanced-", "diffusion", "gan", "vae", "transformer", "multimodal"]):
        difficulty = "Advanced"
    elif any(x in path_str for x in ["cnn-", "rnn", "lstm", "attention"]):
        difficulty = "Intermediate"
    else:
        difficulty = "Beginner"

    # Extract dependencies
    for line in pyproject.splitlines():
        if ">=" in line and "#" not in line:
            dep = line.strip().split(">=")[0].strip('", ')
            if dep and dep not in ["python", "typing"]:
                dependencies.append(dep)

    deps_str = ", ".join(dependencies[:10]) if dependencies else "ai-core, numpy"

    # Determine math template
    math_key = _guess_math_key(app_name, model_src + train_src + api_src)
    math_section = _generate_math_section(math_key)

    # Extract classes and methods from model.py
    classes = []
    try:
        tree = ast.parse(model_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        args = [a.arg for a in item.args.args]
                        methods.append({
                            "name": item.name,
                            "args": args,
                            "doc": ast.get_docstring(item) or "",
                        })
                classes.append({
                    "name": node.name,
                    "doc": ast.get_docstring(node) or "",
                    "methods": methods,
                })
    except Exception:
        pass

    # Generate architecture diagram
    arch_diagram = ""
    if classes:
        class_names = [c["name"] for c in classes]
        arch_diagram = "\n".join(f"  {c}" for c in class_names)
    else:
        arch_diagram = f"{app_name.replace('-', ' ').title()}"

    # Mermaid diagram
    mermaid = f"""graph TD
  A[Input Data] --> B[Preprocessing]
  B --> C[Model Training]
  C --> D[Evaluation]
  D --> E[Model Registry]
  E --> F[Serving API]"""

    # API endpoints
    api_endpoints = []
    if api_src:
        for line in api_src.splitlines():
            m = re.search(r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']\)', line)
            if m:
                api_endpoints.append({"method": m.group(1).upper(), "path": m.group(2)})

    api_section = ""
    if api_endpoints:
        rows = "\n".join(
            f"<tr><td><code>{e['method']}</code></td><td><code>{e['path']}</code></td></tr>"
            for e in api_endpoints
        )
        api_section = f"""<section id="api" class="section api-section">
<h2><span class="section-icon">⚡</span> API Reference</h2>
<p class="section-subtitle">FastAPI endpoints and model interfaces</p>
<table class="api-table">
<thead><tr><th>Method</th><th>Endpoint</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</section>"""

    # Code examples
    code_examples = ""
    if train_src:
        code_examples += f"""<h3>Training Script</h3>
{_code_block(train_src, "python")}"""
    if api_src:
        code_examples += f"""<h3>API Server</h3>
{_code_block(api_src, "python")}"""

    usage_section = f"""<section id="usage" class="section usage-section">
<h2><span class="section-icon">▶</span> Usage</h2>
<p class="section-subtitle">Code examples and CLI commands</p>
{code_examples}
<h3>CLI Commands</h3>
{_code_block(f"uv run python -m {app_name.replace('-', '_')}.train --model-dir ./artifacts/models", "bash")}
</section>"""

    # Related apps
    related = _related_apps(app_name, all_apps)
    related_html = ""
    if related:
        links = "\n".join(f"<li><a href=\"../{r}/README.md\">{r}</a></li>" for r in related[:4])
        related_html = f"""<div class="related-links">
<h3>Related Apps</h3>
<ul>{links}</ul>
</div>"""

    # Benchmark section
    bench_section = """<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>"""

    # Assemble full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{app_name} - AI App Documentation</title>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" onload="renderMath()"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
/* CSS styles here */
</style>
</head>
<body>
{math_section}
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">{arch_diagram}</pre>
</div>
<div class="mermaid-wrapper">
<h3>Data Flow</h3>
<pre class="mermaid">{mermaid}</pre>
</div>
</section>
{api_section}
{usage_section}
{bench_section}
{related_html}
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>{app_name}</strong></p>
</footer>
<script>
function copyCode(id) {{
  const el = document.getElementById(id);
  navigator.clipboard.writeText(el.innerText);
}}
function renderMath() {{
  renderMathInElement(document.body, {{ delimiters: [{{left: "$$", right: "$$", display: true}}] }});
}}
</script>
</body>
</html>"""
    return html


def main():
    apps_root = Path("/Users/avi/Documents/ai/apps")
    all_apps = []
    for pyproject in sorted(apps_root.rglob("pyproject.toml")):
        app_dir = pyproject.parent
        app_name = app_dir.name
        all_apps.append(app_name)

    for app_name in all_apps:
        app_dir = None
        for pyproject in apps_root.rglob("pyproject.toml"):
            if pyproject.parent.name == app_name:
                app_dir = pyproject.parent
                break
        if app_dir is None:
            continue

        readme_path = app_dir / "README.md"
        html = generate_readme(app_dir, app_name, all_apps)
        readme_path.write_text(html, encoding="utf-8")
        print(f"Generated: {readme_path}")

    print(f"\nTotal READMEs generated: {len(all_apps)}")


if __name__ == "__main__":
    main()
