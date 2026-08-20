# Pre-Training and Fine-Tuning in Deep Learning

Deep analysis of pre-training and fine-tuning paradigms in deep learning, covering objectives, architectures, strategies, applications, advantages, and limitations.

## Deep Analysis

### 1. Pre-Training

Pre-training is the initial phase in building machine learning models, especially large language models, where the system learns from large amounts of unlabeled data to capture general patterns and knowledge.

#### How Pre-Training Works

- The model is trained from scratch or initialized with weights
- Learns general features using objectives like masked token prediction or next-token prediction
- A projection head maps learned features to the training objective
- Knowledge gained during pre-training is transferred to the fine-tuning phase
- During fine-tuning, task-specific layers may be added

#### Pre-Training Objectives

| Objective | Description |
|-----------|-------------|
| **Masked Language Modeling (MLM)** | The model learns to predict missing or masked words in a sentence using surrounding context |
| **Next-Token Prediction** | The model predicts the next word in a sequence based on previous words |
| **Context Learning** | The model learns relationships between words and understands context within text |
| **Representation Learning** | The model generates meaningful vector representations that can be used for various downstream tasks |

#### Pre-Training Applications

- **NLP**: Chatbots, sentiment analysis, translation, summarization
- **Computer Vision**: Image classification, object detection, medical analysis
- **Speech**: Speech-to-text, voice assistants, audio classification
- **Code Generation**: Systems that assist in writing and debugging programs

#### Pre-Training Advantages

- Reduces time and computational effort required for fine-tuning
- Improves accuracy and generalization across different tasks
- Requires less labeled data, reducing cost and effort
- Enables the same model to be adapted to multiple tasks without retraining from scratch

---

### 2. Fine-Tuning

Fine-tuning is a technique that adapts a pre-trained model to a new task. It uses the knowledge learned from training on a large dataset and applies it to a smaller, task-specific dataset, improving performance while reducing training time.

#### Key Characteristics

- Uses a pre-trained model as the starting point
- Adjusts model weights to perform better on a new task
- Requires less data and training time than training from scratch
- Commonly used in transfer learning
- Helps improve performance on domain-specific tasks

#### Types of Fine-Tuning

| Type | Description |
|------|-------------|
| **Full Fine-Tuning** | All parameters of the pre-trained model are updated using the new dataset. Updates every layer, provides maximum flexibility and performance, but requires significant computational resources |
| **Feature Extraction** | The pre-trained model is used as a fixed feature extractor and only the final task-specific layers are trained. Most layers remain frozen, faster and more efficient, suitable for small datasets |
| **Partial Fine-Tuning** | Only selected layers of the model are updated while the remaining layers stay frozen. Balances performance and computational cost, preserves general knowledge |
| **Parameter-Efficient Fine-Tuning (PEFT)** | Updates only a small subset of model parameters instead of the entire model. Reduces memory and storage requirements, faster than full fine-tuning |
| **Low-Rank Adaptation (LoRA)** | Adds small trainable matrices to the model while keeping the original weights frozen. Requires fewer trainable parameters, reduces computational cost |
| **Prompt Tuning** | Learns a set of trainable prompts while keeping the model parameters unchanged. Requires minimal training resources, maintains original model weights |

#### Working of Fine-Tuning

1. **Select a Pre-Trained Model**: Choose a model trained on a large and diverse dataset (e.g., BERT for NLP, ResNet for image classification, GPT for text generation)
2. **Freeze Initial Layers**: Early layers are kept unchanged because they have learned general features (edges, shapes, textures for images; basic grammar and word relationships for language)
3. **Fine-Tune Later Layers**: Later layers are updated using the new dataset to learn task-specific patterns
4. **Use a Small Learning Rate**: A lower learning rate makes gradual adjustments to preserve previously learned knowledge while adapting to the new task
5. **Evaluate and Refine**: Test the model on the target task, adjust layers or training parameters to improve accuracy

#### Fine-Tuning Applications

- Adapt general-purpose models to specific domains (healthcare, finance, legal)
- Improve performance on specialized tasks (sentiment analysis, question answering, named entity recognition)
- Help models understand specific languages, dialects, or writing styles
- Enable personalization based on user preferences, vocabulary, or tone
- Effective learning from smaller datasets without training from scratch
- Deployment of optimized models on mobile devices and IoT systems

#### Fine-Tuning Advantages

- Works well even when only a small amount of training data is available
- Improves performance on domain-specific tasks
- Saves time by adapting an existing model instead of training from scratch
- Provides better accuracy by leveraging knowledge from large datasets
- Reduces the risk of overfitting on smaller datasets
- Requires fewer computational resources compared to full model training

#### Fine-Tuning Limitations

- May still suffer from overfitting if the new dataset is too small or lacks diversity
- Can require significant computational resources, especially for large models
- Choosing which layers to freeze and which to fine-tune can be challenging
- Performance depends on the quality and relevance of the pre-trained model
- May not work well when the new task is very different from the original training task

---

### 3. Relationship Between Pre-Training and Fine-Tuning

Pre-training and fine-tuning form a two-stage transfer learning pipeline:

```
Large Unlabeled Data --> [Pre-Training] --> Pre-trained Model --> [Fine-Tuning] --> Task-Specific Model
```

- **Pre-training** builds general knowledge from large-scale unlabeled data
- **Fine-tuning** adapts that general knowledge to a specific downstream task
- The pre-trained model acts as a universal feature extractor
- Fine-tuning is typically much faster and requires less data than pre-training

---

### 4. Network Type

This app implements a Transformer-based architecture that supports both:
- **Pre-training** with Masked Language Modeling (MLM) and Next-Token Prediction objectives
- **Fine-tuning** with Full, Feature Extraction, Partial, and PEFT-style strategies

## Architecture

- **Token Embeddings**: vocab_size -> d_model
- **Positional Encoding**: Sinusoidal sin/cos positional encodings
- **Self-Attention**: Q, K, V vectors with scaled dot-product attention
- **Multi-Head Attention**: h parallel attention heads
- **Feed-Forward Networks**: Position-wise FFN with GELU activation
- **Add & Norm**: Residual connections + Layer Normalization
- **Pre-training Heads**: MLM head (masked token prediction) + NTP head (next-token prediction)
- **Fine-tuning Heads**: Task-specific classification/regression heads with frozen/unfrozen layer control

## Training

### Pre-training Phase
```bash
pre-training-fine-tuning-train --model-dir ./artifacts/models --phase pretrain --n-samples 1000 --n-iterations 200 --objective mlm
```

### Fine-tuning Phase
```bash
pre-training-fine-tuning-train --model-dir ./artifacts/models --phase finetune --n-samples 500 --n-iterations 100 --strategy partial --learning-rate 0.0001
```

## Serving API

```bash
uvicorn pre_training_fine_tuning.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /pretrain` - Run pre-training with MLM or NTP objective
- `POST /finetune` - Fine-tune with selected strategy
- `POST /predict` - Inference with pre-trained or fine-tuned model
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Input Format

### Pre-training
- `tokens`: list of token indices
- `mask_positions`: positions to mask for MLM objective
- `objective`: "mlm" or "ntp"

### Fine-tuning
- `tokens`: list of token indices
- `label`: target class/label
- `strategy`: "full", "feature_extraction", "partial", "peft"

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
