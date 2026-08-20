# Transfer Learning

Transfer learning is a technique where a model trained on one task is reused for a related task, especially when the new task has limited data. This helps in the following ways:

- Uses learned features from the first task
- Reduces training time for the new task
- Improves accuracy with less data
- Uses general features that work across tasks

## Network Type
Transfer Learning with Frozen/Trainable Layers

## Architecture

### Key Components

1. **Pre-trained Model (Base Model)**: Start with a model already trained on a large dataset for a specific task. This pre-trained model has learned general features and patterns that are relevant across related tasks.

2. **Transfer Layers**: Identify layers within the base model that hold generic information applicable to both the original and new tasks. Lower layers capture general features such as edges and textures, while higher layers capture task-specific complex patterns.

3. **Classifier Head**: New task-specific layers added on top of the frozen base model for the target classification/regression task.

4. **Fine-tuning Mechanism**: Gradual unfreezing of top layers for adaptation to the new task while retaining pre-trained knowledge.

### Frozen vs. Trainable Layers

| Aspect | Frozen Layers | Trainable Layers |
|--------|--------------|------------------|
| Definition | Layers whose weights are kept fixed and not updated during training | Layers whose weights are updated during training |
| Purpose | Preserve general features learned from large pre-trained datasets | Adapt to task-specific features of the new dataset |
| Learning Process | No backpropagation updates; remain constant | Updated through backpropagation based on new data |
| Use Case | Used when new dataset is small or similar to the original dataset | Used when new dataset is large or significantly different from the original task |
| Computation Cost | Lower, since fewer parameters are trained | Higher, as more parameters need to be updated |
| Example in CNN | Early convolutional layers that capture edges, textures and basic shapes | Later fully connected layers or deeper convolutional layers for fine-tuned features |

### How to Decide Which Layers to Freeze or Train

- **Small, Similar Dataset**: For smaller datasets that resemble the original dataset, freeze most layers and only fine-tune the last one or two layers to prevent overfitting.
- **Large, Similar Dataset**: With large, similar datasets you can unfreeze more layers allowing the model to adapt while retaining learned features from the base model.
- **Small, Different Dataset**: For smaller, dissimilar datasets, fine-tuning layers closer to the input layer helps the model learn task-specific features from scratch.
- **Large, Different Dataset**: In this case, fine-tuning the entire model helps the model adapt to the new task while using the broad knowledge from the pre-trained model.

## Working of Transfer Learning

1. **Pre-trained Model**: Start with a model already trained on a large dataset for a specific task.
2. **Base Model**: This pre-trained model, known as the base model, includes layers that have processed data to learn hierarchical representations, capturing low-level to complex features.
3. **Transfer Layers**: Identify layers within the base model that hold generic information applicable to both the original and new tasks.
4. **Fine-tuning**: Fine-tune these selected layers with data from the new task. This process helps retain the pre-trained knowledge while adjusting parameters to meet the specific requirements of the new task.

## Training

```bash
transfer_learning-train --model-dir ./artifacts/models --n-samples 500 --n-iterations 100 --freeze-base --fine-tune-layers 1
```

### Parameters
- `--freeze-base`: Freeze the base model (preserve pre-trained features)
- `--fine-tune-layers`: Number of top layers to fine-tune
- `--fine-tune-at`: Epoch at which to start fine-tuning
- `--learning-rate`: Learning rate for new classification head
- `--fine-tune-lr`: Learning rate for fine-tuning base model layers

## Serving API

```bash
uvicorn transfer_learning.api:app --host 0.0.0.0 --port 8013
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Predict class using transfer learning model
- `GET /stats` - Model statistics including frozen/trainable layers info
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Input Format
- `tokens`: list of token indices (max 32)
- `max_len`: max generation length

## Applications

- **Computer Vision**: Used in image recognition where pre-trained models are adapted for tasks like medical imaging, facial recognition and object detection.
- **Natural Language Processing (NLP)**: Models like BERT, GPT and ELMo are pre-trained on large text data and fine-tuned for tasks such as sentiment analysis and question answering.
- **Healthcare**: Helps in building diagnostic systems by applying learned features to medical images like X-rays and MRIs.
- **Finance**: Used for fraud detection, risk assessment and credit scoring by transferring patterns from related financial data.

## Advantages

- **Speed up the training process**: Speeds up training by using a pre-trained model that already understands important features and patterns.
- **Better performance**: Improves performance on the new task by leveraging knowledge learned from the previous task.
- **Handling small datasets**: Works well with limited data by using general features, helping to reduce overfitting.

## Limitations

- **Domain mismatch**: The pre-trained model may perform poorly if the source and target tasks or data distributions are very different.
- **Overfitting**: Excessive fine-tuning can make the model too task-specific, reducing generalization.
- **Complexity**: Requires high computational resources and may need specialized hardware.

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
