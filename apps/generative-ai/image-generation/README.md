# Image Generation

Image generation is a generative AI task that creates new images from text descriptions or reference inputs. Modern approaches use diffusion models and variational autoencoders (VAEs) to learn the distribution of visual data and synthesize novel images that match a given prompt.

## Network Type
Diffusion Model with Variational Autoencoder (VAE)

## Architecture

### Key Components

1. **ImageTokenizer**: Encodes text prompts into token IDs using a vocabulary of visual concepts and words.
2. **TextConditioning**: Transformer-based text encoder that projects prompt embeddings into the latent space to guide image generation.
3. **VariationalAutoencoder (VAE)**: Encoder-decoder architecture that compresses images into a lower-dimensional latent space and reconstructs them, enabling faster diffusion training.
4. **DiffusionModel**: Denoising diffusion probabilistic model (DDPM) that iteratively removes noise from latents to form crisp visual structures.

### Core Technologies

- **Diffusion Models**: Add and remove noise iteratively to form crisp visual structures. The forward process gradually adds Gaussian noise to data, while the reverse process learns to denoise step-by-step.
- **Latent Space**: Compress high-dimensional image data into a compact latent representation for faster generation and training.
- **Transformers**: Understand text prompts and project them into conditioning embeddings that guide the diffusion process.
- **Variational Inference**: Enables learning of structured latent representations with the reparameterization trick.

## How Image Generation Works

1. **Text Encoding**: Convert the text prompt into token IDs and pass through the transformer text encoder to obtain a conditioning vector.
2. **VAE Encoding**: Optionally encode a reference image into a latent vector using the VAE encoder.
3. **Forward Diffusion**: Starting from random noise in latent space, iteratively denoise using the diffusion model conditioned on the text embedding.
4. **VAE Decoding**: Convert the denoised latent vector back into a pixel image using the VAE decoder.

## Applications

- **Text-to-Image**: Generate artwork, photos, or illustrations from descriptive prompts (e.g., "a cat sitting on a windowsill at sunset").
- **Image Editing**: Modify existing images by conditioning the diffusion process on a reference image and a text instruction.
- **Style Transfer**: Generate images in specific artistic styles by conditioning on style descriptors.
- **Super-Resolution**: Upsample low-resolution images by generating plausible high-frequency details.

## Training

```bash
image_generation-train --model-dir ./artifacts/models --n-samples 500 --img-size 32 --latent-dim 64 --n-diffusion-steps 1000
```

## Serving API

```bash
uvicorn image_generation.api:app --host 0.0.0.0 --port 8016
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /generate` - Generate image from text prompt
- `GET /stats` - Model statistics
- `GET /metrics` - Prometheus metrics

### Generation Parameters
- `prompt`: Text description of the desired image
- `n_steps`: Number of denoising steps (more steps = higher quality, slower)

## Dependencies
- Python >= 3.11
- NumPy, FastAPI, Pydantic
- mlops-shared
