# Video Generation

Video generation is a generative AI task that produces short video clips from text descriptions or animates static images using motion prompts. It extends image generation to the temporal dimension, modeling both spatial details and temporal dynamics across frames.

## Network Type
Spatiotemporal Diffusion Model with Latent Video Compression

## Architecture

### Key Components

1. **VideoTokenizer**: Encodes text prompts and motion descriptions into token IDs for conditioning video generation.
2. **TextConditioning**: Transformer-based text encoder that projects prompts into latent conditioning vectors.
3. **LatentVideoEncoder**: VAE-style encoder-decoder that compresses video frame sequences into compact latent representations for efficient diffusion.
4. **SpatiotemporalDiffusionModel**: Denoising diffusion process operating in video latent space, conditioned on text or image inputs.

### Core Technologies

- **Diffusion Models**: Add and remove noise iteratively to form crisp visual structures across video frames.
- **Latent Space**: Compress video data (multiple frames) into lower-dimensional latents for faster processing and generation.
- **Transformers**: Understand text prompts and sequence video frames over time, enabling coherent temporal generation.
- **Image-to-Video**: Animate static photos by conditioning diffusion on a start frame and a motion prompt.

## How Video Generation Works

### Text-to-Video
1. **Text Encoding**: Convert the text prompt into token IDs and pass through the text encoder to obtain a conditioning vector.
2. **Latent Initialization**: Sample random noise in the video latent space.
3. **Reverse Diffusion**: Iteratively denoise the latent, conditioned on the text embedding, to produce a coherent video latent.
4. **Decode**: Convert the denoised latent back into a sequence of video frames.

### Image-to-Video
1. **Image Encoding**: Encode the start frame into a latent using the VAE encoder.
2. **Motion Conditioning**: Combine the image latent with the motion prompt embedding.
3. **Reverse Diffusion**: Denoise from a mixture of random noise and the image latent, guided by the motion prompt.
4. **Decode**: Generate the animated video frames.

## Applications

- **Text-to-Video**: Generate short cinematic clips or animations from a script (e.g., "a cat jumping over a fence in slow motion").
- **Image-to-Video**: Animate static photos by defining a start frame and motion prompt (e.g., "make the waves crash on the shore").
- **Video Inpainting**: Fill in missing video segments conditioned on surrounding context.
- **Style Transfer**: Apply artistic styles to video sequences with temporal consistency.

## Training

```bash
video_generation-train --model-dir ./artifacts/models --n-samples 200 --img-size 32 --n-frames 8 --latent-dim 64 --n-diffusion-steps 1000
```

## Serving API

```bash
uvicorn video_generation.api:app --host 0.0.0.0 --port 8017
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /generate` - Generate video from text prompt (text-to-video)
- `POST /animate` - Animate a static image with a motion prompt (image-to-video)
- `GET /stats` - Model statistics
- `GET /metrics` - Prometheus metrics

### Generation Parameters
- `prompt`: Text description of the desired video
- `n_steps`: Number of denoising steps (more steps = higher quality, slower)
- `mode`: Generation mode (`text-to-video` or `image-to-video`)

## Dependencies
- Python >= 3.11
- NumPy, FastAPI, Pydantic
- mlops-shared
