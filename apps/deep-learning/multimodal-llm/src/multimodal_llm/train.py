"""Training pipeline for Multimodal Language Modeling."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from multimodal_llm.data import (
    VOCAB_SIZE,
    generate_synthetic_multimodal_data,
    save_multimodal_data,
    train_test_split_multimodal,
)
from multimodal_llm.model import MultimodalLLM

logger = get_logger(__name__)


def train(
    model_dir: Path,
    n_samples: int = 500,
    vocab_size: int = VOCAB_SIZE,
    seq_len: int = 64,
    d_model: int = 256,
    text_encoder_dim: int = 256,
    image_encoder_dim: int = 768,
    audio_encoder_dim: int = 80,
    connector_dim: int = 512,
    fusion_type: str = "hybrid",
    max_seq_len: int = 128,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 2,
    d_ff: int = 512,
    learning_rate: float = 0.001,
    n_iterations: int = 100,
    weight_decay: float = 0.01,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
    include_image: bool = True,
    include_audio: bool = True,
) -> dict:
    logger.info("Generating multimodal training data", n_samples=n_samples, include_image=include_image, include_audio=include_audio)
    data = generate_synthetic_multimodal_data(
        n_samples=n_samples,
        vocab_size=vocab_size,
        seq_len=seq_len,
        random_seed=random_seed,
        include_image=include_image,
        include_audio=include_audio,
    )

    train_data, test_data = train_test_split_multimodal(data, test_size=test_size, random_seed=random_seed)
    logger.info("Data split", n_train=len(train_data["text_tokens"]), n_test=len(test_data["text_tokens"]))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_multimodal_data(data, model_dir / "training_data.npz")

    X_train_text = train_data["text_tokens"]
    y_train = train_data["text_targets"]
    X_train_image = train_data.get("image_patches")
    X_train_audio = train_data.get("mel_spectrogram")

    model = MultimodalLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        text_encoder_dim=text_encoder_dim,
        image_encoder_dim=image_encoder_dim,
        audio_encoder_dim=audio_encoder_dim,
        connector_dim=connector_dim,
        fusion_type=fusion_type,
        max_seq_len=max_seq_len,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train_text, y_train, image_patches=X_train_image, mel_spectrogram=X_train_audio)

    X_test_text = test_data["text_tokens"]
    y_test = test_data["text_targets"]
    X_test_image = test_data.get("image_patches")
    X_test_audio = test_data.get("mel_spectrogram")

    model.predict(X_test_text[:5], image_patches=X_test_image[:5] if X_test_image is not None else None, mel_spectrogram=X_test_audio[:5] if X_test_audio is not None else None)
    test_metrics = model.evaluate(X_test_text, y_test)
    logger.info("Training complete", training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f"multimodal_llm_v{model_version}.npz"
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        "training_mode": "supervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_train_samples": float(len(X_train_text)),
        "n_test_samples": float(len(X_test_text)),
        "vocab_size": float(vocab_size),
        "d_model": float(d_model),
        "connector_dim": float(connector_dim),
        "fusion_type": fusion_type,
        "n_encoder_layers": float(n_encoder_layers),
        "n_decoder_layers": float(n_decoder_layers),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="multimodal-llm",
        model_version=model_version,
        model_type="classification",
        metrics=metrics,
        parameters={
            "vocab_size": vocab_size,
            "d_model": d_model,
            "text_encoder_dim": text_encoder_dim,
            "image_encoder_dim": image_encoder_dim,
            "audio_encoder_dim": audio_encoder_dim,
            "connector_dim": connector_dim,
            "fusion_type": fusion_type,
            "max_seq_len": max_seq_len,
            "n_encoder_layers": n_encoder_layers,
            "n_decoder_layers": n_decoder_layers,
            "d_ff": d_ff,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
            "include_image": include_image,
            "include_audio": include_audio,
        },
        artifacts={
            f"multimodal_llm_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "multimodal_llm", "model_type": "MLLM"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="multimodal-llm",
            model_version=model_version,
            metrics=metrics,
            params={"vocab_size": vocab_size, "d_model": d_model, "fusion_type": fusion_type, "n_iterations": n_iterations},
            artifacts={"model": str(model_path)},
            tags={"model_type": "multimodal_llm", "framework": "numpy"},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal LLM")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--vocab-size", type=int, default=int(os.getenv("VOCAB_SIZE", str(VOCAB_SIZE))))
    parser.add_argument("--seq-len", type=int, default=int(os.getenv("SEQ_LEN", "64")))
    parser.add_argument("--d-model", type=int, default=int(os.getenv("D_MODEL", "256")))
    parser.add_argument("--text-encoder-dim", type=int, default=int(os.getenv("TEXT_ENCODER_DIM", "256")))
    parser.add_argument("--image-encoder-dim", type=int, default=int(os.getenv("IMAGE_ENCODER_DIM", "768")))
    parser.add_argument("--audio-encoder-dim", type=int, default=int(os.getenv("AUDIO_ENCODER_DIM", "80")))
    parser.add_argument("--connector-dim", type=int, default=int(os.getenv("CONNECTOR_DIM", "512")))
    parser.add_argument("--fusion-type", type=str, default=os.getenv("FUSION_TYPE", "hybrid"), choices=["early", "late", "hybrid"])
    parser.add_argument("--max-seq-len", type=int, default=int(os.getenv("MAX_SEQ_LEN", "128")))
    parser.add_argument("--n-encoder-layers", type=int, default=int(os.getenv("N_ENCODER_LAYERS", "2")))
    parser.add_argument("--n-decoder-layers", type=int, default=int(os.getenv("N_DECODER_LAYERS", "2")))
    parser.add_argument("--d-ff", type=int, default=int(os.getenv("D_FF", "512")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.001")))
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "100")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.01")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--test-size", type=float, default=float(os.getenv("TEST_SIZE", "0.2")))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument("--register-mlflow", action="store_true", default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true")
    parser.add_argument("--no-image", action="store_true", help="Disable image modality")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio modality")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        n_samples=args.n_samples,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        text_encoder_dim=args.text_encoder_dim,
        image_encoder_dim=args.image_encoder_dim,
        audio_encoder_dim=args.audio_encoder_dim,
        connector_dim=args.connector_dim,
        fusion_type=args.fusion_type,
        max_seq_len=args.max_seq_len,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
        include_image=not args.no_image,
        include_audio=not args.no_audio,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
