"""
uNAS experiment config for Mini-ImageNet (timm/mini-imagenet on HuggingFace).
=============================================================================

100-class subset of ImageNet-1k with 50 k train / 10 k validation / 5 k test
images at their original resolutions.

The dataset is downloaded automatically from HuggingFace on first use.
Provide a local cache directory via -d/--data-dir or 'data_dir' in params.json
to control where it is stored (defaults to ~/.cache/huggingface/datasets).

Usage::

    # Download to default HF cache and run search
    python driver.py -c mini_imagenet

    # Use a custom HF cache directory
    python driver.py -c mini_imagenet -d /data/hf_cache

    # Tighten resource constraints for a smaller MCU
    python driver.py -c mini_imagenet --peak-mem-bound 256000 --model-size-bound 256000

"""

import tensorflow as tf
from functools import partial

from uNAS.config import (
    TrainingConfig,
    AgingEvoConfig,
    BoundConfig,
    ModelSaverConfig,
)
from uNAS.cnn2d import Cnn2DSearchSpace
from uNAS.search_algorithms import AgingEvoSearch

from dataset.mini_imagenet_dataset import MiniImageNetDataset

# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

IMAGE_SIZE = (96, 96)   # (H, W); images are resized from their original size


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def get_mini_imagenet_setup(
    image_size: tuple = IMAGE_SIZE,
    data_dir: str = None,
    batch_size: int = 128,
    fix_seeds: bool = False,
    error_bound: float = 0.60,
    peak_mem_bound: int = 512_000,
    model_size_bound: int = 512_000,
    mac_bound: int = 5_000_000,
):
    """Return the full uNAS setup dictionary for a Mini-ImageNet search."""
    return {
        "config": get_mini_imagenet_config(
            image_size=image_size,
            data_dir=data_dir,
            batch_size=batch_size,
            fix_seeds=fix_seeds,
            error_bound=error_bound,
            peak_mem_bound=peak_mem_bound,
            model_size_bound=model_size_bound,
            mac_bound=mac_bound,
        ),
        "name": "mini_imagenet_cnn2d",
        "load_from": None,
        "save_every": 5,
        "seed": 0,
    }


def get_mini_imagenet_config(
    image_size: tuple = IMAGE_SIZE,
    data_dir: str = None,
    batch_size: int = 128,
    fix_seeds: bool = False,
    error_bound: float = 0.60,
    peak_mem_bound: int = 512_000,
    model_size_bound: int = 512_000,
    mac_bound: int = 5_000_000,
):
    training_config = TrainingConfig(
        # Use partial() + serialized_dataset=True so each Ray worker
        # creates its own MiniImageNetDataset instance rather than pickling
        # a large generator-backed tf.data.Dataset.
        dataset=partial(
            MiniImageNetDataset,
            image_size=image_size,
            data_dir=data_dir,
            fix_seeds=fix_seeds,
        ),
        optimizer="adam",
        batch_size=batch_size,
        epochs=30,
        callbacks=lambda: [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                min_delta=0.001,
                verbose=1,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-6,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
        use_class_weight=False,   # mini-imagenet is balanced across 100 classes
        serialized_dataset=True,  # lazy per-worker instantiation
        use_qat=True,
    )

    search_algorithm = AgingEvoSearch

    search_config = AgingEvoConfig(
        search_space=Cnn2DSearchSpace(),
        serialized_dataset=True,
        checkpoint_dir="artifacts/mini_imagenet_cnn2d",
        max_parallel_evaluations=1,
        rounds=2000,
    )

    # Resource bounds tuned for a typical Cortex-M MCU with 512 KB SRAM/Flash.
    bound_config = BoundConfig(
        error_bound=error_bound,
        peak_mem_bound=peak_mem_bound,
        model_size_bound=model_size_bound,
        mac_bound=mac_bound,
    )

    model_saver_config = ModelSaverConfig(
        save_criteria="all",
    )

    return {
        "training_config": training_config,
        "bound_config": bound_config,
        "search_algorithm": search_algorithm,
        "search_config": search_config,
        "model_saver_config": model_saver_config,
    }
