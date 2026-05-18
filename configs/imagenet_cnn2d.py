"""
ImageNet NAS search configuration for uNAS.
============================================

This configuration searches for efficient 2-D CNN architectures that can
classify ImageNet images while fitting within microcontroller-grade resource
budgets.

Typical MCU deployment targets (adjust to your hardware):
  - Peak SRAM usage  : ≤ 512 KB
  - Flash / model size: ≤ 512 KB
  - MACs              : ≤ 5 M

Because ImageNet is large (~1.2 M training images), the dataset is loaded
lazily with ``serialized_dataset=True`` so that each Ray worker opens its own
file handles rather than attempting to pickle TensorFlow dataset objects.

Directory layout expected by ImageNetDataset:

    /path/to/imagenet/
        train/
            n01440764/
                ILSVRC2012_train_*.JPEG
                ...
            ...
        val/
            n01440764/
                ILSVRC2012_val_*.JPEG
                ...
            ...

Usage
-----
Edit ``IMAGENET_DIR`` and ``IMAGE_SIZE`` below, then run::

    python driver.py  # after pointing driver.py to get_imagenet_setup()

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

from dataset.imagenet_dataset import ImageNetDataset

# ---------------------------------------------------------------------------
# User-configurable parameters – edit these before running a search
# ---------------------------------------------------------------------------

IMAGENET_DIR = "/path/to/imagenet"   # root dir containing train/ and val/
IMAGE_SIZE = (96, 96)                # (H, W); use (64, 64) for tighter budgets


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def get_imagenet_setup(
    image_size: tuple = IMAGE_SIZE,
    data_dir: str = IMAGENET_DIR,
    batch_size: int = 256,
    num_classes: int = 1000,
    fix_seeds: bool = False,
):
    """Return the full uNAS setup dictionary for an ImageNet search."""
    return {
        "config": get_imagenet_config(
            image_size=image_size,
            data_dir=data_dir,
            batch_size=batch_size,
            num_classes=num_classes,
            fix_seeds=fix_seeds,
        ),
        "name": "imagenet_cnn2d",
        "load_from": None,
        "save_every": 5,
        "seed": 0,
    }


def get_imagenet_config(
    image_size: tuple = IMAGE_SIZE,
    data_dir: str = IMAGENET_DIR,
    batch_size: int = 256,
    num_classes: int = 1000,
    fix_seeds: bool = False,
):
    """Return the uNAS config dictionary for an ImageNet NAS search."""

    training_config = TrainingConfig(
        # Use partial() + serialized_dataset=True so each Ray worker
        # creates its own ImageNetDataset instance (avoids pickling issues
        # with large tf.data.Dataset objects).
        dataset=partial(
            ImageNetDataset,
            data_dir=data_dir,
            image_size=image_size,
            num_classes=num_classes,
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
        use_class_weight=False,   # ImageNet is roughly balanced
        serialized_dataset=True,  # lazy per-worker instantiation
        use_qat=True,                # quantization-aware training 
    )

    search_algorithm = AgingEvoSearch

    search_config = AgingEvoConfig(
        search_space=Cnn2DSearchSpace(),
        serialized_dataset=True,
        checkpoint_dir="artifacts/imagenet_cnn2d",
        max_parallel_evaluations=1,
        rounds=2000,
    )

    # Resource bounds tuned for a typical Cortex-M MCU with 512 KB SRAM/Flash.
    # Loosen or tighten depending on your target device.
    bound_config = BoundConfig(
        error_bound=0.70,         # top-1 error ≤ 70 % (top-1 acc ≥ 30 %)
        peak_mem_bound=512_000,   # ≤ 512 KB peak SRAM (bytes)
        model_size_bound=512_000, # ≤ 512 KB model size (bytes)
        mac_bound=5_000_000,      # ≤ 5 M multiply-accumulate ops
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
        "serialized_dataset": True,
    }
