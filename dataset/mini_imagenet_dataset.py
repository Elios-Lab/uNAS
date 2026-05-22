import numpy as np
import tensorflow as tf
import random

from typing import Tuple, Optional

from uNAS.dataset import Dataset


class MiniImageNetDataset(Dataset):
    """
    HuggingFace ``timm/mini-imagenet`` dataset for uNAS.
    =====================================================

    A 100-class subset of ImageNet-1k containing the original images at their
    original resolutions.

    Splits
    ------
    train       50 000 samples (from ImageNet-1k train)
    validation  10 000 samples (from ImageNet-1k train)
    test         5 000 samples (from ImageNet-1k validation, 50/class)

    The dataset is downloaded and cached automatically via the HuggingFace
    ``datasets`` library on the first call.  No ``data_dir`` is required; if
    provided it is used as the HuggingFace cache root (overriding the default
    ``~/.cache/huggingface/datasets``).

    Use with ``serialized_dataset=True`` in ``TrainingConfig`` so that each Ray
    worker creates its own instance and download/cache state is not pickled::

        training_config = TrainingConfig(
            dataset=partial(MiniImageNetDataset, image_size=(96, 96)),
            ...
            serialized_dataset=True,
        )

    Args:
        image_size:  (H, W) to resize every image to.  Defaults to (96, 96).
        data_dir:    Optional path to use as the HuggingFace cache root.
                     ``None`` means the default HF cache is used.
        fix_seeds:   If ``True``, fix NumPy / TF / Python random seeds to 42
                     for reproducibility.
    """

    NUM_CLASSES = 100

    def __init__(
        self,
        image_size: Tuple[int, int] = (96, 96),
        data_dir: Optional[str] = None,
        fix_seeds: bool = False,
    ):
        if fix_seeds:
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        self._image_size = image_size
        self._cache_dir = data_dir          # forwarded to HuggingFace as cache_dir
        self._input_shape = image_size + (3,)

        self._augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ],
            name="train_augmentation",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hf_split(self, split: str):
        """Download (or load from cache) one split of timm/mini-imagenet."""
        from datasets import load_dataset  # lazy import – not required at package level
        return load_dataset(
            "timm/mini-imagenet",
            split=split,
            cache_dir=self._cache_dir,
        )

    def _to_tf_dataset(self, hf_ds, shuffle: bool) -> tf.data.Dataset:
        """Convert a HuggingFace Dataset to an unbatched tf.data.Dataset."""
        h, w = self._image_size

        def generator():
            for sample in hf_ds:
                # PIL Image → uint8 numpy array (H×W×3)
                img = np.array(sample["image"].convert("RGB"), dtype=np.uint8)
                yield img, sample["label"]

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                tf.TensorSpec(shape=(), dtype=tf.int64),
            ),
        )

        # Resize + normalize to [0, 1]
        ds = ds.map(
            lambda img, lbl: (
                tf.image.resize(tf.cast(img, tf.float32) / 255.0, [h, w]),
                lbl,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if shuffle:
            ds = ds.shuffle(buffer_size=10_000, reshuffle_each_iteration=True)

        return ds

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def train_dataset(self) -> tf.data.Dataset:
        hf_ds = self._load_hf_split("train")
        ds = self._to_tf_dataset(hf_ds, shuffle=True)
        ds = ds.map(
            lambda img, lbl: (self._augmentation(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds

    def validation_dataset(self) -> tf.data.Dataset:
        hf_ds = self._load_hf_split("validation")
        return self._to_tf_dataset(hf_ds, shuffle=False)

    def test_dataset(self) -> tf.data.Dataset:
        hf_ds = self._load_hf_split("test")
        return self._to_tf_dataset(hf_ds, shuffle=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return self.NUM_CLASSES

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------

    def class_weight(self):
        """Uniform weights — mini-imagenet is balanced across its 100 classes."""
        return np.ones(self.NUM_CLASSES, dtype=np.float32)
