import numpy as np
import tensorflow as tf
import random

from typing import Tuple, Optional
from uNAS.dataset import Dataset


class ImageNetDataset(Dataset):
    """
    ImageNet Dataset for uNAS.
    ==========================

    Loads ImageNet (or any ImageNet-compatible directory layout) from disk using
    the standard folder structure:

        data_dir/
            train/
                n01440764/          # synset / class folder
                    ILSVRC2012_val_00000293.JPEG
                    ...
                n01443537/
                    ...
            val/
                n01440764/
                    ...
                ...

    The validation split is reused as the test set because ImageNet test labels
    are not publicly available.

    Because ImageNet is very large, this class performs **lazy loading**: images
    are read from disk on the fly and never fully resident in RAM.  Use it with
    ``serialized_dataset=True`` in ``TrainingConfig`` so that each Ray worker
    instantiates its own copy instead of trying to pickle the whole object:

        training_config = TrainingConfig(
            dataset=partial(ImageNetDataset, data_dir="/path/to/imagenet"),
            ...
            serialized_dataset=True,
        )

    Args:
        data_dir:    Root directory that contains ``train/`` and ``val/``
                     sub-directories.
        image_size:  (H, W) to which every image is resized.  Defaults to
                     (96, 96) – a common size for MCU-targeted models.
        num_classes: Number of classes to expose.  Must be ≤ the number of
                     class directories present.  Defaults to 1000.
        fix_seeds:   If ``True``, fix NumPy / TF / Python random seeds to 42
                     for reproducibility.
    """

    NUM_IMAGENET_CLASSES = 1000

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (96, 96),
        num_classes: int = 1000,
        fix_seeds: bool = False,
    ):
        if fix_seeds:
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        if num_classes < 2 or num_classes > self.NUM_IMAGENET_CLASSES:
            raise ValueError(
                f"num_classes must be between 2 and {self.NUM_IMAGENET_CLASSES}, "
                f"got {num_classes}."
            )

        self._data_dir = data_dir
        self._image_size = image_size
        self._num_classes = num_classes
        self._input_shape = image_size + (3,)

        # Training augmentation pipeline (applied element-wise on unbatched data)
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

    def _normalize(self, image: tf.Tensor) -> tf.Tensor:
        """Scale pixel values from [0, 255] to [0, 1]."""
        return tf.cast(image, tf.float32) / 255.0

    def _load_split(self, split: str, shuffle: bool) -> tf.data.Dataset:
        """Return an **unbatched** ``tf.data.Dataset`` for the given split."""
        directory = f"{self._data_dir}/{split}"
        ds = tf.keras.utils.image_dataset_from_directory(
            directory=directory,
            labels="inferred",
            label_mode="int",
            class_names=None,          # infer from sub-directory names
            image_size=self._image_size,
            batch_size=None,           # return individual images (not batched)
            shuffle=shuffle,
            interpolation="bilinear",
        )
        return ds

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def train_dataset(self) -> tf.data.Dataset:
        ds = self._load_split("train", shuffle=True)
        ds = ds.map(
            lambda img, lbl: (
                self._augmentation(
                    self._normalize(img), training=True
                ),
                lbl,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds

    def validation_dataset(self) -> tf.data.Dataset:
        ds = self._load_split("val", shuffle=False)
        ds = ds.map(
            lambda img, lbl: (self._normalize(img), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds

    def test_dataset(self) -> tf.data.Dataset:
        # ImageNet test labels are not publicly available;
        # the validation set is used as a proxy.
        return self.validation_dataset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------

    def class_weight(self):
        """
        Return uniform class weights.

        Iterating over all ~1.2 M ImageNet images to compute balanced weights
        (as the base-class implementation does) would be prohibitively slow.
        ImageNet is already roughly balanced across its 1 000 classes, so
        uniform weights are a safe default.  Enable ``use_class_weight`` in
        ``TrainingConfig`` only if you are working with a heavily imbalanced
        ImageNet subset and are willing to pay the extra pre-processing cost
        (in that case override this method).
        """
        return np.ones(self._num_classes, dtype=np.float32)
