from tensorflow_model_optimization.python.core.keras.compat import keras #for Quantization Aware Training (QAT)
import tensorflow_model_optimization as tfmot #for Post Training Quantization (PTQ)
import os
import random
import numpy as np
import tensorflow as tf #for designing and training the model


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3500)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

#some hyperparameters
img_size = (50, 50)
batch_size = 64
learning_rate = 0.0001
epochs = 100
model_name = "abh_quality"
model_path = "/home/pigo/uNAS/model_aaaabh.h5"

# Define paths to your dataset
train_dir = "wake_vision/train_quality"
validation_dir = "wake_vision/validation"
test_dir = "wake_vision/test"

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",  # For binary classification
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    seed=42,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels="inferred",
    label_mode="binary",  # For binary classification
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    seed=42,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",  # For binary classification
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    seed=42,
)

data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomContrast(0.2),
        ])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


#set validation based early stopping
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".h5",
    monitor='val_accuracy',
    mode='max', save_best_only=True)


early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    mode='max',
    restore_best_weights=True
)

with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(model_path)
    config = model.get_config()

    qa_model = keras.Model.from_config(config)
    qa_model.summary()
    qa_model.set_weights(model.get_weights())


# Compile the model
qa_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

try:
    # Train the model
    history = qa_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[model_checkpoint_callback, early_stopping_callback]
    )
except KeyboardInterrupt:
    print("\n\nTraining interrupted\n\n")
    qa_model.save(model_name + ".h5")
    
print("\n\nTraining completed\n\n")