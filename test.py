# Load and evaluate .h5 or .tflite model on the test set
import os
import tensorflow as tf
# import tf_keras as tk
import numpy as np
from tqdm import tqdm
import tensorflow_model_optimization as tfmot

# Toggle this to switch between evaluating the .h5 model and the .tflite model
USE_TFLITE = True

# Paths to models
model_path = "/home/pigo/uNAS/abh_quality.h5"
# model_path = "/home/pigo/uNAS/0_wv_k_8_c_5.tflite"

# Some hyperparameters
img_size = (50, 50)
batch_size = 1

# Define paths to your dataset
test_dir = "wake_vision/test"
train_dir = "wake_vision/train_quality"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",  # For binary classification
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    seed=123,
)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",  # For binary classification
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    seed=123
)
test_ds = test_ds.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

if not USE_TFLITE:
    ### EVALUATE .H5 MODEL ###
    print("\nEvaluating H5 model...")
    # model = tk.models.load_model(model_path)
    with tfmot.quantization.keras.quantize_scope():
        model = tf.keras.models.load_model(model_path)
    model.evaluate(test_ds)
else:
    ### EVALUATE .TFLITE MODEL ###
    print("\nEvaluating TFLite model...")    
    with tfmot.quantization.keras.quantize_scope():
        model = tf.keras.models.load_model(model_path)
    
    model_name = "quant_aaaabh_quality"
    
    def representative_dataset():
        for data in train_ds.rebatch(1).take(150) :
            yield [tf.dtypes.cast(data[0], tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    with open(model_name + ".tflite", 'wb') as f:
        f.write(tflite_model)
        
    #Test quantized model
    interpreter = tf.lite.Interpreter(model_name + '.tflite')
    interpreter.allocate_tensors()

    output = interpreter.get_output_details()[0]  # Model has single output.
    input = interpreter.get_input_details()[0]  # Model has single input.
    print("Input shape:", input['shape'])
    print("Input type", type(input))

    correct = 0
    wrong = 0
    
    accuracy = tf.keras.metrics.BinaryAccuracy()
    
    for image, label in tqdm(test_ds):
        # Check if the input type is quantized, then rescale input data to uint8
        if input['dtype'] == tf.uint8:
            input_scale, input_zero_point = input["quantization"]
            image = image / input_scale + input_zero_point
            image = tf.dtypes.cast(image, tf.uint8)

        interpreter.set_tensor(input['index'], image)
        interpreter.invoke()

        scaled_output = interpreter.get_tensor(output['index']) / 255.
        predictions = 1 if (scaled_output >= .5) else 0
        if label.numpy() == predictions:
            correct = correct + 1
        else:
            wrong = wrong + 1
        
        accuracy.update_state(label, scaled_output)
            
    print(f"\n\nTflite model test accuracy (AMG): {correct/(correct+wrong)}\n")
    print(f"Tflite model test accuracy: {accuracy.result().numpy()}\n\n")