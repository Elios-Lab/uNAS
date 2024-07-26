import numpy as np
import tensorflow as tf


from uNAS.architecture import Architecture
from uNAS.cnn2d import Cnn2DSearchSpace
from uNAS.cnn1d import Cnn1DSearchSpace
from uNAS.mlp import MlpSearchSpace
from uNAS.resource_models.models import model_size, peak_memory_usage


search_space_options = [MlpSearchSpace, Cnn1DSearchSpace, Cnn2DSearchSpace]

def main():
    np.random.seed(0)

    num_models = 10
    output_dir = ".\\tmp\\tflite"

    report = []

    choice = int(input("Input the search space to use (0: MLP, 1: CNN1D, 2: CNN): "))
    verbose = int(input("Input the verbosity level (0: None, 1: Low, 2: High): "))
    save_report = int(input("Save a report of the generated models? (0: No, 1: Yes): ")) == 1
    ms_req = int(input("Input the model size requirement (in bytes): "))
    pmu_req = int(input("Input the peak memory usage requirement (in bytes): "))


    ss = search_space_options[choice]()

    input_shape = (64, 64, 3) if choice == 2 else (64, 1) if choice == 1 else (28, 28, 1)
    num_classes = 10
    #ms_req, pmu_req = 250_000, 250_000

    def get_resource_requirements(arch: Architecture):
        rg = ss.to_resource_graph(arch, input_shape, num_classes)
        return model_size(rg), peak_memory_usage(rg, exclude_inputs=False)

    def evolve_until_within_req(arch):
        keep_prob = 0.25
        ms, pmu = get_resource_requirements(arch)

        i = 0
        while ms > ms_req or pmu > pmu_req:
            morph = np.random.choice(ss.produce_morphs(arch))
            new_ms, new_pmu = get_resource_requirements(morph)

            i +=1

            if new_ms < ms or new_pmu < pmu: # or np.random.random_sample() < keep_prob: # save only morph within bounds
                ms, pmu = new_ms, new_pmu
                arch = morph
                if verbose > 1:
                    print("---------------------------------------------------------")
                    print(f"Accepted morph {i}")
                    print(f"Current model size: {ms}, peak memory usage: {pmu}")
                    print("Current architecture: ", arch.architecture)
            else:
                if verbose > 1:
                    print("---------------------------------------------------------")
                    print(f"Rejected morph {i}")
                    print(f"Current model size: {ms}, peak memory usage: {pmu}")
                    print("Current architecture: ", arch.architecture)

        if verbose > 1:
            print("---------------------------------------------------------")
            print(f"Final model {i}, size: {ms}, peak memory usage: {pmu}")
            print("Final architecture: ", arch.architecture)
        
        if save_report:
            report.append({"model_size":float(ms), "peak_memory_usage": float(pmu), "architecture": arch.architecture})
        return arch

    def convert_to_tflite(arch: Architecture, output_file):
        model = ss.to_keras_model(arch, input_shape, num_classes)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = \
            lambda: [[np.random.random((1,) + input_shape).astype("float32")] for _ in range(5)]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        model_bytes = converter.convert()

        if output_file is not None:
            with open(output_file, "wb") as f:
                f.write(model_bytes)
            if verbose > 0:
                print(f"Model saved to {output_file}")

    for i in range(num_models):
        print(f"Generating #{i + 1}...")
        arch = evolve_until_within_req(ss.random_architecture())
        convert_to_tflite(arch, output_file=f"{output_dir}/m{i:05d}.tflite")

        if verbose > 0:
            print(f"Model {i + 1} generated.")

    if save_report:
        with open(f"{output_dir}/report.json", "w") as f:
            import json
            json.dump(report, f, indent=4)
        if verbose > 0:
            print(f"Report saved to {output_dir}/report.json")


if __name__ == '__main__':
    main()
