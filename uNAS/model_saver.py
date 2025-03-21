import tensorflow as tf
import json
import os
from uNAS.config import ModelSaverConfig, BoundConfig
from uNAS.utils import NumpyEncoder, generate_nth_id
import pickle
import json
import csv
import time

class ModelSaver:
    """
    ModelSaver Class
    =============

    This class is responsible for saving the models that are generated during the search process.

    The constructor takes a ModelSaverConfig object and a BoundConfig object as arguments.

    ModelSaverConfig object:
    - save_criteria: the criteria used to save the models. It can be one of the following:
        - "all": save all models
        - "soft_pareto": save only the models that are pareto efficient
        - "boundaries": save only the models that respect the constraints
        - "pareto": save only the models that are pareto efficient and respect the constraints
        - "none": do not save any models

    BoundConfig object:
    - error_bound: the maximum error allowed for a model
    - peak_mem_bound: the maximum peak memory usage allowed for a model
    - model_size_bound: the maximum model size allowed for a model
    - mac_bound: the maximum number of MACs allowed for a model


    The class provides the following methods:
    - evaluate_and_save: saves a model if it respects the constraints
    - save_models: saves the models that were stored during the search process
    - dump: prints the ModelSaver object

    """

    stored_models = None
    save_path = None

    save_criteria = None

    error_bound = None
    peak_mem_bound = None
    model_size_bound = None
    mac_bound = None


    def __init__(self, model_saver_config: ModelSaverConfig, constraint_bounds: BoundConfig, ckpt_path = None):

        self.stored_models = []
        self.save_criteria = model_saver_config.save_criteria or "none"
        
        self.error_bound = constraint_bounds.error_bound
        self.peak_mem_bound = constraint_bounds.peak_mem_bound
        self.model_size_bound = constraint_bounds.model_size_bound
        self.mac_bound = constraint_bounds.mac_bound

        self.ckpt_path = ckpt_path

        self.iteration = 0
        print("ModelSaver initialized")

        self.created_at = time.time()

        
        base_path = os.getcwd()
        self.full_path = os.path.join(base_path, self.ckpt_path) 

        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)

        self.models_path = os.path.join(self.full_path, "models")
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

        print(self.__str__())




    def _pack_model(self, model, val_error, test_error, resource_features):
        pmu, ms, macs = resource_features

        layers_config = [layer.get_config() for layer in model.layers]

        return {"model": model, "val_error": float(val_error), "test_error": float(test_error), "pmu": int(pmu), "ms": int(ms), "macs": int(macs), "layers_config": layers_config}

    def _convert_to_tflite(model, representative_dataset, output_file):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        model_bytes = converter.convert()

        if output_file is not None:
            with open(output_file, "wb") as f:
                f.write(model_bytes)

    def _store_model(self, model_obj):
        model = model_obj.pop("model")
        model_name = "model_" + generate_nth_id(self.iteration) + ".h5"
        model.save(self.models_path + "/" + model_name)
        model_obj["model_name"] = model_name
        self.stored_models.append(model_obj)
        self.iteration += 1

        return len(self.stored_models)
    

    def _flush_models(self):
        print("Flushing models")
        pickle.dump(self.stored_models, open(self.full_path + "/temp_models.pkl", "wb"))
        

    def _check_respect_constraints(self, model_obj):
        if self.peak_mem_bound is not None and model_obj["pmu"] > self.peak_mem_bound:
            return False
        if self.model_size_bound is not None and model_obj["ms"] > self.model_size_bound:
            return False
        if self.mac_bound is not None and model_obj["macs"] > self.mac_bound:
            return False
        if self.error_bound is not None and model_obj["test_error"] > self.error_bound:
            return False

        return True

    def _pareto_dominates(self, obj : dict, candidateObj: dict):
        if all(candidateObj[x] >= obj[x] for x in ['test_error', 'pmu', 'ms', 'macs']):
            return False
        better_in_at_least_one = any(candidateObj[x] < obj[x] for x in ['test_error', 'pmu', 'ms', 'macs'])
        return better_in_at_least_one

    def _is_pareto_efficient(self, new_model_obj, models_list = None):
        if models_list is None:
            models_list = self.stored_models
        if len(models_list) == 0:
            return True

        if all((self._pareto_dominates(model, new_model_obj)) for model in models_list if model != new_model_obj):
            return True
        return False

    def _filter_pareto_efficient(self):
        pareto_efficient_objects = []
        for i in range(len(self.stored_models)):
            if all(self._pareto_dominates(self.stored_models[j], self.stored_models[i]) for j in range(len(self.stored_models)) if i != j):
                pareto_efficient_objects.append(self.stored_models[i])

        for model in self.stored_models:
            if not model in pareto_efficient_objects:
                os.remove(self.models_path + "/" + model["model_name"])
        return pareto_efficient_objects


    def evaluate_and_save(self, model, val_error, test_error, resource_features):
        print("Evaluating and saving model")

        model_obj = self._pack_model(model, val_error, test_error, resource_features)

        if self.save_criteria == "all":
            self._store_model(model_obj)
            self._flush_models()
        elif self.save_criteria == "soft_pareto":
            if self._is_pareto_efficient(model_obj):
                self._store_model(model_obj)
                self.stored_models = self._filter_pareto_efficient()
                self._flush_models()
        elif self.save_criteria == "boundaries":
            if self._check_respect_constraints(model_obj):
                self._store_model(model_obj)
                self._flush_models()
        elif self.save_criteria == "pareto":
            if self._check_respect_constraints(model_obj) and self._is_pareto_efficient(model_obj):
                self._store_model(model_obj)                
                self.stored_models = self._filter_pareto_efficient()
                self._flush_models()
        elif self.save_criteria == "none":
            pass
        else:
            raise ValueError("Invalid save criteria")
        print("Stored models: ", len(self.stored_models))

    def save_models(self):

        if self.save_criteria == "none":
            print("No models to save")
            return

        print("Saving models")

        if os.path.exists(self.full_path + "/temp_models.pkl"):
            stored_models = pickle.load(open(self.full_path + "/temp_models.pkl", "rb"))
        else:
            print("It was not possible to load the temp_models.pkl file")
            return


        if len(stored_models) == 0:
            return

        metadatas = []


        for stored_model in stored_models:
            is_pareto = self._is_pareto_efficient(stored_model, stored_models)
            stored_model["is_pareto"] = is_pareto
            metadatas.append(stored_model)

        json.dump(metadatas, open(self.full_path + "/metadatas.json", "w"),cls=NumpyEncoder, indent=4)

        metadatas = [{k: v for k, v in metadata.items() if k != 'layers_config'} for metadata in metadatas]

        with open(self.full_path + "/metadatas.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metadatas[0].keys())
            writer.writeheader()
            writer.writerows(metadatas)

        os.remove(self.full_path + "/temp_models.pkl")

        print(f"Models saved to {self.full_path}")
        return self.stored_models
    

    def __str__(self):
        return f"ModelSaver(save_criteria={self.save_criteria}, save_path={self.full_path}), created_at={self.created_at})"
    
    def dump(self):
        print(self.__str__())
        print("number of models saved: ", len(self.stored_models))

        





        