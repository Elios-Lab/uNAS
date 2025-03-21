# μNAS

μNAS (micro-NAS or mu-NAS) is a neural architecture search system that specialises in finding
 ultra-small models suitable for deploying on microcontrollers: think < 64 KB memory and storage
 requirement. μNAS achieves this by explicitly targeting three primary resource bottlenecks:
 model size, latency and peak memory usage.

For a full description of methodology and experimental results, please see the accompanying paper
 [_"μNAS: Constrained Neural Architecture Search for Microcontrollers"_](https://arxiv.org/abs/2010.14246). 
 
*Changelog from arXiv v1:* 

* correctly reported the number of MACs for the DS-CNN baseline for the Speech Commands dataset.
* fixed Speech Commands hyperparameters and updated found models
* add smaller CIFAR-10 model in the comparison table
* add search times to the comparison table
* update discussion on pruning, search convergence and the use of soft constraints

*Changelog from original implementation:*

* 1D Convolutional Neural Networks are now supported
* 2D Convolutional Neural Networks contain **hard-coded** (WIP) quantisation aware training (QAT) layer definition, as well as input and output model quantisation
* Dynamic dataset loading is now supported and tested through the TensorFlow Dataset API
* Output format is Keras 2 model (.h5 format), `test.py` includes the conversion to TF Lite format
 
 
## Usage 
 
### Setup
 
μNAS uses Python 3.7+ with the environment described by `requirements.txt`. To install a specific version of TensorFlow, modify it and run `pip install -r requirements.txt`.

*Changelog*

The current version of μNAS has been deployed and tested for **Tensorflow 2.18.0** and **Python 3.10**. If the output models are not compatible with [ST developer tools](https://stm32ai-cs.st.com/home) even after TFlite conversion, please downgrade to lower Python and Tensorflow versions.

- `configs`: example search configurations,

- `dataset`: loaders for various datasets, conforming to the interface in `dataset/cnn_wakeviz.py`

- `search_state_processor.py`: loads and visualises μNAS search state files.

- `train.py`/`test.py`: Helper scripts for continuining the training phase of the chosen model, converting it to TFlite and evaluating it.

### Pipeline

The μNAS pipeline consists of three main steps:

1. Defining a search configuration script in the `configs` folder. This script specifies the search algorithm, training parameters, and resource constraints. Other task examples in the folder can be used for reference.

2. Creating a dataset class in the `dataset` folder. This Python script should handle loading and preprocessing the data, either dynamically or statically, conforming to the interface in `dataset/dataset.py`.

3. Executing the `driver.py` script to initiate the algorithmic search. This script uses the defined configuration and dataset to begin the neural architecture search process.

## Additional Tools and File description (internal μNAS directory)

- `cnn1d`/`cnn2d`/`mlp`: contains a search space description for convolutional neural networks / multilayer
 perceptrons, together with all allowed morphisms (changes) to a candidate architecture.

- `dragonfly_adapters`: (Bayesian optimisation only) extra code to interoperate with 
[Dragonfly](https://github.com/dragonfly/dragonfly). We found that we had to rely on internal
 implementation of the framework for it to correctly use our customised kernel, search space and
 a genetic algorithm optimiser for acq. functions, thus the module contains a fair amount of
  monkey-patches.
  
- `resource_models`: an independent library that allows representing and computing resource usage
 of arbitrary computation graphs.
 
- `search_algorithms`: implements aging evolution and Bayesian optimisation search algorithms;
 each search algorithm is also responsible for scheduling model training and correctly
 serialising & restoring the search state. Both use `ray` under the hood to parallelise the search.
 
- `teachers`: a collection of teacher models for distillation.

- `model_trainer.py`: code for training candidate models.

- `pruning.py`: implements [Dynamic Model Pruning with Feedback](https://openreview.net/forum?id=SJem8lSFwB)
 as a Keras callback, used during training.

- `generate_tflite_models.py`: generates random small models for latency benchmarking on a
 microcontroller.
 
- `architecture.py`/`config.py`/`search_space.py`/`schema_types.py` base classes for candidate
 architectures, search configuration and free variables of the search space.


## Notes on deploying found models

In the interest of storage, μNAS does not save final weights of discovered models (though
 it can be modified to do so): μNAS uses aging evolution and does not share trained weights
 across candidate models, which encourages finding models that can be trained to good accuracy
 from scratch. You can easily instantiate a Keras model from a found architecture (see
  API in `architecture.py`).
 
 μNAS assumes a runtime where each operator is executed one
 at a time and in full, such as ["TensorFlow Lite Micro"](https://www.tensorflow.org/lite/microcontrollers). 
 You can quantise and convert Keras models to the TFLite format using helper functions in `utils.py`.
  Note that:
  
 - μNAS only calculates resource usage of a model and does not take particular framework overheads
  into account.
  
 - μNAS assumes that one of the input buffers to an `Add` operator can be reused as an output buffer
 if it is not used elsewhere (to minimise peak memory usage); this optimisation is not available
  in TF Lite Micro at the time of writing.
 
 - The operator execution order that gives the smallest peak memory usage is not recorded in the
  model: use [`tflite-tools`](https://github.com/eliberis/tflite-tools) to optimise your tflite
   model prior to deploying.