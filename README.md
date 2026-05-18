# μNAS

μNAS (micro-NAS or mu-NAS) is a neural architecture search system that specialises in finding
ultra-small models suitable for deploying on microcontrollers: think < 64 KB memory and storage
requirement. μNAS achieves this by explicitly targeting three primary resource bottlenecks:
model size, latency and peak memory usage.

For a full description of methodology and experimental results, please see the accompanying paper
[_"μNAS: Constrained Neural Architecture Search for Microcontrollers"_](https://arxiv.org/abs/2010.14246).

---

## Changelog

*From arXiv v1:*
- Correctly reported the number of MACs for the DS-CNN baseline for the Speech Commands dataset.
- Fixed Speech Commands hyperparameters and updated found models.
- Added smaller CIFAR-10 model in the comparison table.
- Added search times to the comparison table.
- Updated discussion on pruning, search convergence and the use of soft constraints.

*From original implementation:*
- 1D and 2D Convolutional Neural Networks are now supported.
- Quantisation Aware Training (QAT) is supported during NAS search and fine-tuning, selectable via config.
- Dynamic dataset loading is supported through the TensorFlow Dataset API.
- Output format is Keras 2 model (`.h5`); `test.py` handles conversion to TFLite INT8.
- ImageNet dataset support added.
- `driver.py` now exposes a full CLI — no editing required to switch experiments.
- Complexity penalty in aging evolution reduces residual-heavy architectures that do not improve accuracy.
- macOS / Apple Silicon MPS GPU acceleration supported via `tensorflow-metal`.

---

## Setup

μNAS requires **Python 3.10** and **TensorFlow 2.18.0**.

```bash
pip install -r requirements.txt
```

`requirements.txt` automatically selects the right TensorFlow variant for your platform:

| Platform | Package installed |
|---|---|
| Linux | `tensorflow[and-cuda]` (NVIDIA GPU) |
| macOS | `tensorflow` + `tensorflow-metal` (Apple Silicon MPS) |
| Windows | `tensorflow<2.11` |

> If output models are not compatible with [ST developer tools](https://stm32ai-cs.st.com/home) after TFLite conversion, try downgrading to an earlier Python / TensorFlow version.

---

## Three-step pipeline

### Step 1 — Run the NAS search

```bash
python driver.py -c <config> [options]
```

`driver.py` accepts a short config name via `-c` and automatically imports only the selected
config module. No editing is required.

**Available configs**

| Name | Task | Search space |
|---|---|---|
| `imagenet` | ImageNet classification (1000 classes) | 2D CNN |
| `wakeviz` | Wake-word visual binary classification | 2D CNN |
| `har` | Human Activity Recognition | 1D CNN |
| `sr` | Speech Commands keyword recognition | 1D CNN |
| `dia` | Diabetes health indicators | 1D CNN / MLP |
| `z24` | Z24 bridge vibration anomaly detection | 1D CNN |
| `regression` | Generic regression example | 1D CNN |
| `dummy_2d` | Synthetic 2D data (quick smoke-test) | 2D CNN |

**Common options**

```
-c, --config NAME        Experiment to run (required, see table above)
-d, --data-dir PATH      Root data directory (required for imagenet, wakeviz, sr)
-l, --load-from FILE     Resume from a saved search-state .pickle file
    --save-every N       Checkpoint search state every N evaluations
    --seed INT           Override the global random seed
    --batch-size N       Override the training batch size
    --image-size H W     Override input image resolution (imagenet / wakeviz)
    --num-classes N      Override number of output classes (imagenet)
```

**Examples**

```bash
# ImageNet search at 96×96
python driver.py -c imagenet -d /data/imagenet

# ImageNet at lower resolution for tighter MCU budgets
python driver.py -c imagenet -d /data/imagenet --image-size 64 64

# Resume an interrupted ImageNet search
python driver.py -c imagenet -d /data/imagenet \
  -l artifacts/imagenet_cnn2d/imagenet_cnn2d_agingevosearch_state.pickle

# Human Activity Recognition with a fixed seed
python driver.py -c har --seed 42

# Speech Commands with larger batches
python driver.py -c sr --batch-size 256

# Quick functionality test with a dummy dataset
python driver.py -c dummy_2d
```

The search saves the best models (`.h5`) and a state checkpoint (`.pickle`) under
`artifacts/<experiment_name>/`.

---

### Step 2 — QAT fine-tuning

After picking a model from the search artifacts, run `train.py` to apply
Quantisation Aware Training (QAT) fine-tuning:

```bash
python train.py
```

Edit the variables at the top of `train.py` before running:

| Variable | Description |
|---|---|
| `model_path` | Path to the `.h5` model found by the NAS search |
| `model_name` | Output filename (saved as `<model_name>.h5`) |
| `train_dir` / `validation_dir` / `test_dir` | Dataset split directories |
| `img_size` | Must match the image size used during search |
| `epochs` / `learning_rate` / `batch_size` | Fine-tuning hyperparameters |

`train.py` loads the float32 model, wraps it with
`tfmot.quantization.keras.quantize_model()` to insert fake-quantisation nodes, and
fine-tunes it so weights and activations learn to be robust to INT8 rounding.

> **Tip:** To enable QAT directly during the NAS search (instead of only at fine-tuning
> time), set `use_qat=True` in the `TrainingConfig` of your config file. This makes
> the search optimise for quantised accuracy from the start.

---

### Step 3 — Evaluate and convert to TFLite

```bash
python test.py
```

The script will interactively ask whether to:
1. Evaluate the QAT `.h5` model directly, or
2. Convert it to INT8 TFLite first and then evaluate.

The TFLite INT8 conversion uses a representative dataset sample for calibration and
targets `TFLITE_BUILTINS_INT8` ops with `uint8` input/output — ready for deployment
on TF Lite Micro targets.

---

## Adding a new dataset

1. Create `dataset/my_dataset.py` inheriting from `uNAS.dataset.Dataset` and implementing:
   - `train_dataset() -> tf.data.Dataset` — unbatched, individual samples
   - `validation_dataset() -> tf.data.Dataset`
   - `test_dataset() -> tf.data.Dataset`
   - `num_classes` property
   - `input_shape` property

2. Register it in `dataset/__init__.py`:
   ```python
   from .my_dataset import MyDataset
   ```

3. Create `configs/my_config.py` returning a setup dict with keys
   `config`, `name`, `load_from`, `save_every`, `seed`.
   See `configs/test_HAR.py` (1D) or `configs/imagenet_cnn2d.py` (2D) for reference.

4. Add an entry to the `_CONFIGS` registry in `driver.py`:
   ```python
   "my_experiment": ("configs.my_config", "get_my_setup"),
   ```

5. Run: `python driver.py -c my_experiment`

### Large / streaming datasets

For datasets that cannot fit in RAM (e.g. ImageNet), use lazy loading:
- Return an **unbatched** `tf.data.Dataset` from each split method (batching is handled by `ModelTrainer`).
- Pass `dataset=partial(MyDataset, ...)` and `serialized_dataset=True` in `TrainingConfig` so each Ray worker instantiates its own copy.
- Override `class_weight()` to avoid iterating the full dataset (return `np.ones(num_classes)` for balanced datasets).

---

## Key configuration knobs

### `TrainingConfig`

| Field | Default | Description |
|---|---|---|
| `dataset` | — | Dataset instance or `partial(DatasetClass, ...)` |
| `optimizer` | — | Keras optimizer string or instance |
| `epochs` | `75` | Training epochs per candidate |
| `batch_size` | `128` | Training batch size |
| `use_qat` | `False` | Enable QAT during NAS search |
| `distillation` | `None` | `DistillationConfig` for knowledge distillation |
| `pruning` | `None` | `PruningConfig` for DPF-style pruning |
| `use_class_weight` | `False` | Rebalance loss by class frequency |
| `serialized_dataset` | `False` | Lazy per-worker dataset instantiation |

### `AgingEvoConfig`

| Field | Default | Description |
|---|---|---|
| `search_space` | — | `Cnn1DSearchSpace()`, `Cnn2DSearchSpace()`, or `MlpSearchSpace()` |
| `rounds` | `2000` | Total number of architectures to evaluate |
| `population_size` | `100` | Size of the living population |
| `sample_size` | `25` | Tournament sample size |
| `complexity_penalty` | `0.5` | Down-weight mutations that add residual branches or layers (0 = off, higher = stronger preference for simpler models) |
| `checkpoint_dir` | `"artifacts"` | Where to save search state and models |

### `BoundConfig`

All bounds are soft constraints enforced via the multi-objective fitness function.

| Field | Description |
|---|---|
| `error_bound` | Maximum acceptable validation error (e.g. `0.3` = 70 % top-1 accuracy) |
| `peak_mem_bound` | Peak SRAM usage in bytes |
| `model_size_bound` | Model weight storage in bytes |
| `mac_bound` | Multiply-accumulate operations |

---

## Internal modules

| Module | Description |
|---|---|
| `cnn1d` / `cnn2d` / `mlp` | Search space, morphisms and random generators for each architecture family |
| `search_algorithms` | Aging evolution and Bayesian optimisation search algorithms |
| `resource_models` | Graph-based library for computing peak memory, model size and MACs |
| `dragonfly_adapters` | Bayesian optimisation interop with [Dragonfly](https://github.com/dragonfly/dragonfly) |
| `teachers` | Pre-trained teacher models for knowledge distillation |
| `model_trainer.py` | Trains and evaluates a single candidate model |
| `pruning.py` | [DPF pruning](https://openreview.net/forum?id=SJem8lSFwB) Keras callback |
| `utils.py` | Quantised accuracy evaluation, TFLite conversion helpers |
| `generate_tflite_models.py` | Generates random small models for on-device latency benchmarking |

---

## Notes on deploying found models

μNAS does not save final weights of discovered models by default (it can be modified to do so).
Because aging evolution does not share weights across candidates, each model is trained from
scratch, which encourages architectures that learn well independently.

μNAS assumes a runtime where each operator is executed one at a time and in full, such as
[TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers). When converting:

- μNAS resource estimates do **not** include framework overheads.
- μNAS assumes one input buffer of an `Add` operator can be reused as the output buffer when not needed elsewhere (minimising peak memory). This optimisation is not available in TF Lite Micro at the time of writing.
- Use [`tflite-tools`](https://github.com/eliberis/tflite-tools) to optimise operator execution order in the `.tflite` file before deploying.
