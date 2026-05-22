import os
import json
import logging
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from uNAS import uNAS

# ---------------------------------------------------------------------------
# Config registry
# Each entry: short_name -> (module, function_name)
# ---------------------------------------------------------------------------
_CONFIGS = {
    "imagenet":       ("configs.imagenet_cnn2d",       "get_imagenet_setup"),
    "mini_imagenet":  ("configs.mini_imagenet_cnn2d",  "get_mini_imagenet_setup"),
    "wakeviz":        ("configs.cnn_wakeviz",           "get_wakeviz_setup"),
    "har":            ("configs.test_HAR",              "get_HAR_setup"),
    "sr":             ("configs.test_SR",               "get_speechcommands_setup"),
    "dia":            ("configs.test_DIA",              "get_DIA_setup"),
    "z24":            ("configs.test_Z24",              "get_Z24_setup"),
    "regression":     ("configs.test_regression",       "get_REG_setup"),
    "dummy_2d":       ("configs.test_dummy_dataset",    "get_dummy_2D_setup"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driver.py",
        description=(
            "μNAS — Neural Architecture Search for microcontrollers.\n\n"
            "When called with no arguments, reads defaults from params.json\n"
            "(or the file specified with --params-file)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available configs:\n  " + "\n  ".join(_CONFIGS.keys()),
    )

    parser.add_argument(
        "--params-file",
        default="params.json",
        metavar="JSON",
        help="JSON file of default argument values. CLI flags override file values. "
             "Default: params.json.",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        metavar="NAME",
        choices=list(_CONFIGS.keys()),
        help="Experiment config to run (see list at the bottom of --help).",
    )
    parser.add_argument(
        "-d", "--data-dir",
        default=None,
        metavar="PATH",
        help="Root data directory.  Forwarded to the config as 'data_dir' when supported "
             "(e.g. imagenet, wakeviz, sr).",
    )
    parser.add_argument(
        "-l", "--load-from",
        default=None,
        metavar="FILE",
        help="Path to a search-state .pickle file to resume an interrupted search.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        metavar="N",
        help="Override how often (in evaluations) the search state is checkpointed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Override the global random seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Override the training batch size.",
    )
    # Image-related overrides (imagenet / wakeviz)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Override the input image size, e.g. --image-size 96 96.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of output classes (imagenet).",
    )

    # BoundConfig overrides
    parser.add_argument(
        "--error-bound",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Override the maximum validation error bound "
             "(e.g. 0.3 means top-1 accuracy must be ≥ 70%%).",
    )
    parser.add_argument(
        "--peak-mem-bound",
        type=int,
        default=None,
        metavar="BYTES",
        help="Override the peak SRAM usage bound in bytes (e.g. 524288 for 512 KB).",
    )
    parser.add_argument(
        "--model-size-bound",
        type=int,
        default=None,
        metavar="BYTES",
        help="Override the model weight storage bound in bytes.",
    )
    parser.add_argument(
        "--mac-bound",
        type=int,
        default=None,
        metavar="N",
        help="Override the multiply-accumulate operations bound.",
    )
    return parser


def _merge_json_defaults(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Load argument defaults from a JSON file; explicit CLI values take priority.

    JSON keys must match argparse ``dest`` names (underscores, not hyphens).
    Example params.json::

        {
            "config": "imagenet",
            "data_dir": "/data/imagenet",
            "image_size": [96, 96],
            "error_bound": 0.7,
            "peak_mem_bound": 524288,
            "model_size_bound": 524288,
            "mac_bound": 5000000
        }
    """
    json_path = args.params_file
    if not os.path.exists(json_path):
        return args
    try:
        with open(json_path) as f:
            defaults = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        parser.error(f"Could not read {json_path!r}: {e}")

    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    return args


def load_setup(args) -> dict:
    """Import the chosen config module lazily and call its setup function."""
    import importlib

    module_path, fn_name = _CONFIGS[args.config]
    module = importlib.import_module(module_path)
    setup_fn = getattr(module, fn_name)

    # Build keyword arguments from CLI flags that the setup function accepts.
    import inspect
    sig = inspect.signature(setup_fn)
    supported = sig.parameters.keys()

    kwargs = {}
    if "data_dir"    in supported and args.data_dir    is not None:
        kwargs["data_dir"]    = args.data_dir
    if "batch_size"  in supported and args.batch_size  is not None:
        kwargs["batch_size"]  = args.batch_size
    if "input_size"  in supported and args.image_size  is not None:
        kwargs["input_size"]  = tuple(args.image_size)
    if "image_size"  in supported and args.image_size  is not None:
        kwargs["image_size"]  = tuple(args.image_size)
    if "num_classes" in supported and args.num_classes is not None:
        kwargs["num_classes"] = args.num_classes
    if "error_bound"      in supported and args.error_bound      is not None:
        kwargs["error_bound"]      = args.error_bound
    if "peak_mem_bound"   in supported and args.peak_mem_bound   is not None:
        kwargs["peak_mem_bound"]   = args.peak_mem_bound
    if "model_size_bound" in supported and args.model_size_bound is not None:
        kwargs["model_size_bound"] = args.model_size_bound
    if "mac_bound"        in supported and args.mac_bound        is not None:
        kwargs["mac_bound"]        = args.mac_bound

    setup = setup_fn(**kwargs)

    # Apply CLI overrides for generic setup-dict keys.
    if args.load_from  is not None:
        setup["load_from"]  = args.load_from
    if args.save_every is not None:
        setup["save_every"] = args.save_every
    if args.seed       is not None:
        setup["seed"]       = args.seed

    # Apply BoundConfig overrides.
    bound = setup["config"]["bound_config"]
    if args.error_bound      is not None:
        bound.error_bound      = args.error_bound
    if args.peak_mem_bound   is not None:
        bound.peak_mem_bound   = args.peak_mem_bound
    if args.model_size_bound is not None:
        bound.model_size_bound = args.model_size_bound
    if args.mac_bound        is not None:
        bound.mac_bound        = args.mac_bound

    return setup


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Merge JSON file defaults (CLI values win over JSON values).
    args = _merge_json_defaults(args, parser)

    # Validate config after JSON has been merged.
    if args.config is None:
        parser.error(
            "a config name is required. Provide it with -c/--config or via params.json.\n"
            f"Available configs: {', '.join(_CONFIGS)}"
        )
    if args.config not in _CONFIGS:
        parser.error(
            f"invalid config: {args.config!r}. Choose from: {', '.join(_CONFIGS)}"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("Driver")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    setup = load_setup(args)
    logger.info(f"Running experiment: {setup.get('name', args.config)}")

    unas = uNAS(setup, logger)
    unas.run()


if __name__ == "__main__":
    main()
