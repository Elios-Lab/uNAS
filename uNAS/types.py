
def get_example_unas_config():
    """
    Returns an example configuration for uNAS.

    The nas module configuration is a dictionary with the following keys
    and values:
    - config_file: str, path to the configuration file
    - name: str, name of the experiment
    - load_from: str, path to the search state file to resume from
    - save_every: int, after how many search steps to save the state
    - seed: int, a seed for the global NumPy and TensorFlow random state
    """
    return {
        'config_file': 'configs/test_dummy_dataset.py',
        'name': 'test_uNAS_module',
        'load_from': None,
        'save_every': 5,
        'seed': 0
        }