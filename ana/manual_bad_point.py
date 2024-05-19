"""Standalone script to execute write_manual_bad_point."""

import pathlib
import pprint
import sys
from typing import Final

import hydra
from omegaconf import DictConfig  # noqa: TC002

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# pylint: disable=import-error,useless-suppression,duplicate-code
# pylint: enable=useless-suppression
from utils.shared_functions import read_secure_pickle

# isort: off
from utils.bayesian_opt import write_manual_bad_point

# PyTorch NN Models
# from TSModelWrappers.NBEATSModelWrapper import NBEATSModelWrapper
from TSModelWrappers.NHiTSModelWrapper import NHiTSModelWrapper

# isort: on
# pylint: enable=import-error

__all__: list[str] = []


@hydra.main(version_base=None, config_path="..", config_name="config")
def run_write_manual_bad_point(
    cfg: DictConfig,
) -> None:
    """Run the write_manual_bad_point script.

    Args:
        cfg (DictConfig): Hydra configuration.
    """
    # Setup variables
    # pylint: disable=invalid-name
    PACKAGE_PATH: Final = pathlib.Path(cfg["general"]["package_path"]).expanduser()
    MODELS_PATH: Final = PACKAGE_PATH / "ana" / "models"
    BAYESIAN_OPT_WORK_DIR_NAME: Final = "bayesian_optimization"

    # Load PARENT_WRAPPER from pickle
    PARENT_WRAPPER_PATH: Final = MODELS_PATH / BAYESIAN_OPT_WORK_DIR_NAME / "parent_wrapper.pickle"
    PARENT_WRAPPER: Final = read_secure_pickle(PARENT_WRAPPER_PATH)
    # pylint: enable=invalid-name

    if PARENT_WRAPPER is None:
        print(f"Failed to load PARENT_WRAPPER from {PARENT_WRAPPER_PATH}!")
        sys.exit(3)

    # Manually specify bad points and model

    # model_wrapper_class = NBEATSModelWrapper
    # bad_point_to_write = {
    #     "batch_size": 182.67288601975548,
    #     "covariates_to_use": 4.0,
    #     "dropout": 0.15,
    #     "expansion_coefficient_dim": 10.0,
    #     "input_chunk_length": 1.0,
    #     "layer_widths": 845.7812745971257,
    #     "num_blocks": 10.0,
    #     "num_layers": 10.0,
    #     "num_stacks": 50.0,
    #     "time_bin_size_in_minutes": 20.0,
    #     "y_presentation": 2.0,
    # }
    # bad_point_to_write_clean = {
    #     "batch_size": 182,
    #     "covariates_to_use": 4,
    #     "dropout": 0.15,
    #     "expansion_coefficient_dim": 10,
    #     "input_chunk_length": 1,
    #     "layer_widths": 845,
    #     "num_blocks": 10,
    #     "num_layers": 10,
    #     "num_stacks": 50,
    #     "time_bin_size_in_minutes": 20,
    #     "y_presentation": 2,
    # }

    model_wrapper_class = NHiTSModelWrapper
    bad_point_to_write = {
        "MaxPool1d": 0.0,
        "batch_size": 955.0581345768601,
        "covariates_to_use": 4.0,
        "dropout": 0.0,
        "input_chunk_length": 60.0,
        "layer_widths": 719.959976362605,
        "num_blocks": 10.0,
        "num_layers": 10.0,
        "num_stacks": 50.0,
        "time_bin_size_in_minutes": 20.0,
        "y_presentation": 2.0,
    }
    bad_point_to_write_clean = {
        "MaxPool1d": False,
        "batch_size": 955,
        "covariates_to_use": 4,
        "dropout": 0.0,
        "input_chunk_length": 60,
        "layer_widths": 719,
        "num_blocks": 10,
        "num_layers": 10,
        "num_stacks": 50,
        "time_bin_size_in_minutes": 20,
        "y_presentation": 2,
    }

    print(
        f"""
bad_point_to_write = {pprint.pformat(bad_point_to_write)}

bad_point_to_write_clean = {pprint.pformat(bad_point_to_write_clean)}
"""
    )

    _model_name = model_wrapper_class.__name__.replace("Wrapper", "")
    response = input(
        f"Are you sure you want to manually write the above bad point for {_model_name}? "
    )
    if response.lower() not in ["y", "yes"]:
        sys.exit()

    response = input("Are you REALLY sure? ")
    if response.lower() not in ["y", "yes"]:
        sys.exit()

    write_manual_bad_point(
        bad_point_to_write=bad_point_to_write,
        bad_point_to_write_clean=bad_point_to_write_clean,
        parent_wrapper=PARENT_WRAPPER,
        model_wrapper_class=model_wrapper_class,
    )


if __name__ == "__main__":
    run_write_manual_bad_point()  # pylint: disable=no-value-for-parameter
