"""Standalone script to execute run_bayesian_opt.

Used to brute force GPU memory resets between runs.

Returns the number of completed points as the exit code.
"""

################################################################################
# python imports
import datetime
import pathlib
import pickle  # nosec B403
import sys
from typing import TYPE_CHECKING, Final

from hydra import compose, initialize

sys.path.append(str(pathlib.Path.cwd().parent))

# pylint: disable=unused-import,import-error,useless-suppression
# isort: off
from utils.bayesian_opt import run_bayesian_opt

# PyTorch NN Models
from utils.NBEATSModelWrapper import NBEATSModelWrapper
from utils.NHiTSModelWrapper import NHiTSModelWrapper
from utils.TCNModelWrapper import TCNModelWrapper
from utils.TransformerModelWrapper import TransformerModelWrapper
from utils.TFTModelWrapper import TFTModelWrapper
from utils.DLinearModelWrapper import DLinearModelWrapper
from utils.NLinearModelWrapper import NLinearModelWrapper
from utils.TiDEModelWrapper import TiDEModelWrapper
from utils.RNNModelWrapper import RNNModelWrapper
from utils.BlockRNNModelWrapper import BlockRNNModelWrapper

# isort: on
# pylint: enable=unused-import,import-error

################################################################################
# Setup variables

initialize(version_base=None, config_path="..")
cfg = compose(config_name="config")
PACKAGE_PATH: Final = pathlib.Path(cfg["general"]["package_path"]).expanduser()
MODELS_PATH: Final = PACKAGE_PATH / "ana" / "models"
BAYESIAN_OPT_WORK_DIR_NAME: Final = "bayesian_optimization"

################################################################################
# Pick model to run

model_kwarg_list = [
    # PyTorch NN Models
    {"model_wrapper_class": NBEATSModelWrapper},
    # {"model_wrapper_class": NHiTSModelWrapper},
    # {"model_wrapper_class": TCNModelWrapper},
    # {"model_wrapper_class": TransformerModelWrapper},
    # {"model_wrapper_class": TFTModelWrapper},
    # {"model_wrapper_class": DLinearModelWrapper},
    # {"model_wrapper_class": NLinearModelWrapper},
    # {"model_wrapper_class": TiDEModelWrapper},
    # {"model_wrapper_class": RNNModelWrapper, "model_wrapper_kwargs": {"model": "RNN"}},
    # {"model_wrapper_class": RNNModelWrapper, "model_wrapper_kwargs": {"model": "LSTM"}},
    # {"model_wrapper_class": RNNModelWrapper, "model_wrapper_kwargs": {"model": "GRU"}},
    # {"model_wrapper_class": BlockRNNModelWrapper, "model_wrapper_kwargs": {"model": "RNN"}},
    # {"model_wrapper_class": BlockRNNModelWrapper, "model_wrapper_kwargs": {"model": "LSTM"}},
    # {"model_wrapper_class": BlockRNNModelWrapper, "model_wrapper_kwargs": {"model": "GRU"}},
]

################################################################################
# Load PARENT_WRAPPER from pickle

PARENT_WRAPPER = None
PARENT_WRAPPER_PATH: Final = MODELS_PATH / BAYESIAN_OPT_WORK_DIR_NAME / "parent_wrapper.pickle"
with open(PARENT_WRAPPER_PATH, "rb") as f_pickle:
    PARENT_WRAPPER = pickle.load(f_pickle)  # nosec B301

if PARENT_WRAPPER is None:
    raise ValueError(f"Failed to load PARENT_WRAPPER from {PARENT_WRAPPER_PATH}!")

print("PARENT_WRAPPER =")
print(PARENT_WRAPPER)

################################################################################
# Run Bayesian Optimization

if len(model_kwarg_list) != 1:
    raise ValueError(f"Require 1 = {len(model_kwarg_list) = } to run this script!")

model_kwarg = model_kwarg_list[0]

MODEL_NAME: Final = model_kwarg["model_wrapper_class"].__name__.replace("Wrapper", "")
print(f"Optimizing {MODEL_NAME}")

prod_kwargs = {
    "parent_wrapper": PARENT_WRAPPER,
    "bayesian_opt_work_dir_name": BAYESIAN_OPT_WORK_DIR_NAME,
    "verbose": 2,
    "disregard_training_exceptions": True,
    "n_iter": 1,
    "max_time_per_model": datetime.timedelta(minutes=30),
}


if prod_kwargs["n_iter"] != 1:
    raise ValueError(f"Require 1 = n_iter = {prod_kwargs['n_iter'] } to run this script!")

if TYPE_CHECKING:
    assert isinstance(prod_kwargs, dict)  # noqa: SCS108 # nosec assert_used
    assert isinstance(model_kwarg, dict)  # noqa: SCS108 # nosec assert_used

_, optimizer = run_bayesian_opt(
    **prod_kwargs,  # type: ignore[arg-type]
    **model_kwarg,  # type: ignore[arg-type]
)

n_points = len(optimizer.space)

print(f"Completed {n_points = }")

sys.exit(n_points)
