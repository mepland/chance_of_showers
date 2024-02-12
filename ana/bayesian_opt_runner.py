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

import tqdm
from hydra import compose, initialize

sys.path.append(str(pathlib.Path.cwd().parent))

# pylint: disable=unused-import,import-error,useless-suppression
# isort: off
from utils.bayesian_opt import run_bayesian_opt

# Prophet
from utils.ProphetWrapper import ProphetWrapper

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

# Statistical Models
from utils.AutoARIMAWrapper import AutoARIMAWrapper
from utils.BATSWrapper import BATSWrapper
from utils.TBATSWrapper import TBATSWrapper
from utils.FourThetaWrapper import FourThetaWrapper
from utils.StatsForecastAutoThetaWrapper import StatsForecastAutoThetaWrapper
from utils.FFTWrapper import FFTWrapper
from utils.KalmanForecasterWrapper import KalmanForecasterWrapper
from utils.CrostonWrapper import CrostonWrapper

# Regression Models
from utils.LinearRegressionModelWrapper import LinearRegressionModelWrapper
from utils.RandomForestWrapper import RandomForestWrapper
from utils.LightGBMModelWrapper import LightGBMModelWrapper
from utils.XGBModelWrapper import XGBModelWrapper
from utils.CatBoostModelWrapper import CatBoostModelWrapper

# Naive Models
from utils.NaiveMeanWrapper import NaiveMeanWrapper
from utils.NaiveSeasonalWrapper import NaiveSeasonalWrapper
from utils.NaiveDriftWrapper import NaiveDriftWrapper
from utils.NaiveMovingAverageWrapper import NaiveMovingAverageWrapper

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
# Select settings and model(s) to run

prod_kwargs = {
    "bayesian_opt_work_dir_name": BAYESIAN_OPT_WORK_DIR_NAME,
    "verbose": 2,
    "enable_progress_bar": False,
    "disregard_training_exceptions": True,
    "n_iter": 1,
    "max_time_per_model": datetime.timedelta(minutes=30),
}

model_kwarg_list = [
    # Prophet
    # {"model_wrapper_class": ProphetWrapper},
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
    # Statistical Models
    # {"model_wrapper_class": AutoARIMAWrapper},
    # {"model_wrapper_class": BATSWrapper},
    # {"model_wrapper_class": TBATSWrapper},
    # {"model_wrapper_class": FourThetaWrapper},
    # {"model_wrapper_class": StatsForecastAutoThetaWrapper},
    # {"model_wrapper_class": FFTWrapper},
    # {"model_wrapper_class": KalmanForecasterWrapper},
    # {"model_wrapper_class": CrostonWrapper, "model_wrapper_kwargs": {"version": "optimized"}},
    # {"model_wrapper_class": CrostonWrapper, "model_wrapper_kwargs": {"version": "classic"}},
    # {"model_wrapper_class": CrostonWrapper, "model_wrapper_kwargs": {"version": "sba"}},
    # Regression Models
    # {"model_wrapper_class": LinearRegressionModelWrapper},
    # {"model_wrapper_class": RandomForestWrapper},
    # {"model_wrapper_class": LightGBMModelWrapper},
    # {"model_wrapper_class": XGBModelWrapper},
    # {"model_wrapper_class": CatBoostModelWrapper},
    # Naive Models
    # {"model_wrapper_class": NaiveMeanWrapper},
    # {"model_wrapper_class": NaiveSeasonalWrapper},
    # {"model_wrapper_class": NaiveDriftWrapper},
    # {"model_wrapper_class": NaiveMovingAverageWrapper},
]

################################################################################
# Load PARENT_WRAPPER from pickle

PARENT_WRAPPER = None
PARENT_WRAPPER_PATH: Final = MODELS_PATH / BAYESIAN_OPT_WORK_DIR_NAME / "parent_wrapper.pickle"
with open(PARENT_WRAPPER_PATH, "rb") as f_pickle:
    PARENT_WRAPPER = pickle.load(f_pickle)  # nosec B301

if PARENT_WRAPPER is None:
    raise ValueError(f"Failed to load PARENT_WRAPPER from {PARENT_WRAPPER_PATH}!")

prod_kwargs["parent_wrapper"] = PARENT_WRAPPER


################################################################################
# Run Bayesian Optimization

SINGLE_ITER_FLAG: Final = len(model_kwarg_list) == 1 and prod_kwargs["n_iter"] == 1

if not SINGLE_ITER_FLAG:
    response = input(
        f"Are you sure you want to optimize {len(model_kwarg_list)} models, for {prod_kwargs['n_iter']} iterations each, in one script?"
    )
    if response.lower() not in ["y", "yes"]:
        sys.exit()

for model_kwarg in (pbar := tqdm.auto.tqdm(model_kwarg_list)):
    _model_name = model_kwarg["model_wrapper_class"].__name__.replace("Wrapper", "")
    pbar.set_postfix_str(f"Optimizing {_model_name}")

    if TYPE_CHECKING:
        assert isinstance(prod_kwargs, dict)  # noqa: SCS108 # nosec assert_used
        assert isinstance(model_kwarg, dict)  # noqa: SCS108 # nosec assert_used

    _, optimizer = run_bayesian_opt(
        **prod_kwargs,  # type: ignore[arg-type]
        **model_kwarg,  # type: ignore[arg-type]
    )

    n_points = len(optimizer.space)
    print(f"Completed {n_points = }")

    if SINGLE_ITER_FLAG:
        sys.exit(n_points)
