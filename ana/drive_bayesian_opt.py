"""Standalone script to execute run_bayesian_opt.

Used to brute force GPU memory resets between runs.

Returns 10 + the number of completed points as the exit code, or a integer 0 < status < 10 for exceptions.
"""

################################################################################
# python imports
import datetime
import os
import pathlib
import sys
from typing import TYPE_CHECKING, Final

import hydra
import tqdm
from omegaconf import DictConfig  # noqa: TC002

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# pylint: disable=import-error,useless-suppression
# pylint: enable=useless-suppression
from utils.shared_functions import read_secure_pickle

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
# pylint: enable=import-error


@hydra.main(version_base=None, config_path="..", config_name="config")
def drive_bayesian_opt(
    cfg: DictConfig,
) -> None:
    """Run the drive_bayesian_opt script.

    Args:
        cfg: Hydra configuration.
    """
    ################################################################################
    # Setup variables
    # pylint: disable=invalid-name
    PACKAGE_PATH: Final = pathlib.Path(cfg["general"]["package_path"]).expanduser()
    MODELS_PATH: Final = PACKAGE_PATH / "ana" / "models"
    BAYESIAN_OPT_WORK_DIR_NAME: Final = "bayesian_optimization"
    # pylint: enable=invalid-name

    ################################################################################
    # Select settings and model(s) to run

    prod_kwargs = {
        "n_iter": 1,
        "verbose": 9,
        "enable_torch_warnings": False,
        "enable_torch_model_summary": True,
        "enable_torch_progress_bars": False,
        "disregard_training_exceptions": True,
        "max_time_per_model": datetime.timedelta(minutes=45),
        "bayesian_opt_work_dir_name": BAYESIAN_OPT_WORK_DIR_NAME,
    }

    model_kwarg_list = [
        # Prophet
        {"model_wrapper_class": ProphetWrapper},  # +i_model=0
        # PyTorch NN Models
        {"model_wrapper_class": NBEATSModelWrapper},  # +i_model=1
        {"model_wrapper_class": NHiTSModelWrapper},  # +i_model=2
        {"model_wrapper_class": TCNModelWrapper},  # +i_model=3
        {"model_wrapper_class": TransformerModelWrapper},  # +i_model=4
        {"model_wrapper_class": TFTModelWrapper},  # +i_model=5
        {"model_wrapper_class": DLinearModelWrapper},  # +i_model=6
        {"model_wrapper_class": NLinearModelWrapper},  # +i_model=7
        {"model_wrapper_class": TiDEModelWrapper},  # +i_model=8
        {
            "model_wrapper_class": RNNModelWrapper,
            "model_wrapper_kwargs": {"model": "RNN"},
        },  # +i_model=9
        {
            "model_wrapper_class": RNNModelWrapper,
            "model_wrapper_kwargs": {"model": "LSTM"},
        },  # +i_model=10
        {
            "model_wrapper_class": RNNModelWrapper,
            "model_wrapper_kwargs": {"model": "GRU"},
        },  # +i_model=11
        {
            "model_wrapper_class": BlockRNNModelWrapper,
            "model_wrapper_kwargs": {"model": "RNN"},
        },  # +i_model=12
        {
            "model_wrapper_class": BlockRNNModelWrapper,
            "model_wrapper_kwargs": {"model": "LSTM"},
        },  # +i_model=13
        {
            "model_wrapper_class": BlockRNNModelWrapper,
            "model_wrapper_kwargs": {"model": "GRU"},
        },  # +i_model=14
        # Statistical Models
        {"model_wrapper_class": AutoARIMAWrapper},  # +i_model=15
        {"model_wrapper_class": BATSWrapper},  # +i_model=16
        {"model_wrapper_class": TBATSWrapper},  # +i_model=17
        {"model_wrapper_class": FourThetaWrapper},  # +i_model=18
        {"model_wrapper_class": StatsForecastAutoThetaWrapper},  # +i_model=19
        {"model_wrapper_class": FFTWrapper},  # +i_model=20
        {"model_wrapper_class": KalmanForecasterWrapper},  # +i_model=21
        {
            "model_wrapper_class": CrostonWrapper,
            "model_wrapper_kwargs": {"version": "optimized"},
        },  # +i_model=22
        {
            "model_wrapper_class": CrostonWrapper,
            "model_wrapper_kwargs": {"version": "classic"},
        },  # +i_model=23
        {
            "model_wrapper_class": CrostonWrapper,
            "model_wrapper_kwargs": {"version": "sba"},
        },  # +i_model=24
        # Regression Models
        {"model_wrapper_class": LinearRegressionModelWrapper},  # +i_model=25
        {"model_wrapper_class": RandomForestWrapper},  # +i_model=26
        {"model_wrapper_class": LightGBMModelWrapper},  # +i_model=27
        {"model_wrapper_class": XGBModelWrapper},  # +i_model=28
        {"model_wrapper_class": CatBoostModelWrapper},  # +i_model=29
        # Naive Models
        {"model_wrapper_class": NaiveMeanWrapper},  # +i_model=30
        {"model_wrapper_class": NaiveSeasonalWrapper},  # +i_model=31
        {"model_wrapper_class": NaiveDriftWrapper},  # +i_model=32
        {"model_wrapper_class": NaiveMovingAverageWrapper},  # +i_model=33
    ]

    # accept i_model CLI argument to only run one model
    i_model = cfg.get("i_model")
    if i_model is not None:
        if i_model not in range(len(model_kwarg_list)):
            print(f"Received {i_model =} but {len(model_kwarg_list) =}!")
            sys.exit(3)

        model_kwarg_list = [model_kwarg_list[i_model]]

    ################################################################################
    # Load PARENT_WRAPPER from pickle

    # pylint: disable=invalid-name
    PARENT_WRAPPER_PATH: Final = MODELS_PATH / BAYESIAN_OPT_WORK_DIR_NAME / "parent_wrapper.pickle"
    PARENT_WRAPPER: Final = read_secure_pickle(PARENT_WRAPPER_PATH)
    # pylint: enable=invalid-name

    if PARENT_WRAPPER is None:
        print(f"Failed to load PARENT_WRAPPER from {PARENT_WRAPPER_PATH}!")
        sys.exit(3)

    prod_kwargs["parent_wrapper"] = PARENT_WRAPPER

    ################################################################################
    # Run Bayesian Optimization

    single_iter_flag = len(model_kwarg_list) == 1 and prod_kwargs["n_iter"] == 1

    if not single_iter_flag:
        response = input(
            f"Are you sure you want to optimize {len(model_kwarg_list)} models, for {prod_kwargs['n_iter']} iterations each, in one script? "
        )
        if response.lower() not in ["y", "yes"]:
            sys.exit()

    # pylint: disable=duplicate-code
    for model_kwarg in (pbar := tqdm.auto.tqdm(model_kwarg_list)):
        _model_name = model_kwarg["model_wrapper_class"].__name__.replace("Wrapper", "")
        if not os.environ.get("TQDM_DISABLE", 0):
            pbar.set_postfix_str(f"Optimizing {_model_name}")
        else:
            print(f"Optimizing {_model_name}")

        if TYPE_CHECKING:
            assert isinstance(prod_kwargs, dict)  # noqa: SCS108 # nosec assert_used
            assert isinstance(model_kwarg, dict)  # noqa: SCS108 # nosec assert_used

        _, optimizer, exception_status = run_bayesian_opt(
            **prod_kwargs,  # type: ignore[arg-type]
            **model_kwarg,  # type: ignore[arg-type]
        )
        # pylint: enable=duplicate-code

        n_points = len(optimizer.space)

        if exception_status:
            sys.exit(exception_status)
        elif single_iter_flag:
            sys.exit(10 + n_points)


if __name__ == "__main__":
    drive_bayesian_opt()  # pylint: disable=no-value-for-parameter
