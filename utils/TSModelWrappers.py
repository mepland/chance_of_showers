# pylint: disable=invalid-name
"""Wrapper classes for time series models."""

# pylint: enable=invalid-name

import datetime
import gc
import logging
import math
import os
import pprint
import traceback
import warnings
import zoneinfo
from typing import TYPE_CHECKING, Any, Final

import bayes_opt
import humanize
import numpy as np
import pandas as pd
import psutil
import sympy
import torch
import torchmetrics
from bayes_opt.event import DEFAULT_EVENTS, Events
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.util import load_logs
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils.missing_values import fill_missing_values, missing_values_ratio
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# isort: off
from darts.models import (
    #    AutoARIMA,
    #    RandomForest,
    #    XGBModel,
    #    LightGBMModel,
    #    CatBoostModel,
    #    RNNModel,
    #    BlockRNNModel,
    NBEATSModel,
    #    NHiTSModel,
    #    TCNModel,
    #    TransformerModel,
    #    TFTModel,
    #    DLinearModel,
    #    NLinearModel,
    #    TiDEModel,
)

# isort: on

from utils.shared_functions import (
    create_datetime_component_cols,
    rebin_chance_of_showers_time_series,
)

################################################################################
# Setup global parameters

warnings.filterwarnings(
    "ignore",
    message=r"The number of training batches (\d*) is smaller than the logging interval",
    category=PossibleUserWarning,
)

# loss function
LOSS_FN: Final = torchmetrics.MeanSquaredError()

# metrics to log at each epoch
METRIC_COLLECTION: Final = torchmetrics.MetricCollection(
    [
        torchmetrics.MeanSquaredError(),
        torchmetrics.MeanAbsolutePercentageError(),
    ]
)


def print_memory_usage(*, header: str | None = None) -> None:
    """Print system memory usage statistics.

    Args:
        header: Header to print before the rest of the memory usage.
    """
    ram_info = psutil.virtual_memory()
    process = psutil.Process()
    if header is not None and header != "":
        header = f"{header}\n"
    else:
        header = ""
    memory_usage_str = (
        header
        + f"RAM Available: {humanize.naturalsize(ram_info.available)}, "
        + f"System Used: {humanize.naturalsize(ram_info.used)}, {ram_info.percent:.2f}%, "
        + f"Process Used: {humanize.naturalsize(process.memory_info().rss)}"
    )

    if torch.cuda.is_available():
        gpu_memory_stats = torch.cuda.memory_stats()
    memory_usage_str += (
        f", GPU RAM Current: {humanize.naturalsize(gpu_memory_stats['allocated_bytes.all.current'])}, "
        + f"Peak: {humanize.naturalsize(gpu_memory_stats['allocated_bytes.all.peak'])}"
    )
    print(memory_usage_str)


# EarlyStopping stops training when validation loss does not decrease more than min_delta over a period of patience epochs
# copy docs from
# https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/early_stopping.html#EarlyStopping
# https://lightning.ai/docs/pytorch/stable/common/trainer.html
def get_pl_trainer_kwargs(
    es_min_delta: float,
    es_patience: int,
    *,
    enable_progress_bar: bool,
    max_time: str | datetime.timedelta | dict[str, int] | None,
    log_every_n_steps: int | None = None,
) -> dict:
    """Get pl_trainer_kwargs, i.e. PyTorch lightning trainer keyword arguments.

    Args:
        es_min_delta: Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
            change of less than or equal to `min_delta`, will count as no improvement.
        es_patience: Number of checks with no improvement
            after which training will be stopped. Under the default configuration, one check happens after
            every training epoch. However, the frequency of validation can be modified by setting various parameters on
            the ``Trainer``, for example ``check_val_every_n_epoch`` and ``val_check_interval``.
            .. note::
                It must be noted that the patience parameter counts the number of validation checks with
                no improvement, and not the number of training epochs. Therefore, with parameters
                ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training
                epochs before being stopped.
        enable_progress_bar: Enable torch progress bar during training.
        max_time: Set the maximum amount of time for training. Training will get interrupted mid-epoch.
        log_every_n_steps: How often to log within steps.

    Returns:
        pl_trainer_kwargs.
    """
    return {
        "enable_progress_bar": enable_progress_bar,
        "max_time": max_time,
        "log_every_n_steps": log_every_n_steps,
        "callbacks": [
            EarlyStopping(
                min_delta=es_min_delta,
                patience=es_patience,
                monitor="val_loss",
                mode="min",
            )
        ],
    }


# ReduceLROnPlateau will lower learning rate if model is in a plateau
# copy docs from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau
def get_lr_scheduler_kwargs(lr_factor: float, lr_patience: int) -> dict:
    """Get lr_scheduler_kwargs, i.e. PyTorch learning rate scheduler keyword arguments.

    Args:
        lr_factor: Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        lr_patience: Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.

    Returns:
        lr_scheduler_kwargs.
    """
    return {
        "factor": lr_factor,
        "patience": lr_patience,
        "threshold": 0.0001,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr": 0.0,
        "eps": 1e-08,
        "verbose": False,
    }


################################################################################
# Setup global hyperparameter sets

# data hyperparameters
DATA_REQUIRED_HYPERPARAMS: Final = [
    "time_bin_size_in_minutes",
    "prediction_length_in_minutes",
    "rebin_y",
    "y_bin_edges",
    # not required by all models, but we'll check
    # supports_future_covariates and supports_past_covariates for each model later and configure appropriately
    "covariates",
    # not really data, but needed by all models for Bayesian optimization
    "random_state",
]

DATA_VARIABLE_HYPERPARAMS: Final = {
    "time_bin_size_in_minutes": {
        "min": 1,
        "max": 20,
        "default": 10,
    },
    "rebin_y": {
        "min": 0,
        "max": 1,
        "default": 0,
    },
    "y_bin_edges": [-float("inf"), 0.6, 0.8, 0.9, 1.0],
    # technically had_flow is a measured, i.e. past, covariate, but we can assume we know it in the future and that it is always 0
    # unless we are actually making live predictions in production, then we can just take the current value from the DAQ
    "covariates": {
        "allowed": ["had_flow", "day_of_week_frac", "time_of_day_frac", "is_holiday"],
        "default": ["had_flow", "day_of_week_frac", "time_of_day_frac", "is_holiday"],
    },
}

DATA_FIXED_HYPERPARAMS: Final = {
    "prediction_length_in_minutes": 15,  # user specification for model
}

# NN hyperparameters

NN_REQUIRED_HYPERPARAMS: Final = [
    "input_chunk_length",
    "output_chunk_length",
    "dropout",
    "n_epochs",
    "batch_size",
    "work_dir",
    "model_name",
    "pl_trainer_kwargs",
    "loss_fn",
    "torch_metrics",
    "log_tensorboard",
    "lr_scheduler_cls",
    "lr_scheduler_kwargs",
]

NN_ALLOWED_VARIABLE_HYPERPARAMS: Final = {
    # All NN
    "input_chunk_length": {
        "min": 1,
        "max": 60,
        "default": 2,
    },
    "batch_size": {
        "min": 1,
        "max": 1000,
        "default": 32,
    },
    "dropout": {
        "min": 0.0,
        "max": 0.15,
        "default": 0.05,
    },
    "es_min_delta": {
        "min": 0.0,
        "max": 0.15,
        "default": 0.05,
    },
    "es_patience": {
        "min": 10,
        "max": 50,
        "default": 20,
    },
    "lr_factor": {
        "min": 0.0,
        "max": 0.9,
        "default": 0.1,
    },
    "lr_patience": {
        "min": 0,
        "max": 20,
        "default": 5,
    },
    # NBEATSModel
    "num_stacks": {
        "min": 1,
        "max": 50,
        "default": 30,
    },
    "num_blocks": {
        "min": 1,
        "max": 10,
        "default": 1,
    },
    "num_layers": {
        "min": 1,
        "max": 10,
        "default": 1,
    },
    "layer_widths": {
        "min": 16,
        "max": 512,
        "default": 256,
    },
    "expansion_coefficient_dim": {
        "min": 1,
        "max": 10,
        "default": 5,
    },
}

NN_FIXED_HYPERPARAMS: Final = {
    "n_epochs": 100,
    "loss_fn": LOSS_FN,
    "torch_metrics": METRIC_COLLECTION,
    "log_tensorboard": True,
    "lr_scheduler_cls": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "enable_progress_bar": True,
    "max_time": None,
}

INTEGER_HYPERPARAMS: Final = [
    "rebin_y",
    "input_chunk_length",
    "output_chunk_length",
    "n_epochs",
    "batch_size",
    "es_patience",
    "lr_patience",
    # NBEATSModel
    "num_stacks",
    "num_blocks",
    "num_layers",
    "layer_widths",
    "expansion_coefficient_dim",
]


################################################################################
# parent class
class TSModelWrapper:  # pylint: disable=too-many-instance-attributes
    """Parent class for all time series wrappers."""

    def __init__(
        self: "TSModelWrapper",
        # required
        dfp_trainable_evergreen: pd.DataFrame,
        dt_val_start_datetime_local: datetime.datetime,
        work_dir_base: str,
        random_state: int,
        date_fmt: str,
        time_fmt: str,
        fname_datetime_fmt: str,
        local_timezone: zoneinfo.ZoneInfo,
        # optional, will later load in child classes
        *,
        model_class: ForecastingModel | None = None,
        is_nn: bool | None = None,
        work_dir: str | None = None,
        model_name_tag: str | None = None,
        required_hyperparams_data: list[str] | None = None,
        required_hyperparams_model: list[str] | None = None,
        allowed_variable_hyperparams: dict | None = None,
        variable_hyperparams: dict | None = None,
        fixed_hyperparams: dict | None = None,
    ) -> None:
        """Int method.

        Args:
            dfp_trainable_evergreen: Time series data.
            dt_val_start_datetime_local: Date to cut the validation from the training set.
            work_dir_base: Top level directory for saving model files.
            random_state: Random seed.
            date_fmt: String format of dates.
            time_fmt: String format of times.
            fname_datetime_fmt: String format of date times for file names.
            local_timezone: Local timezone.
            model_class: Dart model class.
            is_nn: Flag for if the model is a neural network (NN).
            work_dir: Full path to directory to save this model's files.
            model_name_tag: Descriptive tag to add to the model name, optional.
            required_hyperparams_data: List of required data hyperparameters for this model.
            required_hyperparams_model: List of required hyperparameters for this model's constructor.
            allowed_variable_hyperparams: Dictionary of allowed variable hyperparameters for this model.
            variable_hyperparams: Dictionary of variable hyperparameters for this model.
            fixed_hyperparams: Dictionary of fixed hyperparameters for this model.
        """
        if required_hyperparams_data is None:
            required_hyperparams_data = []
        if required_hyperparams_model is None:
            required_hyperparams_model = []

        self.dfp_trainable_evergreen = dfp_trainable_evergreen
        self.dt_val_start_datetime_local = dt_val_start_datetime_local.replace(tzinfo=None)
        self.work_dir_base = work_dir_base
        self.random_state = random_state
        self.date_fmt = date_fmt
        self.time_fmt = time_fmt
        self.fname_datetime_fmt = fname_datetime_fmt
        self.local_timezone = local_timezone

        self.model_class = model_class
        self.is_nn = is_nn
        self.work_dir = work_dir
        self.model_name_tag = model_name_tag
        self.required_hyperparams_data = required_hyperparams_data
        self.required_hyperparams_model = required_hyperparams_model
        self.allowed_variable_hyperparams = allowed_variable_hyperparams
        self.variable_hyperparams = variable_hyperparams
        self.fixed_hyperparams = fixed_hyperparams

        self.model_name: str | None = None
        self.chosen_hyperparams: dict = {}
        self.model = None
        self.is_trained = False

    def __str__(self: "TSModelWrapper") -> str:
        """Redefine the str method.

        Returns:
            Description of model as str.
        """
        return f"""
{self.dfp_trainable_evergreen.index.size = }
{self.dt_val_start_datetime_local = }
{self.work_dir_base = }
{self.random_state = }
{self.date_fmt = }
{self.time_fmt = }
{self.fname_datetime_fmt = }
{self.local_timezone = }

{self.model_class = }
{self.is_nn = }
{self.work_dir = }
{self.model_name_tag = }
self.required_hyperparams_data = {pprint.pformat(self.required_hyperparams_data)}
self.required_hyperparams_model = {pprint.pformat(self.required_hyperparams_model)}
self.allowed_variable_hyperparams = {pprint.pformat(self.allowed_variable_hyperparams)}
self.variable_hyperparams = {pprint.pformat(self.variable_hyperparams)}
self.fixed_hyperparams = {pprint.pformat(self.fixed_hyperparams)}

{self.model_name = }
self.chosen_hyperparams = {pprint.pformat(self.chosen_hyperparams)}
{self.model = }
{self.is_trained = }
"""

    def reset_wrapper(self: "TSModelWrapper") -> None:
        """Reset the wrapper after training, used in Bayesian optimization."""
        self.model_name = None
        self.chosen_hyperparams = {}
        self.model = None
        self.is_trained = False

    def get_random_state(self: "TSModelWrapper", default_random_state: int = 42) -> int:
        """Get the random_state of this model wrapper.

        Args:
            default_random_state: Default random state to return if random_state is not set.

        Returns:
            Random state of this model wrapper.
        """
        random_state = self.random_state

        if random_state is None:
            random_state = default_random_state

        if TYPE_CHECKING:
            assert isinstance(random_state, int)  # noqa: SCS108 # nosec assert_used

        return random_state

    def get_model(self: "TSModelWrapper") -> ForecastingModel:
        """Get the model object from this model wrapper.

        Returns:
            Model object from this model wrapper
        """
        return self.model

    def set_enable_progress_bar_and_max_time(
        self: "TSModelWrapper",
        *,
        enable_progress_bar: bool,
        max_time: str | datetime.timedelta | dict[str, int] | None,
    ) -> None:
        """Set the enable_progress_bar flag for this model wrapper. Also configures torch warning messages about training devices and CUDA, globally, via the logging module.

        Args:
            enable_progress_bar: Enable torch progress bar during training.
            max_time: Set the maximum amount of time for training. Training will get interrupted mid-epoch.
        """
        _fixed_hyperparams = self.fixed_hyperparams
        if not _fixed_hyperparams:
            _fixed_hyperparams = {}
        _fixed_hyperparams["enable_progress_bar"] = enable_progress_bar
        _fixed_hyperparams["max_time"] = max_time

        self.fixed_hyperparams = _fixed_hyperparams

        # Turn off torch warning messages about training devices and CUDA
        # Adapted from
        # https://github.com/Lightning-AI/lightning/issues/13378#issuecomment-1170258489
        # and
        # https://github.com/Lightning-AI/lightning/issues/3431#issuecomment-1527945684
        logger_level = logging.WARNING
        if not enable_progress_bar:
            logger_level = logging.ERROR
        logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logger_level)
        logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logger_level)

    def set_work_dir(
        self: "TSModelWrapper",
        *,
        work_dir_relative_to_base: str | None = None,
        work_dir_absolute: str | None = None,
    ) -> None:
        """Set the work_dir for this model wrapper.

        Args:
            work_dir_relative_to_base: Set work_dir with this extension to work_dir_base.
            work_dir_absolute: Absolute path for work_dir, disregard work_dir_base.

        Raises:
            ValueError: Bad configuration.
        """
        if work_dir_relative_to_base is not None and work_dir_absolute is not None:
            raise ValueError("Can not use both parameters, choose one!")
        if work_dir_relative_to_base is not None and work_dir_relative_to_base != "":
            if self.work_dir_base is None:
                raise ValueError("Must have a valid work_dir_base!")
            self.work_dir = os.path.join(self.work_dir_base, work_dir_relative_to_base)
        elif (
            work_dir_absolute is not None and work_dir_absolute != ""
        ):  # pylint: disable=no-else-raise
            self.work_dir = work_dir_absolute
        else:
            raise ValueError("Must use at least one parameter!")

    def set_model_name_tag(
        self: "TSModelWrapper",
        model_name_tag: str,
    ) -> None:
        """Set the model_name_tag for this model wrapper.

        Args:
            model_name_tag: Descriptive tag to add to the model name, optional.
        """
        self.model_name_tag = model_name_tag

    def get_generic_model_name(self: "TSModelWrapper") -> str:
        """Get the generic name for this model, without a time stamp.

        Raises:
            ValueError: Bad configuration.

        Returns:
            The generic name for this model, without a time stamp.
        """
        if not issubclass(self.model_class, ForecastingModel):  # type: ignore[arg-type]
            raise ValueError("Unknown model name, should not happen!")

        if TYPE_CHECKING:
            assert isinstance(  # noqa: SCS108 # nosec assert_used
                self.model_class, ForecastingModel
            )

        if self.model_name_tag is not None and self.model_name_tag != "":
            _model_name_tag = f"_{self.model_name_tag}"
        else:
            _model_name_tag = ""

        return f"{self.model_class.__name__}{_model_name_tag}"

    def _name_model(self: "TSModelWrapper") -> None:
        """Name this model, with training time stamp."""
        self.model_name = f"{self.get_generic_model_name()}_{datetime.datetime.now(self.local_timezone).strftime(self.fname_datetime_fmt)}"

    def get_configurable_hyperparams(self: "TSModelWrapper", *, for_opt_only: bool = True) -> dict:
        """Get the configurable hyperparameters for this model.

        Args:
            for_opt_only: Flag to only return optimizable hyperparameters, i.e. exclude covariates and y_bin_edges.

        Returns:
            Dictionary of hyperparameters showing allowed values.
        """
        # construct model object
        if TYPE_CHECKING:
            assert isinstance(  # noqa: SCS108 # nosec assert_used
                self.allowed_variable_hyperparams, dict
            )
            assert isinstance(  # noqa: SCS108 # nosec assert_used
                self.required_hyperparams_model, list
            )
            assert isinstance(  # noqa: SCS108 # nosec assert_used
                self.required_hyperparams_data, list
            )

        hyperparams_to_return = [
            _
            for _ in self.required_hyperparams_model + self.required_hyperparams_data
            if not for_opt_only or _ not in ["covariates", "y_bin_edges"]
        ]

        return {
            k: v for k, v in self.allowed_variable_hyperparams.items() if k in hyperparams_to_return
        }

    def _assemble_hyperparams(  # noqa: C901 # pylint: disable=too-many-statements
        self: "TSModelWrapper",
    ) -> None:
        """Assemble the hyperparameters for this model instance.

        Raises:
            ValueError: Bad configuration.
        """
        required_hyperparams_all = []
        if isinstance(self.required_hyperparams_data, list):
            required_hyperparams_all += self.required_hyperparams_data
        if isinstance(self.required_hyperparams_model, list):
            required_hyperparams_all += self.required_hyperparams_model

        if not required_hyperparams_all or not (
            isinstance(self.allowed_variable_hyperparams, dict)
            and isinstance(self.variable_hyperparams, dict)
            and isinstance(self.fixed_hyperparams, dict)
        ):
            raise ValueError("Need to give model the hyperparams first, should not happen!")

        def get_hyperparam_value(hyperparam: str) -> str | float | int | None:
            """Get hyperparam value from variable and fixed hyperparams dicts.

            Args:
                hyperparam: Key to search for.

            Returns:
                hyperparam_value

            Raises:
                ValueError: Bad configuration.
            """
            if TYPE_CHECKING:
                assert isinstance(  # noqa: SCS108 # nosec assert_used
                    self.allowed_variable_hyperparams, dict
                )
                assert isinstance(  # noqa: SCS108 # nosec assert_used
                    self.variable_hyperparams, dict
                )
                assert isinstance(self.fixed_hyperparams, dict)  # noqa: SCS108 # nosec assert_used

            hyperparam_value = None
            if hyperparam in self.variable_hyperparams:
                hyperparam_value = self.variable_hyperparams[hyperparam]
            elif hyperparam in self.allowed_variable_hyperparams:
                hyperparam_value = self.allowed_variable_hyperparams[hyperparam]
                if isinstance(hyperparam_value, dict) and "default" in hyperparam_value:
                    hyperparam_value = hyperparam_value["default"]
            elif hyperparam in self.fixed_hyperparams:  # noqa: SIM908
                hyperparam_value = self.fixed_hyperparams[hyperparam]

            if hyperparam_value is None:
                raise ValueError(f"Could not find value for required hyperparameter {hyperparam}!")

            return hyperparam_value

        # prep time_bin_size parameter
        if "time_bin_size_in_minutes" in required_hyperparams_all:
            time_bin_size_in_minutes = get_hyperparam_value("time_bin_size_in_minutes")
            if TYPE_CHECKING:
                assert isinstance(  # noqa: SCS108 # nosec assert_used
                    time_bin_size_in_minutes, float
                )

            # Ensure that time_bin_size_in_minutes is a divisor of 60 minutes
            def get_closest_divisor(input_divisor: float, *, n: int = 60) -> int:
                """Get the integer divisor of n that is the closest to input_divisor.

                Args:
                    input_divisor: Input divisor to search around.
                    n: Number to divide.

                Returns:
                    Integer divisor.
                """
                if n % input_divisor == 0:  # noqa: S001
                    return int(input_divisor)
                divisors = sympy.divisors(n)
                delta = [abs(input_divisor - _) for _ in divisors]
                return divisors[delta.index(min(delta))]

            time_bin_size_in_minutes = get_closest_divisor(time_bin_size_in_minutes)

            self.chosen_hyperparams["time_bin_size_in_minutes"] = time_bin_size_in_minutes
            self.chosen_hyperparams["time_bin_size"] = datetime.timedelta(
                minutes=time_bin_size_in_minutes
            )

        # set required hyperparams
        for hyperparam in required_hyperparams_all:
            if hyperparam == "model_name":
                hyperparam_value: Any = self.model_name
            elif hyperparam == "random_state":
                hyperparam_value = self.random_state
            elif hyperparam == "work_dir":
                hyperparam_value = self.work_dir
            elif hyperparam == "rebin_y":
                hyperparam_value = get_hyperparam_value(hyperparam)
                if hyperparam_value:
                    self.chosen_hyperparams["y_bin_edges"] = get_hyperparam_value("y_bin_edges")
                else:
                    self.chosen_hyperparams["y_bin_edges"] = None
            elif hyperparam in [  # noqa: R507
                # these are not needed
                "y_bin_edges",
                "time_bin_size",
                "time_bin_size_in_minutes",
                "prediction_length_in_minutes",
                "es_min_delta",
                "es_patience",
                "enable_progress_bar",
                "max_time",
                "lr_factor",
                "lr_patience",
            ]:
                continue
            elif hyperparam == "output_chunk_length":
                prediction_length_in_minutes = get_hyperparam_value("prediction_length_in_minutes")
                if TYPE_CHECKING:
                    assert isinstance(  # noqa: SCS108 # nosec assert_used
                        prediction_length_in_minutes, float
                    )
                prediction_length = datetime.timedelta(minutes=prediction_length_in_minutes)
                self.chosen_hyperparams["output_chunk_length"] = math.ceil(
                    prediction_length.seconds / self.chosen_hyperparams["time_bin_size"].seconds
                )
                self.chosen_hyperparams["prediction_length_in_minutes"] = (
                    self.chosen_hyperparams["output_chunk_length"]
                    * self.chosen_hyperparams["time_bin_size_in_minutes"]
                )

                continue
            elif hyperparam == "pl_trainer_kwargs":
                self.chosen_hyperparams["es_min_delta"] = get_hyperparam_value("es_min_delta")
                self.chosen_hyperparams["es_patience"] = get_hyperparam_value("es_patience")
                self.chosen_hyperparams["enable_progress_bar"] = get_hyperparam_value(
                    "enable_progress_bar"
                )
                self.chosen_hyperparams["max_time"] = get_hyperparam_value("max_time")
                hyperparam_value = get_pl_trainer_kwargs(
                    self.chosen_hyperparams["es_min_delta"],
                    self.chosen_hyperparams["es_patience"],
                    enable_progress_bar=self.chosen_hyperparams["enable_progress_bar"],
                    max_time=self.chosen_hyperparams["max_time"],
                    # Could set log_every_n_steps = 1 to avoid a warning, but that would make large logs, so just use filterwarnings instead
                    # https://github.com/Lightning-AI/lightning/issues/10644
                    # https://github.com/Lightning-AI/lightning/blob/c68ff6482f97f388d6b0c37151fafc0fae789094/src/lightning/pytorch/loops/fit_loop.py#L292-L298
                )
            elif hyperparam == "lr_scheduler_kwargs":
                self.chosen_hyperparams["lr_factor"] = get_hyperparam_value("lr_factor")
                self.chosen_hyperparams["lr_patience"] = get_hyperparam_value("lr_patience")
                hyperparam_value = get_lr_scheduler_kwargs(
                    self.chosen_hyperparams["lr_factor"], self.chosen_hyperparams["lr_patience"]
                )
            else:
                hyperparam_value = get_hyperparam_value(hyperparam)

            self.chosen_hyperparams[hyperparam] = hyperparam_value

        # make sure int hyperparameters are int, as Bayesian optimization will always give floats
        # and check chosen hyperparams are in the allowed ranges / sets
        for k, v in self.chosen_hyperparams.items():
            if k in INTEGER_HYPERPARAMS:
                self.chosen_hyperparams[k] = int(v)
            if k in list(self.fixed_hyperparams.keys()) + ["y_bin_edges"]:
                pass
            elif k in self.allowed_variable_hyperparams:
                allowed_min = self.allowed_variable_hyperparams[k].get("min")
                allowed_max = self.allowed_variable_hyperparams[k].get("max")
                allowed_values = self.allowed_variable_hyperparams[k].get("allowed")
                if allowed_min is not None and allowed_max is not None:
                    if v < allowed_min or allowed_max < v:
                        raise ValueError(
                            f"Hyperparameter {k} with value {v} is not allowed, expected to be between {allowed_min} and {allowed_max}!"
                        )
                elif allowed_values is not None and isinstance(allowed_values, list):
                    if not set(v).issubset(set(allowed_values)):
                        raise ValueError(
                            f"Hyperparameter {k} with value {v} is not allowed, expected to be a subset of {allowed_values}!"
                        )
                else:
                    raise ValueError(f"Hyperparameter {k} with value {v} can not be checked!")
            if k == "time_bin_size_in_minutes" and 60 % v != 0:
                raise ValueError(
                    f"Hyperparameter {k} with value {v} is not allowed, {60 % v = } should be 0!"
                )

    def preview_hyperparameters(self: "TSModelWrapper", **kwargs: float) -> dict:
        """Return hyperparameters the model would actually run with if trained now, used in Bayesian optimization.

        Args:
            **kwargs: Hyperparameters to change.

        Returns:
            Hyperparameters the model would actually run with if trained now.
        """
        if kwargs:
            self.variable_hyperparams = kwargs
        self._assemble_hyperparams()
        return self.chosen_hyperparams

    def train_model(self: "TSModelWrapper", **kwargs: float) -> float:
        """Train the model and return loss.

        Args:
            **kwargs: Hyperparameters to change, used in Bayesian optimization.

        Returns:
            Loss.
        """
        # setup hyperparams
        self._name_model()
        _ = self.preview_hyperparameters(**kwargs)

        # construct model object
        if TYPE_CHECKING:
            assert isinstance(  # noqa: SCS108 # nosec assert_used
                self.model_class, ForecastingModel
            )
            assert isinstance(  # noqa: SCS108 # nosec assert_used
                self.required_hyperparams_model, list
            )

        chosen_hyperparams_model = {
            k: v for k, v in self.chosen_hyperparams.items() if k in self.required_hyperparams_model
        }

        self.model = self.model_class(**chosen_hyperparams_model)
        if TYPE_CHECKING:
            assert isinstance(self.model, ForecastingModel)  # noqa: SCS108 # nosec assert_used

        # data prep
        time_bin_size = self.chosen_hyperparams["time_bin_size"]
        freq_str = f'{self.chosen_hyperparams["time_bin_size_in_minutes"]}min'
        y_bin_edges = self.chosen_hyperparams["y_bin_edges"]

        dfp_trainable = rebin_chance_of_showers_time_series(
            self.dfp_trainable_evergreen,
            "ds",
            "y",
            time_bin_size=time_bin_size,
            other_cols_to_agg_dict={"had_flow": "max"},
            y_bin_edges=y_bin_edges,
        )

        dfp_trainable = create_datetime_component_cols(
            dfp_trainable, datetime_col="ds", date_fmt=self.date_fmt, time_fmt=self.time_fmt
        )

        dart_series_y_trainable = TimeSeries.from_dataframe(
            dfp_trainable,
            "ds",
            "y",
            freq=freq_str,
            fill_missing_dates=True,
        )

        # setup covariates
        covariates_type = None
        if not self.model.supports_future_covariates and self.model.supports_past_covariates:
            covariates_type = "past"
        elif self.model.supports_future_covariates:
            covariates_type = "future"

        if covariates_type is not None:
            dart_series_covariates_trainable = TimeSeries.from_dataframe(
                dfp_trainable,
                "ds",
                self.chosen_hyperparams["covariates"],
                freq=freq_str,
                fill_missing_dates=True,
            )

        # fill missing values
        frac_missing = missing_values_ratio(dart_series_y_trainable)
        if 0 < frac_missing:
            # Missing {frac_missing:.2%} of values, filling via interpolation
            dart_series_y_trainable = fill_missing_values(dart_series_y_trainable)
            if covariates_type is not None:
                dart_series_covariates_trainable = fill_missing_values(
                    dart_series_covariates_trainable
                )

        # split train and validation sets
        dart_series_y_train, dart_series_y_val = dart_series_y_trainable.split_before(
            pd.Timestamp(self.dt_val_start_datetime_local)
        )

        model_covariates_kwargs = {}
        prediction_covariates_kwargs = {}
        if covariates_type is not None:
            (
                dart_series_covariates_train,
                dart_series_covariates_val,
            ) = dart_series_covariates_trainable.split_before(
                pd.Timestamp(self.dt_val_start_datetime_local)
            )
            model_covariates_kwargs[f"{covariates_type}_covariates"] = dart_series_covariates_train
            model_covariates_kwargs[
                f"val_{covariates_type}_covariates"
            ] = dart_series_covariates_val
            prediction_covariates_kwargs[
                f"{covariates_type}_covariates"
            ] = dart_series_covariates_train.append(dart_series_covariates_val)

        # train
        _ = self.model.fit(
            dart_series_y_train,
            val_series=dart_series_y_val,
            verbose=self.chosen_hyperparams.get("enable_progress_bar", False),
            **model_covariates_kwargs,
        )

        # measure loss on validation
        y_pred_val = self.model.predict(
            dart_series_y_val.n_timesteps,
            num_samples=1,
            verbose=self.chosen_hyperparams.get("enable_progress_bar", False),
            **prediction_covariates_kwargs,
        )

        y_val_tensor = torch.Tensor(dart_series_y_val["y"].values())
        y_pred_val_tensor = torch.Tensor(y_pred_val["y"].values())

        loss = float(LOSS_FN(y_val_tensor, y_pred_val_tensor))

        # clean up
        # small objects, can ignore: time_bin_size, freq_str, y_bin_edges, covariates_type
        del y_val_tensor
        del y_pred_val_tensor

        del y_pred_val

        del dart_series_y_val
        del dart_series_y_train
        del dart_series_y_trainable

        if covariates_type is not None:
            del model_covariates_kwargs
            del prediction_covariates_kwargs
            del dart_series_covariates_val
            del dart_series_covariates_train
            del dart_series_covariates_trainable

        del dfp_trainable

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # return negative loss as we want to maximize the target, and set the is_trained flag
        self.is_trained = True
        return -loss


################################################################################
# model classes
class NBEATSModelWrapper(TSModelWrapper):
    """NBEATSModel wrapper."""

    # config wrapper for NBEATSModel
    _model_class = NBEATSModel
    _is_nn = True
    _required_hyperparams_data = DATA_REQUIRED_HYPERPARAMS
    _required_hyperparams_model = NN_REQUIRED_HYPERPARAMS + [
        "num_stacks",
        "num_blocks",
        "num_layers",
        "layer_widths",
        "expansion_coefficient_dim",
    ]
    _allowed_variable_hyperparams = {**DATA_VARIABLE_HYPERPARAMS, **NN_ALLOWED_VARIABLE_HYPERPARAMS}
    _fixed_hyperparams = {**DATA_FIXED_HYPERPARAMS, **NN_FIXED_HYPERPARAMS}

    def __init__(self: "NBEATSModelWrapper", **kwargs: Any) -> None:  # noqa: ANN401
        # boilerplate - the same for all models below here
        """Int method.

        Args:
            **kwargs: Keyword arguments.
                Can be a parent TSModelWrapper instance plus the undefined parameters,
                or all the necessary parameters.
        """
        # NOTE using `isinstance(kwargs["TSModelWrapper"], TSModelWrapper)`,
        # or even `issubclass(type(kwargs["TSModelWrapper"]), TSModelWrapper)` would be preferable
        # but they do not work if the kwargs["TSModelWrapper"] parent instance was updated between child __init__ calls
        if (
            "TSModelWrapper" in kwargs
            and type(kwargs["TSModelWrapper"].__class__)  # pylint: disable=unidiomatic-typecheck
            == type(TSModelWrapper)  # <class 'type'>
            and str(kwargs["TSModelWrapper"].__class__)
            == str(TSModelWrapper)  # <class 'utils.TSModelWrappers.TSModelWrapper'>
        ):
            self.__dict__ = kwargs["TSModelWrapper"].__dict__.copy()
            self.model_class = self._model_class
            self.is_nn = self._is_nn
            self.work_dir = kwargs.get("work_dir")
            self.model_name_tag = kwargs.get("model_name_tag")
            self.required_hyperparams_data = self._required_hyperparams_data
            self.required_hyperparams_model = self._required_hyperparams_model
            self.allowed_variable_hyperparams = self._allowed_variable_hyperparams
            self.variable_hyperparams = kwargs.get("variable_hyperparams", {})
            self.fixed_hyperparams = self._fixed_hyperparams
        else:
            super().__init__(
                dfp_trainable_evergreen=kwargs["dfp_trainable_evergreen"],
                dt_val_start_datetime_local=kwargs["dt_val_start_datetime_local"],
                work_dir_base=kwargs["work_dir_base"],
                random_state=kwargs["random_state"],
                date_fmt=kwargs["date_fmt"],
                time_fmt=kwargs["time_fmt"],
                fname_datetime_fmt=kwargs["fname_datetime_fmt"],
                local_timezone=kwargs["local_timezone"],
                model_class=self._model_class,
                is_nn=self._is_nn,
                work_dir=kwargs["work_dir"],
                model_name_tag=kwargs.get("model_name_tag"),
                required_hyperparams_data=self._required_hyperparams_data,
                required_hyperparams_model=self._required_hyperparams_model,
                allowed_variable_hyperparams=self._allowed_variable_hyperparams,
                variable_hyperparams=kwargs.get("variable_hyperparams"),
                fixed_hyperparams=self._fixed_hyperparams,
            )


################################################################################
# Setup Bayesian optimization
n_points = 0  # # pylint: disable=invalid-name


def run_bayesian_opt(  # noqa: C901 # pylint: disable=too-many-statements,too-many-locals
    parent_wrapper: TSModelWrapper,
    model_wrapper_class: type[NBEATSModelWrapper],  # expand with more classes
    *,
    hyperparams_to_opt: list[str] | None = None,
    n_iter: int = 100,
    allow_duplicate_points: bool = False,
    utility_kind: str = "ucb",
    utility_kappa: float = 2.576,
    verbose: int = 2,
    display_memory_usage: bool = False,
    enable_progress_bar: bool = False,
    max_time_per_model: str | datetime.timedelta | dict[str, int] | None = None,
    enable_json_logging: bool = True,
    enable_reloading: bool = True,
    enable_model_saves: bool = False,
    bayesian_opt_work_dir_name: str = "bayesian_optimization",
) -> tuple[dict, bayes_opt.BayesianOptimization]:
    """Run Bayesian optimization for this model wrapper.

    Args:
        parent_wrapper: TSModelWrapper object containing all parent configs.
        model_wrapper_class: TSModelWrapper class to optimize.
        hyperparams_to_opt: List of hyperparameters to optimize.
            If None, use all configurable hyperparameters.
        n_iter: How many iterations of Bayesian optimization to perform.
            This is the number of new models to train, in addition to any duplicated or reloaded points.
        allow_duplicate_points: If True, the optimizer will allow duplicate points to be registered.
            This behavior may be desired in high noise situations where repeatedly probing
            the same point will give different answers. In other situations, the acquisition
            may occasionally generate a duplicate point.
        utility_kind: {'ucb', 'ei', 'poi'}
            * 'ucb' stands for the Upper Confidence Bounds method
            * 'ei' is the Expected Improvement method
            * 'poi' is the Probability Of Improvement criterion.
        utility_kappa: Parameter to indicate how closed are the next parameters sampled.
            Higher value = favors spaces that are least explored.
            Lower value = favors spaces where the regression function is the highest.
        verbose: Optimizer verbosity, 2 prints all iterations, 1 prints only when a maximum is observed, and 0 is silent.
        display_memory_usage: Print memory usage at each training iteration.
        enable_progress_bar: Enable torch progress bar during training.
        max_time_per_model: Set the maximum amount of time for NN training. Training will get interrupted mid-epoch.
        enable_json_logging: Enable JSON logging of points.
        enable_reloading: Enable reloading of prior points from JSON log.
        enable_model_saves: Save the trained model at each iteration.
        bayesian_opt_work_dir_name: Directory name to save logs and models in, within the parent_wrapper.work_dir_base.

    Returns:
        optimal_values: Optimal hyperparameter values.
        optimizer: bayes_opt.BayesianOptimization object for further details.

    Raises:
        ValueError: Bad configuration.
        RuntimeError: Run time error that is not "out of memory".
    """
    global n_points

    # Setup hyperparameters
    _model_wrapper = model_wrapper_class(TSModelWrapper=parent_wrapper)
    configurable_hyperparams = _model_wrapper.get_configurable_hyperparams()
    if hyperparams_to_opt is None:
        hyperparams_to_opt = list(configurable_hyperparams.keys())

    # Setup hyperparameter bounds
    hyperparam_bounds = {}
    for hyperparam in hyperparams_to_opt:
        hyperparam_min = configurable_hyperparams.get(hyperparam, {}).get("min")
        hyperparam_max = configurable_hyperparams.get(hyperparam, {}).get("max")
        if hyperparam_min is None or hyperparam_max is None:
            raise ValueError(f"Could not load hyperparameter definition for {hyperparam = }!")
        hyperparam_bounds[hyperparam] = (hyperparam_min, hyperparam_max)

    # Setup Bayesian optimization objects
    # https://github.com/bayesian-optimization/BayesianOptimization/blob/11a0c6aba1fcc6b5d2716052da5222a84259c5b9/bayes_opt/util.py#L113
    utility = bayes_opt.UtilityFunction(kind=utility_kind, kappa=utility_kappa)

    optimizer = bayes_opt.BayesianOptimization(
        f=None,
        pbounds=hyperparam_bounds,
        random_state=_model_wrapper.get_random_state(),
        verbose=verbose,
        allow_duplicate_points=allow_duplicate_points,
    )

    # Setup Logging
    generic_model_name: Final = _model_wrapper.get_generic_model_name()
    bayesian_opt_work_dir: Final = os.path.expanduser(
        os.path.join(_model_wrapper.work_dir_base, bayesian_opt_work_dir_name, generic_model_name)
    )

    fname_json_log: Final = os.path.join(
        bayesian_opt_work_dir, f"bayesian_opt_{generic_model_name}.json"
    )

    # Reload prior points, must be done before json_logger is recreated to avoid duplicating past runs
    n_points = 0
    if enable_reloading and os.path.isfile(fname_json_log):
        print(f"Resuming Bayesian optimization from:\n{fname_json_log}\n")
        optimizer.dispatch(Events.OPTIMIZATION_START)
        load_logs(optimizer, logs=fname_json_log)
        n_points = len(optimizer.space)
        print(f"Loaded {n_points} existing points.\n")

    # Continue to setup logging
    if enable_json_logging:
        os.makedirs(bayesian_opt_work_dir, exist_ok=True)
        json_logger = JSONLogger(path=fname_json_log, reset=False)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, json_logger)

    if 0 < verbose:
        screen_logger = ScreenLogger(verbose=verbose)
        for event in DEFAULT_EVENTS:
            optimizer.subscribe(event, screen_logger)

    # Define function to complete an iteration
    def complete_iter(
        model_wrapper: TSModelWrapper,
        target: float,
        point_to_probe: dict,
        *,
        probed_point: dict | None = None,
    ) -> None:
        """Complete this iteration, register point(s) and clean up.

        Args:
            model_wrapper: Model wrapper object to rest.
            target: Target value to register.
            point_to_probe: Raw point to probe.
            probed_point: Point that was actually probed.
        """
        global n_points
        optimizer.register(params=point_to_probe, target=target)
        n_points += 1
        if probed_point:
            optimizer.register(params=probed_point, target=target)
            n_points += 1

        model_wrapper.reset_wrapper()
        del model_wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if display_memory_usage:
            print_memory_usage()
        print(f"Completed {i_iter = }, with {n_points = }")

    # clean up _model_wrapper
    del _model_wrapper

    # run Bayesian optimization iterations
    try:
        for i_iter in range(n_iter):
            if i_iter == 0:
                optimizer.dispatch(Events.OPTIMIZATION_START)
            print(f"\nStarting {i_iter = }, with {n_points = }")
            next_point_to_probe = optimizer.suggest(utility)

            # Create a fresh model_wrapper object to try to avoid GPU memory leaks TODO probably can safely revert
            model_wrapper = model_wrapper_class(TSModelWrapper=parent_wrapper)
            model_wrapper.set_work_dir(work_dir_absolute=bayesian_opt_work_dir)
            model_wrapper.set_enable_progress_bar_and_max_time(
                enable_progress_bar=enable_progress_bar, max_time=max_time_per_model
            )

            # Check if we already tested this chosen_hyperparams point
            # If it has been tested, save the raw next_point_to_probe with the same target and continue
            chosen_hyperparams = model_wrapper.preview_hyperparameters(**next_point_to_probe)
            next_point_to_probe_cleaned = {k: chosen_hyperparams[k] for k in hyperparams_to_opt}

            is_duplicate_point = False
            for i_param in range(optimizer.space.params.shape[0]):
                if np.array_equiv(
                    optimizer.space.params_to_array(next_point_to_probe_cleaned),
                    optimizer.space.params[i_param],
                ):
                    target = optimizer.space.target[i_param]
                    print(
                        f"On iteration {i_iter} testing prior point {i_param}, returning prior {target = } for the raw next_point_to_probe."
                    )
                    complete_iter(model_wrapper, target, next_point_to_probe)
                    is_duplicate_point = True
                    break
            if is_duplicate_point:
                continue

            # set model_name_tag for this iteration
            model_wrapper.set_model_name_tag(model_name_tag=f"iteration_{n_points}")

            # train the model
            try:
                target = model_wrapper.train_model(**next_point_to_probe)
            except RuntimeError as error:
                if "out of memory" in str(error):
                    print("Ran out of memory, returning -inf as loss")
                    complete_iter(
                        model_wrapper,
                        -float("inf"),
                        next_point_to_probe,
                        probed_point=next_point_to_probe_cleaned,
                    )
                    continue
                raise error

            if enable_model_saves:
                fname_model = os.path.join(
                    bayesian_opt_work_dir, f"iteration_{n_points}_{generic_model_name}.pt"
                )
                model_wrapper.get_model().save(fname_model)

            # Register the point
            complete_iter(
                model_wrapper, target, next_point_to_probe, probed_point=next_point_to_probe_cleaned
            )

    except bayes_opt.util.NotUniqueError as error:
        print(
            str(error).replace(
                '. You can set "allow_duplicate_points=True" to avoid this error', ""
            )
            + ", stopping optimization here."
        )
    except Exception as error:
        print(
            f"Unexpected error in run_bayesian_opt():\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nReturning with current objects."
        )
    optimizer.dispatch(Events.OPTIMIZATION_END)

    return optimizer.max, optimizer
