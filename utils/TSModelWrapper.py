# pylint: disable=invalid-name
"""Wrapper class for time series models."""
# pylint: enable=invalid-name


import datetime
import gc
import logging
import math
import operator
import pathlib
import pprint
import sys
import warnings
import zoneinfo
from typing import TYPE_CHECKING, Any, Final

import pandas as pd
import sympy
import torch
import torchmetrics
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils.missing_values import fill_missing_values, missing_values_ratio
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from utils.shared_functions import (
    create_datetime_component_cols,
    rebin_chance_of_showers_time_series,
)

################################################################################
# Setup global parameters

# Control warnings

# torch
warnings.filterwarnings(
    "ignore",
    message="The number of training batches",
    category=PossibleUserWarning,
)

# prophet / cmdstanpy
logger_cmdstanpy = logging.getLogger("cmdstanpy")
logger_cmdstanpy.addHandler(logging.NullHandler())
logger_cmdstanpy.propagate = False
logger_cmdstanpy.setLevel(logging.ERROR)

# logging
logger_ts_wrapper = logging.getLogger(__name__)
logger_ts_wrapper.setLevel(logging.INFO)
if not logger_ts_wrapper.handlers:
    logger_ts_wrapper.addHandler(logging.StreamHandler(sys.stdout))

# loss function
LOSS_FN: Final = torchmetrics.MeanSquaredError()

# metrics to log at each epoch
METRIC_COLLECTION: Final = torchmetrics.MetricCollection(
    [
        torchmetrics.MeanSquaredError(),
        torchmetrics.MeanAbsolutePercentageError(),
    ]
)


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
def get_lr_scheduler_kwargs(lr_factor: float, lr_patience: int, verbose: int) -> dict:
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
        verbose: If non-zero, prints a message to stdout for each update.

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
        "verbose": bool(verbose),
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
        "type": bool,
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
    "random_state",
]

NN_ALLOWED_VARIABLE_HYPERPARAMS: Final = {
    # All NN
    "input_chunk_length": {
        "min": 1,
        "max": 60,
        "default": 2,
        "type": int,
    },
    "batch_size": {
        "min": 1,
        "max": 1000,
        "default": 32,
        "type": int,
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
        "type": int,
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
        "type": int,
    },
    # NBEATSModel and NHiTSModel
    "num_stacks": {
        "min": 1,
        "max": 50,
        "default": 30,
        "type": int,
    },
    "num_blocks": {
        "min": 1,
        "max": 10,
        "default": 1,
        "type": int,
    },
    "num_layers": {  # and TCNModel
        "min": 1,
        "max": 10,
        "default": 1,
        "type": int,
    },
    "layer_widths": {
        "min": 16,
        "max": 1024,
        "default": 256,
        "type": int,
    },
    "expansion_coefficient_dim": {  # NBEATSModel only
        "min": 1,
        "max": 10,
        "default": 5,
        "type": int,
    },
    "MaxPool1d": {  # NHiTSModel only
        "min": 0,
        "max": 1,
        "default": 1,
        "type": bool,
    },
    # TCNModel
    "kernel_size": {  # and DLinearModel
        "min": 1,
        "max": 100,
        "default": 50,
        "type": int,
    },
    "num_filters": {
        "min": 0,
        "max": 10,
        "default": 3,
        "type": int,
    },
    "dilation_base": {
        "min": 1,
        "max": 10,
        "default": 2,
        "type": int,
    },
    "weight_norm": {
        "min": 0,
        "max": 1,
        "default": 1,
        "type": bool,
    },
    # TransformerModel
    "d_model": {
        "min": 0,
        "max": 128,
        "default": 64,
        "type": int,
    },
    "nhead": {
        "min": 0,
        "max": 20,
        "default": 4,
        "type": int,
    },
    "num_encoder_layers": {  # and TiDEModel
        "min": 0,
        "max": 20,
        "default": 3,
        "type": int,
    },
    "num_decoder_layers": {  # and TiDEModel
        "min": 0,
        "max": 20,
        "default": 3,
        "type": int,
    },
    "dim_feedforward": {
        "min": 0,
        "max": 1024,
        "default": 512,
        "type": int,
    },
    # TFTModel
    "hidden_size": {  # and TiDEModel
        "min": 1,
        "max": 256,
        "default": 16,
        "type": int,
    },
    "lstm_layers": {
        "min": 1,
        "max": 5,
        "default": 1,
        "type": int,
    },
    "num_attention_heads": {
        "min": 1,
        "max": 10,
        "default": 4,
        "type": int,
    },
    "full_attention": {
        "min": 0,
        "max": 1,
        "default": 0,
        "type": bool,
    },
    "hidden_continuous_size": {
        "min": 1,
        "max": 20,
        "default": 8,
        "type": int,
    },
    # DLinearModel and NLinearModel
    "const_init": {
        "min": 0,
        "max": 1,
        "default": 1,
        "type": bool,
    },
    # NLinearModel
    "normalize": {
        "min": 0,
        "max": 1,
        "default": 0,
        "type": bool,
    },
    # TiDEModel
    "decoder_output_dim": {
        "min": 1,
        "max": 50,
        "default": 16,
        "type": int,
    },
    "temporal_width_past": {
        "min": 1,
        "max": 10,
        "default": 4,
        "type": int,
    },
    "temporal_width_future": {
        # We only have 1 covariate, so temporal_width_future should either be 0 or 1, but is still an int. See:
        # https://github.com/unit8co/darts/blob/962fd78cb526887c47bddc33bea4b731adf72a87/darts/models/forecasting/tide_model.py#L668-L672
        "min": 0,
        "max": 1,
        "default": 1,
        "type": int,
    },
    "temporal_decoder_hidden": {
        "min": 1,
        "max": 64,
        "default": 32,
        "type": int,
    },
    "use_layer_norm": {
        "min": 0,
        "max": 1,
        "default": 0,
        "type": bool,
    },
    # RNNModel and BlockRNNModel
    "hidden_dim": {
        "min": 1,
        "max": 50,
        "default": 25,
        "type": int,
    },
    "n_rnn_layers": {
        "min": 1,
        "max": 20,
        "default": 1,
        "type": int,
    },
    "training_length": {  # RNNModel only
        "min": 1,
        "max": 50,
        "default": 24,
        "type": int,
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

TREE_REQUIRED_HYPERPARAMS: Final = [
    "output_chunk_length",
    "lags",
    "lags_past_covariates",
    "multi_models",
    "random_state",
]

TREE_ALLOWED_VARIABLE_HYPERPARAMS: Final = {
    # All Trees
    "lags": {
        "min": 1,
        "max": 100,
        "default": 10,
        "type": int,
    },
    "lags_past_covariates": {
        "min": 1,
        "max": 100,
        "default": 10,
        "type": int,
    },
    "multi_models": {
        "min": 0,
        "max": 1,
        "default": 1,
        "type": bool,
    },
    # RandomForest
    "n_estimators": {
        "min": 1,
        "max": 300,
        "default": 100,
        "type": int,
    },
    "max_depth": {
        "min": 2,
        "max": 30,
        "default": 10,
        "type": int,
    },
}

VARIABLE_HYPERPARAMS: Final = {
    **DATA_VARIABLE_HYPERPARAMS,
    **NN_ALLOWED_VARIABLE_HYPERPARAMS,
    **TREE_ALLOWED_VARIABLE_HYPERPARAMS,
}

boolean_hyperparams = []
integer_hyperparams = [
    "output_chunk_length",
    "n_epochs",
]
for _k, _v in VARIABLE_HYPERPARAMS.items():
    if _k == "y_bin_edges":
        continue
    if TYPE_CHECKING:
        assert isinstance(_v, dict)  # noqa: SCS108 # nosec assert_used
    if _v.get("type") == bool:
        boolean_hyperparams.append(_k)
    elif _v.get("type") == int:
        integer_hyperparams.append(_k)


################################################################################
# parent class
class TSModelWrapper:  # pylint: disable=too-many-instance-attributes
    """Parent class for all time series wrappers."""

    def __init__(
        self: "TSModelWrapper",
        # required
        dfp_trainable_evergreen: pd.DataFrame,
        dt_val_start_datetime_local: datetime.datetime,
        work_dir_base: pathlib.Path,
        random_state: int,
        date_fmt: str,
        time_fmt: str,
        fname_datetime_fmt: str,
        local_timezone: zoneinfo.ZoneInfo,
        # optional, will later load in child classes
        *,
        model_class: ForecastingModel | None = None,
        is_nn: bool | None = None,
        verbose: int = 1,
        work_dir: pathlib.Path | None = None,
        model_name_tag: str | None = None,
        required_hyperparams_data: list[str] | None = None,
        required_hyperparams_model: list[str] | None = None,
        allowed_variable_hyperparams: dict | None = None,
        variable_hyperparams: dict | None = None,
        fixed_hyperparams: dict | None = None,
        hyperparams_conditions: list[dict] | None = None,
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
            verbose: Verbosity level.
            work_dir: Full path to directory to save this model's files.
            model_name_tag: Descriptive tag to add to the model name, optional.
            required_hyperparams_data: List of required data hyperparameters for this model.
            required_hyperparams_model: List of required hyperparameters for this model's constructor.
            allowed_variable_hyperparams: Dictionary of allowed variable hyperparameters for this model.
            variable_hyperparams: Dictionary of variable hyperparameters for this model.
            fixed_hyperparams: Dictionary of fixed hyperparameters for this model.
            hyperparams_conditions: List of dictionaries with hyperparameter conditions for this model.
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
        self.verbose = verbose
        self.work_dir = work_dir
        self.model_name_tag = model_name_tag
        self.required_hyperparams_data = required_hyperparams_data
        self.required_hyperparams_model = required_hyperparams_model
        self.allowed_variable_hyperparams = allowed_variable_hyperparams
        self.variable_hyperparams = variable_hyperparams
        self.fixed_hyperparams = fixed_hyperparams
        self.hyperparams_conditions = hyperparams_conditions

        self.model_name: str | None = None
        self.chosen_hyperparams: dict | None = None
        self.model = None
        self.is_trained = False

        if 0 < self.verbose:
            logger_ts_wrapper.setLevel(logging.DEBUG)

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
{self.verbose = }
{self.work_dir = }
{self.model_name_tag = }
self.required_hyperparams_data = {pprint.pformat(self.required_hyperparams_data)}
self.required_hyperparams_model = {pprint.pformat(self.required_hyperparams_model)}
self.variable_hyperparams = {pprint.pformat(self.variable_hyperparams)}
self.fixed_hyperparams = {pprint.pformat(self.fixed_hyperparams)}
self.hyperparams_conditions = {pprint.pformat(self.hyperparams_conditions)}

{self.model_name = }
self.chosen_hyperparams = {pprint.pformat(self.chosen_hyperparams)}
{self.model = }
{self.is_trained = }
"""

    def reset_wrapper(self: "TSModelWrapper") -> None:
        """Reset the wrapper after training, used in Bayesian optimization."""
        self.model_name = None
        self.chosen_hyperparams = None
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

    def get_n_prediction_steps_and_time_bin_size(
        self: "TSModelWrapper",
    ) -> tuple[int, datetime.timedelta]:
        """Get number of prediction steps from prediction_length_in_minutes and time_bin_size, used for Prophet predictions.

        Raises:
            ValueError: Bad configuration.

        Returns:
            Number of prediction steps and time bin size.
        """
        if not (
            isinstance(self.fixed_hyperparams, dict) and isinstance(self.chosen_hyperparams, dict)
        ):
            raise ValueError("Need to assemble the hyperparams first!")

        prediction_length_in_minutes = self.fixed_hyperparams.get("prediction_length_in_minutes")
        if not isinstance(prediction_length_in_minutes, (int, float)):
            raise ValueError("Could not load prediction_length_in_minutes from fixed_hyperparams!")

        time_bin_size = self.chosen_hyperparams.get("time_bin_size")
        if not isinstance(time_bin_size, datetime.timedelta):
            raise ValueError("Could not load time_bin_size from chosen_hyperparams!")

        prediction_length = datetime.timedelta(minutes=prediction_length_in_minutes)
        n_prediction_steps = math.ceil(prediction_length.seconds / time_bin_size.seconds)

        return n_prediction_steps, time_bin_size

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
        work_dir_relative_to_base: pathlib.Path | None = None,
        work_dir_absolute: pathlib.Path | None = None,
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
        if work_dir_relative_to_base is not None:
            if self.work_dir_base is None:
                raise ValueError("Must have a valid work_dir_base!")
            self.work_dir = pathlib.Path(self.work_dir_base, work_dir_relative_to_base)
        elif work_dir_absolute is not None:  # pylint: disable=no-else-raise
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
        self.chosen_hyperparams = {}
        required_hyperparams_all = []
        if isinstance(self.required_hyperparams_data, list):
            required_hyperparams_all += self.required_hyperparams_data
        if isinstance(self.required_hyperparams_model, list):
            required_hyperparams_all += self.required_hyperparams_model

        # Remove duplicates, e.g. random_state, while preserving order
        required_hyperparams_all = list(dict.fromkeys(required_hyperparams_all))

        if not required_hyperparams_all or not (
            isinstance(self.allowed_variable_hyperparams, dict)
            and isinstance(self.variable_hyperparams, dict)
            and isinstance(self.fixed_hyperparams, dict)
        ):
            raise ValueError("Need to give model the hyperparams first, should not happen!")

        def get_hyperparam_value(
            hyperparam: str, *, return_none_if_not_found: bool = False
        ) -> str | float | int | None:
            """Get hyperparam value from variable and fixed hyperparams dicts.

            Args:
                hyperparam: Key to search for.
                return_none_if_not_found: Return None if the key is not found. Defaults to True, i.e. raise a ValueError.

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

            if not return_none_if_not_found and hyperparam_value is None:
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
                hyperparam_value = math.ceil(
                    prediction_length.seconds / self.chosen_hyperparams["time_bin_size"].seconds
                )
                self.chosen_hyperparams["output_chunk_length"] = hyperparam_value
                # Will update prediction_length_in_minutes to match later when output_chunk_length is final
            elif hyperparam == "pl_trainer_kwargs":
                self.chosen_hyperparams["es_min_delta"] = get_hyperparam_value("es_min_delta")
                self.chosen_hyperparams["es_patience"] = get_hyperparam_value("es_patience")
                self.chosen_hyperparams["enable_progress_bar"] = get_hyperparam_value(
                    "enable_progress_bar"
                )
                self.chosen_hyperparams["max_time"] = get_hyperparam_value(
                    "max_time", return_none_if_not_found=True
                )
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
                    self.chosen_hyperparams["lr_factor"],
                    self.chosen_hyperparams["lr_patience"],
                    self.verbose,
                )
            elif hyperparam == "verbose":
                hyperparam_value = self.verbose
            else:
                hyperparam_value = get_hyperparam_value(hyperparam)

            # Check for additional conditions
            if isinstance(self.hyperparams_conditions, list) and len(self.hyperparams_conditions):
                for condition_dict in self.hyperparams_conditions:
                    if condition_dict["hyperparam"] != hyperparam:
                        continue
                    # This hyperparam has conditions set on other hyperparams, we will ensure they is satisfied
                    rhs_value = self.chosen_hyperparams.get(
                        condition_dict["rhs"], get_hyperparam_value(condition_dict["rhs"])
                    )
                    op_func = condition_dict["condition"]
                    if op_func == operator.lt:  # pylint: disable=comparison-with-callable
                        new_value = rhs_value - 1
                        if new_value < 0 <= rhs_value:
                            logger_ts_wrapper.warning(
                                "Keeping hyperparam %s from becoming negative, model might complain!",
                                hyperparam,
                            )
                            new_value = 0
                    elif op_func == operator.ge:  # pylint: disable=comparison-with-callable
                        new_value = rhs_value
                    else:
                        raise ValueError(f"Uknown {op_func = }! Need to extend code for this use.")
                    if not op_func(hyperparam_value, rhs_value):
                        logger_ts_wrapper.info(
                            "For hyperparam %s, setting value = %s to satisfy condition:\n%s",
                            hyperparam,
                            new_value,
                            pprint.pformat(condition_dict),
                        )
                        hyperparam_value = new_value

            # Finally set the hyperparam value
            self.chosen_hyperparams[hyperparam] = hyperparam_value

        # Update prediction_length_in_minutes, now that any conditions have been applied
        if "output_chunk_length" in self.chosen_hyperparams:
            self.chosen_hyperparams["prediction_length_in_minutes"] = (
                self.chosen_hyperparams["output_chunk_length"]
                * self.chosen_hyperparams["time_bin_size_in_minutes"]
            )

        # Make sure int (bool) hyperparameters are int (bool), as Bayesian optimization will always give floats
        # and check chosen hyperparams are in the allowed ranges / sets
        for _k, _v in self.chosen_hyperparams.items():
            if _k in boolean_hyperparams:
                self.chosen_hyperparams[_k] = bool(_v)
            elif _k in integer_hyperparams:
                self.chosen_hyperparams[_k] = int(_v)
            if _k in list(self.fixed_hyperparams.keys()) + ["y_bin_edges"]:
                pass
            elif _k in self.allowed_variable_hyperparams:
                allowed_min = self.allowed_variable_hyperparams[_k].get("min")
                allowed_max = self.allowed_variable_hyperparams[_k].get("max")
                allowed_values = self.allowed_variable_hyperparams[_k].get("allowed")
                if allowed_min is not None and allowed_max is not None:
                    if _v < allowed_min or allowed_max < _v:
                        raise ValueError(
                            f"Hyperparameter {_k} with value {_v} is not allowed, expected to be between {allowed_min} and {allowed_max}!"
                        )
                elif allowed_values is not None and isinstance(allowed_values, list):
                    if not set(_v).issubset(set(allowed_values)):
                        raise ValueError(
                            f"Hyperparameter {_k} with value {_v} is not allowed, expected to be a subset of {allowed_values}!"
                        )
                else:
                    raise ValueError(f"Hyperparameter {_k} with value {_v} can not be checked!")
            if _k == "time_bin_size_in_minutes" and 60 % _v != 0:
                raise ValueError(
                    f"Hyperparameter {_k} with value {_v} is not allowed, {60 % _v = } should be 0!"
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
        if TYPE_CHECKING:
            assert isinstance(self.chosen_hyperparams, dict)  # noqa: SCS108 # nosec assert_used
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
            assert isinstance(self.chosen_hyperparams, dict)  # noqa: SCS108 # nosec assert_used

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
            interpolate_method = "linear"
            interpolate_limit_direction = "both"
            if self.chosen_hyperparams.get("rebin_y", 0):
                interpolate_method = "pad"
                interpolate_limit_direction = "forward"
            logger_ts_wrapper.debug(
                "Missing %.1g%% of values, filling via %s interpolation.",
                100.0 * frac_missing,
                interpolate_method,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="DataFrame.interpolate with method=pad is deprecated",
                    category=FutureWarning,
                )
                # ignore this for now and just use pad, would require a rewrite of the following code in darts
                # https://github.com/unit8co/darts/blob/4362df272c4a3e51ab33cf6596fb2d159be82b73/darts/utils/missing_values.py#L176
                # See the following for context
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ffill.html

                dart_series_y_trainable = fill_missing_values(
                    dart_series_y_trainable,
                    method=interpolate_method,
                    limit_direction=interpolate_limit_direction,
                )
                if covariates_type is not None:
                    dart_series_covariates_trainable = fill_missing_values(
                        dart_series_covariates_trainable,
                        method=interpolate_method,
                        limit_direction=interpolate_limit_direction,
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
            if self.is_nn:
                model_covariates_kwargs[
                    f"val_{covariates_type}_covariates"
                ] = dart_series_covariates_val
            prediction_covariates_kwargs[
                f"{covariates_type}_covariates"
            ] = dart_series_covariates_train.append(dart_series_covariates_val)

        # train
        train_kwargs = {}
        if self.is_nn:
            train_kwargs["val_series"] = dart_series_y_val
            train_kwargs["verbose"] = self.verbose

        _ = self.model.fit(
            dart_series_y_train,
            **train_kwargs,
            **model_covariates_kwargs,
        )

        # measure loss on validation
        y_pred_val = self.model.predict(
            dart_series_y_val.n_timesteps,
            num_samples=1,
            verbose=self.verbose,
            **prediction_covariates_kwargs,
            # Do not show warnings like
            # https://github.com/unit8co/darts/blob/20ee5ece4e02ed7c1e84db07679c83ceeb1f8a13/darts/models/forecasting/forecasting_model.py#L2334-L2337
            show_warnings=False,
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