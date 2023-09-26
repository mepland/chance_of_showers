# pylint: disable=invalid-name
"""Wrapper classes for time series models."""

# pylint: enable=invalid-name

import datetime
import zoneinfo
from typing import TYPE_CHECKING, Any, Final

import pandas as pd
import torch
import torchmetrics
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils.missing_values import fill_missing_values, missing_values_ratio
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
# copy docs from https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/early_stopping.html#EarlyStopping
def get_pl_trainer_kwargs(es_min_delta: float, es_patience: int) -> dict:
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

    Returns:
        pl_trainer_kwargs.
    """
    return {
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
    "time_bin_size_minutes",
    "rebin_y",
    "y_bin_edges",
]

DATA_HYPERPARAMETERS: Final = {
    "time_bin_size_minutes": {
        "min": 1.0,
        "max": 20.0,
        "default": 10.0,
    },
    "rebin_y": {
        "min": 0,
        "max": 1,
        "default": 0,
    },
    "y_bin_edges": [-float("inf"), 0.6, 0.8, 0.9, 1.0],
    "covariates_past": {
        "allowed": ["had_flow"],
        "default": ["had_flow"],
    },
    "covariates_future": {
        "allowed": ["day_of_week_frac", "time_of_day_frac", "is_holiday"],
        "default": ["day_of_week_frac", "time_of_day_frac", "is_holiday"],
    },
}

# NN hyperparameters

NN_REQUIRED_HYPERPARAMS: Final = [
    "input_chunk_length",
    "output_chunk_length",
    "dropout",
    "n_epochs",
    "work_dir",
    "model_name",
    "random_state",
    "pl_trainer_kwargs",
    "loss_fn",
    "torch_metrics",
    "log_tensorboard",
    "lr_scheduler_cls",
    "lr_scheduler_kwargs",
]

NN_ALLOWED_VARIABLE_HYPERPARAMS: Final = {
    "input_chunk_length_in_minutes": {
        "min": 1.0,
        "max": 60.0,
        "default": 20.0,
    },
    "output_chunk_length_in_minutes": {
        "min": 5.0,
        "max": 30.0,
        "default": 30.0,
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
}

NN_FIXED_HYPERPARAMS: Final = {
    "n_epochs": 100,
    "loss_fn": LOSS_FN,
    "torch_metrics": METRIC_COLLECTION,
    "log_tensorboard": True,
    "lr_scheduler_cls": torch.optim.lr_scheduler.ReduceLROnPlateau,
}

INTEGER_HYPERPARAMS: Final = [
    "rebin_y",
    "es_patience",
    "lr_patience",
    "n_epochs",
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
        work_dir: str,
        random_state: int,
        date_fmt: str,
        time_fmt: str,
        fname_datetime_fmt: str,
        local_timezone: zoneinfo.ZoneInfo,
        # optional, will later load in child classes
        *,
        model_class: ForecastingModel | None = None,
        is_nn: bool | None = None,
        model_name: str | None = None,
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
            work_dir: Working directory to save model files.
            random_state: Random seed.
            date_fmt: String format of dates.
            time_fmt: String format of times.
            fname_datetime_fmt: String format of date times for file names.
            local_timezone: Local timezone.
            model_class: Dart model class.
            is_nn: Flag for if the model is a neural network (NN).
            model_name: Model name.
            required_hyperparams_data: List of required data hyperparameters for this model.
            required_hyperparams_model: List of required hyperparameters for this model's constructor.
            allowed_variable_hyperparams: Dictionary of allowed variable hyperparameters for this model.
            variable_hyperparams: Dictionary of variable hyperparameters for this model.
            fixed_hyperparams: Dictionary of fixed hyperparameters for this model.
        """
        self.model_class = model_class
        self.is_nn = is_nn
        self.model_name = model_name
        self.required_hyperparams_data = required_hyperparams_data
        self.required_hyperparams_model = required_hyperparams_model
        self.allowed_variable_hyperparams = allowed_variable_hyperparams
        self.variable_hyperparams = variable_hyperparams
        self.fixed_hyperparams = fixed_hyperparams

        self.dfp_trainable_evergreen = dfp_trainable_evergreen
        self.dt_val_start_datetime_local = dt_val_start_datetime_local.replace(tzinfo=None)
        self.work_dir = work_dir
        self.random_state = random_state
        self.date_fmt = date_fmt
        self.time_fmt = time_fmt
        self.fname_datetime_fmt = fname_datetime_fmt
        self.local_timezone = local_timezone

        self.chosen_hyperparams: dict = {}
        self.model = None

    def __str__(self: "TSModelWrapper") -> str:
        """Redefine the str method.

        Returns:
            Description of model as str.
        """
        return f"""
{self.model_class = }
{self.is_nn = }
{self.model_name = }
{self.required_hyperparams_data = }
{self.required_hyperparams_model = }
{self.allowed_variable_hyperparams = }
{self.variable_hyperparams = }
{self.fixed_hyperparams = }

{self.dfp_trainable_evergreen.index.size = }
{self.dt_val_start_datetime_local = }
{self.work_dir = }
{self.random_state = }
{self.date_fmt = }
{self.time_fmt = }
{self.fname_datetime_fmt = }
{self.local_timezone = }

{self.chosen_hyperparams = }
{self.model = }
"""

    def _name_model(self: "TSModelWrapper") -> None:
        """Name this model.

        Raises:
            ValueError: Bad configuration.
        """
        if self.model_name is not None and self.model_name != "":
            model_name_base = self.model_name
        elif issubclass(self.model_class, ForecastingModel):  # type: ignore[arg-type]
            if TYPE_CHECKING:
                assert isinstance(  # noqa: SCS108 # nosec assert_used
                    self.model_class, ForecastingModel
                )
            model_name_base = self.model_class.__name__
        else:
            raise ValueError("Unknown model name, should not happen!")

        # TODO rerunning training will keep appending datetime, rethink this...
        self.model_name = f"{model_name_base}_{datetime.datetime.now(self.local_timezone).strftime(self.fname_datetime_fmt)}"

    def _assemble_hyperparams(self: "TSModelWrapper") -> None:  # noqa: C901
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

            hyperparam_value = self.variable_hyperparams.get(
                hyperparam,
                self.allowed_variable_hyperparams.get(hyperparam, {}).get(
                    "default", self.fixed_hyperparams.get(hyperparam)
                ),
            )

            if hyperparam_value is None:
                raise ValueError(f"Could not find value for required hyperparameter {hyperparam}!")

            return hyperparam_value

        # prep time_bin_size parameter
        if "time_bin_size_minutes" in required_hyperparams_all:
            time_bin_size_minutes = get_hyperparam_value("time_bin_size_minutes")
            if TYPE_CHECKING:
                assert isinstance(time_bin_size_minutes, float)  # noqa: SCS108 # nosec assert_used

            self.chosen_hyperparams["time_bin_size_minutes"] = time_bin_size_minutes
            self.chosen_hyperparams["time_bin_size"] = datetime.timedelta(
                minutes=time_bin_size_minutes
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
            elif hyperparam == "y_bin_edges":  # noqa: R507
                continue
            elif hyperparam == "input_chunk_length":
                self.chosen_hyperparams["input_chunk_length_in_minutes"] = get_hyperparam_value(
                    "input_chunk_length_in_minutes"
                )
                input_chunk_length = datetime.timedelta(
                    minutes=self.chosen_hyperparams["input_chunk_length_in_minutes"]
                )
                hyperparam_value = (
                    input_chunk_length.seconds // self.chosen_hyperparams["time_bin_size"].seconds
                )
            elif hyperparam == "output_chunk_length":
                self.chosen_hyperparams["output_chunk_length_in_minutes"] = get_hyperparam_value(
                    "output_chunk_length_in_minutes"
                )
                output_chunk_length = datetime.timedelta(
                    minutes=self.chosen_hyperparams["output_chunk_length_in_minutes"]
                )
                hyperparam_value = (
                    output_chunk_length.seconds // self.chosen_hyperparams["time_bin_size"].seconds
                )
            elif hyperparam == "pl_trainer_kwargs":
                self.chosen_hyperparams["es_min_delta"] = get_hyperparam_value("es_min_delta")
                self.chosen_hyperparams["es_patience"] = get_hyperparam_value("es_patience")
                hyperparam_value = get_pl_trainer_kwargs(
                    self.chosen_hyperparams["es_min_delta"], self.chosen_hyperparams["es_patience"]
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

        # check chosen hyperparams are in the allowed ranges / sets and
        # make sure int hyperparameters are int, as Bayesian optimization will always give floats
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

    def train_model(self: "TSModelWrapper") -> float:
        """Train the model and return loss.

        Returns:
            Loss.
        """
        # setup hyperparams
        self._name_model()
        self._assemble_hyperparams()

        # data prep
        time_bin_size = self.chosen_hyperparams["time_bin_size"]
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
            dfp_trainable, "ds", "y", freq=time_bin_size
        )
        dart_series_covariates_trainable = TimeSeries.from_dataframe(
            dfp_trainable,
            "ds",
            self.chosen_hyperparams["covariates_past"]
            + self.chosen_hyperparams["covariates_future"],
            freq=time_bin_size,
        )

        frac_missing = missing_values_ratio(dart_series_y_trainable)
        if 0 < frac_missing:
            # Missing {frac_missing:.2%} of values, filling via interpolation
            dart_series_y_trainable = fill_missing_values(dart_series_y_trainable)
            dart_series_covariates_trainable = fill_missing_values(dart_series_covariates_trainable)

        dart_series_y_train, dart_series_y_val = dart_series_y_trainable.split_before(
            pd.Timestamp(self.dt_val_start_datetime_local)
        )

        (
            dart_series_covariates_train,
            dart_series_covariates_val,
        ) = dart_series_covariates_trainable.split_before(
            pd.Timestamp(self.dt_val_start_datetime_local)
        )

        # construct object
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

        # train
        if TYPE_CHECKING:
            assert isinstance(self.model, ForecastingModel)  # noqa: SCS108 # nosec assert_used
        _ = self.model.fit(
            dart_series_y_train,
            val_series=dart_series_y_val,
            # TODO make past vs future covariates configurable
            past_covariates=dart_series_covariates_train,
            val_past_covariates=dart_series_covariates_val,
        )

        return 0.0


################################################################################
# model classes
class NBEATSModelWrapper(TSModelWrapper):
    """NBEATSModel wrapper."""

    # config wrapper for NBEATSModel
    _model_class = NBEATSModel
    _is_nn = True
    _required_hyperparams_data = DATA_REQUIRED_HYPERPARAMS + [
        "covariates_past",
        "covariates_future",
    ]
    _required_hyperparams_model = NN_REQUIRED_HYPERPARAMS
    _allowed_variable_hyperparams = {**DATA_HYPERPARAMETERS, **NN_ALLOWED_VARIABLE_HYPERPARAMS}
    _fixed_hyperparams = {**NN_FIXED_HYPERPARAMS}

    def __init__(self: "NBEATSModelWrapper", **kwargs: Any) -> None:  # noqa: ANN401
        # boilerplate - the same for all models
        """Int method.

        Args:
            **kwargs: Keyword arguments.
                Can be a parent TSModelWrapper instance plus the undefined parameters,
                or all the necessary parameters.
        """
        if (
            # Need the .keys() or there is an intermittent bug
            # pylint: disable-next=consider-iterating-dictionary
            "TSModelWrapper" in kwargs.keys()  # noqa: SIM118
            and isinstance(kwargs["TSModelWrapper"], TSModelWrapper)
        ):
            self.__dict__ = kwargs["TSModelWrapper"].__dict__.copy()
            self.model_class = self._model_class
            self.is_nn = self._is_nn
            self.model_name = kwargs.get("model_name")
            self.required_hyperparams_data = self._required_hyperparams_data
            self.required_hyperparams_model = self._required_hyperparams_model
            self.allowed_variable_hyperparams = self._allowed_variable_hyperparams
            self.variable_hyperparams = kwargs["variable_hyperparams"]
            self.fixed_hyperparams = self._fixed_hyperparams
        else:
            super().__init__(
                model_class=self._model_class,
                is_nn=self._is_nn,
                model_name=kwargs.get("model_name"),
                required_hyperparams_data=self._required_hyperparams_data,
                required_hyperparams_model=self._required_hyperparams_model,
                allowed_variable_hyperparams=self._allowed_variable_hyperparams,
                variable_hyperparams=kwargs["variable_hyperparams"],
                fixed_hyperparams=self._fixed_hyperparams,
                dfp_trainable_evergreen=kwargs["dfp_trainable_evergreen"],
                dt_val_start_datetime_local=kwargs["dt_val_start_datetime_local"],
                work_dir=kwargs["work_dir"],
                random_state=kwargs["random_state"],
                date_fmt=kwargs["date_fmt"],
                time_fmt=kwargs["time_fmt"],
                fname_datetime_fmt=kwargs["fname_datetime_fmt"],
                local_timezone=kwargs["local_timezone"],
            )


# Goal:
# train method takes variables hyperparams and data, trains the model
# saves model and logs to disk
# returns objective float to hyper opt caller function
