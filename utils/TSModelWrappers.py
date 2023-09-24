import datetime
import zoneinfo
from typing import Final

import pandas as pd
import torch
import torchmetrics
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.missing_values import fill_missing_values, missing_values_ratio
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.shared_functions import (
    create_datetime_component_cols,
    rebin_chance_of_showers_time_series,
)

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
def gen_pl_trainer_kwargs(es_min_delta: float, es_patience: int) -> dict:
    """Generate pl_trainer_kwargs, i.e. PyTorch lightning trainer keyword arguments.

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
def gen_lr_scheduler_kwargs(lr_factor: float, lr_patience: int) -> dict:
    """Generate lr_scheduler_kwargs, i.e. PyTorch learning rate scheduler keyword arguments.

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
        "min_lr": 0,
        "eps": 1e-08,
        "verbose": False,
    }


################################################################################
# Setup global hyperparameter sets

# data hyperparameters
DATA_HYPERPARAMETERS: Final = {
    "time_bin_size_minutes": {
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
    "covariates": {
        "allowed": {
            "past": ["had_flow"],
            "future": ["day_of_week_frac", "time_of_day_frac", "is_holiday"],
        },
        "default": {
            "past": ["had_flow"],
            "future": ["day_of_week_frac", "time_of_day_frac", "is_holiday"],
        },
    },
}

NN_ALLOWED_VARIABLE_HYPERPARAMS: Final = {
    "input_chunk_length_in_minutes": {
        "min": 1,
        "max": 60,
        "default": 20,
    },
    "output_chunk_length_in_minutes": {
        "min": 5,
        "max": 30,
        "default": 30,
    },
    "dropout": {
        "min": 0.0,
        "max": 0.15,
        "default": 0.05,
    },
}
# pl_trainer_kwargs
# lr_scheduler_kwargs

NN_FIXED_HYPERPARAMS: Final = {
    "loss_fn": LOSS_FN,
    "torch_metrics": METRIC_COLLECTION,
    "log_tensorboard": True,
    "lr_scheduler_cls": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


################################################################################
# parent class
class TSModelWrapper:
    def __init__(
        self,
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
        model_class: PastCovariatesTorchModel | None = None,
        is_nn: bool | None = None,
        model_name: str | None = None,
        allowed_variable_hyperparams: dict | None = None,
        variable_hyperparams: dict | None = None,
        fixed_hyperparams: dict | None = None,
    ):
        self.model_class = model_class
        self.is_nn = is_nn
        self.model_name = model_name
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

    def __str__(self) -> str:
        return f"""
{self.model_class = }
{self.is_nn = }
{self.model_name = }
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
"""

    def __name_model(self) -> None:
        """Name this model."""
        if self.model_name is not None and self.model_name != "":
            model_name_base = self.model_name
        elif isinstance(self.model_class, PastCovariatesTorchModel):
            model_name_base = self.model_class.__name__
        else:
            raise ValueError("Unknown model name, should not happen!")

        self.model_name = f"{model_name_base}_{datetime.datetime.now(self.local_timezone).strftime(self.fname_datetime_fmt)}"

    def assemble_hyperparams(self) -> None:
        """Assemble the hyperparameters for this model"""
        if not (
            isinstance(self.allowed_variable_hyperparams, dict)
            and isinstance(self.variable_hyperparams, dict)
            and isinstance(self.fixed_hyperparams, dict)
        ):
            raise ValueError("Need to give model hyperparams first, should not happen!")

        # self.allowed_variable_hyperparams
        # self.variable_hyperparams
        # self.fixed_hyperparams

    def train_model(self) -> float:
        """Train the model and return loss.

        Returns:
            Loss.
        """

        self.__name_model()

        time_bin_size = datetime.timedelta(minutes=self.chosen_hyperparams["time_bin_size_minutes"])
        y_bin_edges = None
        if int(self.chosen_hyperparams["rebin_y"]):
            y_bin_edges = self.chosen_hyperparams["y_bin_edges"]

        dfp_trainable = rebin_chance_of_showers_time_series(
            self.dfp_trainable_evergreen,
            "ds",
            "y",
            time_bin_size=time_bin_size,
            other_cols_to_agg_dict={"had_flow": "max"},
            y_bin_edges=y_bin_edges,
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
            # f"Missing {frac_missing:.2%} of values, filling via interpolation"
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

        if self.is_nn:
            input_time_size = datetime.timedelta(
                minutes=self.chosen_hyperparams["input_time_size_minutes"]
            )
            self.chosen_hyperparams["input_chunk_length"] = (
                input_time_size.seconds // time_bin_size.seconds
            )

            output_time_size = datetime.timedelta(
                minutes=self.chosen_hyperparams["output_time_size_minutes"]
            )
            self.chosen_hyperparams["output_chunk_length"] = (
                output_time_size.seconds // time_bin_size.seconds
            )

        return 0.0


################################################################################
# model classes


class NBEATSModelWrapper(TSModelWrapper):
    __model_class = NBEATSModel
    __allowed_variable_hyperparams = {**DATA_HYPERPARAMETERS, **NN_ALLOWED_VARIABLE_HYPERPARAMS}
    __fixed_hyperparams = {**NN_FIXED_HYPERPARAMS}
    #    input_chunk_length=input_chunk_length,
    #    output_chunk_length=output_chunk_length,
    #    dropout=0.05,
    #    n_epochs=100,
    #    work_dir=MODELS_PATH,
    #    model_name=model_name,
    #    random_state=RANDOM_SEED,
    #    pl_trainer_kwargs=pl_trainer_kwargs,
    #    loss_fn=torchmetrics.MeanSquaredError(),
    #    torch_metrics=metric_collection,
    #    log_tensorboard=True,
    #    lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
    #    lr_scheduler_kwargs=lr_scheduler_kwargs,

    # def __init__(self, model_name: str, variable_hyperparams: dict):

    def __init__(self, **kargs):
        if "TSModelWrapper" in kargs and type(kargs["TSModelWrapper"]) is TSModelWrapper:
            self.__dict__ = kargs["TSModelWrapper"].__dict__.copy()
            self.model_class=self.__model_class,
            self.model_name = kargs.get("model_name")
            self.allowed_variable_hyperparams=self.__allowed_variable_hyperparams
            self.variable_hyperparams = kargs["variable_hyperparams"]
            self.fixed_hyperparams=self.__fixed_hyperparams
        else:
            super().__init__(
                model_class=self.__model_class,
                is_nn=True,
                model_name=kargs.get("model_name"),
                allowed_variable_hyperparams=self.__allowed_variable_hyperparams,
                variable_hyperparams=kargs["variable_hyperparams"],
                fixed_hyperparams=self.__fixed_hyperparams,
                dfp_trainable_evergreen=kargs["dfp_trainable_evergreen"],
                dt_val_start_datetime_local=kargs["dt_val_start_datetime_local"],
                work_dir=kargs["work_dir"],
                random_state=kargs["random_state"],
                date_fmt=kargs["date_fmt"],
                time_fmt=kargs["time_fmt"],
                fname_datetime_fmt=kargs["fname_datetime_fmt"],
                local_timezone=kargs["local_timezone"],
            )


# Goal:
# multiple model classes, inherating from TSModelWrapper
#
# each has defined fixed hyperparams as "private" variable
# each has defined allowed variable hyperparameters as "private" variable
#
# train method takes variables hyperparams and data
# checks variable hyperparams against allowed list
# saves model and logs to disk
# returns objective to hyper opt caller
