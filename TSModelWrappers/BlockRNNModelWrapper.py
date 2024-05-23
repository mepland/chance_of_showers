# pylint: disable=invalid-name,duplicate-code
"""Wrapper for BlockRNN."""
# pylint: enable=invalid-name

import operator
from typing import Any

from darts.models import BlockRNNModel
from darts.models.forecasting.block_rnn_model import CustomBlockRNNModule

from TSModelWrappers.TSModelWrapper import (
    DATA_FIXED_HYPERPARAMS,
    DATA_REQUIRED_HYPERPARAMS,
    DATA_VARIABLE_HYPERPARAMS,
    NN_ALLOWED_VARIABLE_HYPERPARAMS,
    NN_FIXED_HYPERPARAMS,
    NN_REQUIRED_HYPERPARAMS,
    TSModelWrapper,
)

__all__ = ["BlockRNNModelWrapper"]


class BlockRNNModelWrapper(TSModelWrapper):
    """BlockRNNModel wrapper.

    https://unit8co.github.io/darts/generated_api/darts.models.forecasting.block_rnn_model.html
    """

    # config wrapper for BlockRNNModel
    _model_class = BlockRNNModel
    _model_type = "torch"
    _required_hyperparams_data = DATA_REQUIRED_HYPERPARAMS
    _required_hyperparams_model = NN_REQUIRED_HYPERPARAMS + [
        "model",
        "hidden_dim",
        "n_rnn_layers",
    ]
    _allowed_variable_hyperparams = {**DATA_VARIABLE_HYPERPARAMS, **NN_ALLOWED_VARIABLE_HYPERPARAMS}
    _fixed_hyperparams = {**DATA_FIXED_HYPERPARAMS, **NN_FIXED_HYPERPARAMS}

    _hyperparams_conditions = [
        # The length of both input (target and covariates) and output (target) time series used during training.
        # Generally speaking, training_length should have a higher value than input_chunk_length
        # because otherwise during training the BlockRNN is never run for as many iterations as it will during inference.
        {
            "hyperparam": "training_length",
            "condition": operator.ge,
            "rhs": "input_chunk_length",
        },
    ]

    _valid_models = ["RNN", "LSTM", "GRU"]

    def __init__(self: "BlockRNNModelWrapper", **kwargs: Any) -> None:  # noqa: ANN401
        # setup the model parameter correctly
        if "model" in kwargs:
            model = kwargs["model"]
            # check validity of model, and set model_name_tag appropriately
            if model in self._valid_models:
                if "model_name_tag" in kwargs and len(kwargs["model_name_tag"]):
                    kwargs["model_name_tag"] = f'{model}_{kwargs["model_name_tag"]}'
                else:
                    kwargs["model_name_tag"] = model

            elif isinstance(model, type) and issubclass(model, CustomBlockRNNModule):
                if "model_name_tag" not in kwargs:
                    raise ValueError(
                        "Require a descriptive model_name_tag in kwargs when using CustomBlockRNNModule for model parameter!"
                    )

            else:
                valid_models_str = ", ".join([f"{_!r}" for _ in self._valid_models])
                raise ValueError(
                    f"{model = } must be in {valid_models_str} or be a subclass of CustomBlockRNNModule"
                )

            self._fixed_hyperparams["model"] = model
            # remove model from kwargs so it does not cause later complications
            del kwargs["model"]
        else:
            raise ValueError("'model' is required in kwargs for BlockRNNModelWrapper!")

        # boilerplate - the same for all models below here

        # NOTE using `isinstance(kwargs["TSModelWrapper"], TSModelWrapper)`,
        # or even `issubclass(type(kwargs["TSModelWrapper"]), TSModelWrapper)` would be preferable
        # but they do not work if the kwargs["TSModelWrapper"] parent instance was updated between child __init__ calls
        if (
            "TSModelWrapper" in kwargs
            and type(  # noqa: E721 # pylint: disable=unidiomatic-typecheck
                kwargs["TSModelWrapper"].__class__
            )
            == type(TSModelWrapper)  # <class 'type'>
            and str(kwargs["TSModelWrapper"].__class__)
            == str(TSModelWrapper)  # <class 'TSModelWrappers.TSModelWrappers.TSModelWrapper'>
        ):
            self.__dict__ = kwargs["TSModelWrapper"].__dict__.copy()
            self.model_class = self._model_class
            self.model_type = self._model_type
            self.verbose = kwargs.get("verbose", 1)
            self.work_dir = kwargs.get("work_dir")
            self.model_name_tag = kwargs.get("model_name_tag")
            self.required_hyperparams_data = self._required_hyperparams_data
            self.required_hyperparams_model = self._required_hyperparams_model
            self.allowed_variable_hyperparams = self._allowed_variable_hyperparams
            self.variable_hyperparams = kwargs.get("variable_hyperparams", {})
            self.fixed_hyperparams = self._fixed_hyperparams
            self.hyperparams_conditions = self._hyperparams_conditions
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
                model_type=self._model_type,
                verbose=kwargs.get("verbose", 1),
                work_dir=kwargs["work_dir"],
                model_name_tag=kwargs.get("model_name_tag"),
                required_hyperparams_data=self._required_hyperparams_data,
                required_hyperparams_model=self._required_hyperparams_model,
                allowed_variable_hyperparams=self._allowed_variable_hyperparams,
                variable_hyperparams=kwargs.get("variable_hyperparams"),
                fixed_hyperparams=self._fixed_hyperparams,
                hyperparams_conditions=self._hyperparams_conditions,
            )
