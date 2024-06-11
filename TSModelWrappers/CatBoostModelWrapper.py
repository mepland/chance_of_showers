# pylint: disable=invalid-name,duplicate-code
"""Wrapper for CatBoostModel."""
# pylint: enable=invalid-name

from types import MappingProxyType
from typing import Any

from darts.models import CatBoostModel

from TSModelWrappers.TSModelWrapper import (
    DATA_FIXED_HYPERPARAMS,
    DATA_REQUIRED_HYPERPARAMS,
    DATA_VARIABLE_HYPERPARAMS,
    TREE_ALLOWED_VARIABLE_HYPERPARAMS,
    TREE_REQUIRED_HYPERPARAMS,
    TSModelWrapper,
)

__all__ = ["CatBoostModelWrapper"]


class CatBoostModelWrapper(TSModelWrapper):
    """CatBoostModel wrapper.

    https://unit8co.github.io/darts/generated_api/darts.models.forecasting.catboost_model.html
    """

    # config wrapper for CatBoostModel
    _model_class = CatBoostModel
    _model_type = "regression"
    _required_hyperparams_data = DATA_REQUIRED_HYPERPARAMS
    _required_hyperparams_model = TREE_REQUIRED_HYPERPARAMS

    _allowed_variable_hyperparams = MappingProxyType(
        {
            **DATA_VARIABLE_HYPERPARAMS,
            **TREE_ALLOWED_VARIABLE_HYPERPARAMS,
        }
    )
    _fixed_hyperparams = DATA_FIXED_HYPERPARAMS

    def __init__(self: "CatBoostModelWrapper", **kwargs: Any) -> None:  # noqa: ANN401
        # boilerplate - the same for all models below here
        # NOTE using `isinstance(kwargs["TSModelWrapper"], TSModelWrapper)`,
        # or even `issubclass(type(kwargs["TSModelWrapper"]), TSModelWrapper)` would be preferable
        # but they do not work if the kwargs["TSModelWrapper"] parent instance was updated between child __init__ calls
        if (
            "TSModelWrapper" in kwargs
            and type(kwargs["TSModelWrapper"].__class__)  # noqa: E721
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
            self.allowed_variable_hyperparams = dict(self._allowed_variable_hyperparams)
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
                model_type=self._model_type,
                verbose=kwargs.get("verbose", 1),
                work_dir=kwargs["work_dir"],
                model_name_tag=kwargs.get("model_name_tag"),
                required_hyperparams_data=self._required_hyperparams_data,
                required_hyperparams_model=self._required_hyperparams_model,
                allowed_variable_hyperparams=dict(self._allowed_variable_hyperparams),
                variable_hyperparams=kwargs.get("variable_hyperparams"),
                fixed_hyperparams=self._fixed_hyperparams,
            )
