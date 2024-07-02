# pylint: disable=duplicate-code,invalid-name
"""Wrapper for Prophet."""
# pylint: enable=invalid-name

from types import MappingProxyType
from typing import Any, Final

from darts.models import Prophet

from TSModelWrappers.TSModelWrapper import (
    DATA_FIXED_HYPERPARAMS,
    DATA_REQUIRED_HYPERPARAMS,
    DATA_VARIABLE_HYPERPARAMS,
    TSModelWrapper,
)

__all__ = ["ProphetWrapper"]


class ProphetWrapper(TSModelWrapper):
    """Prophet wrapper.

    https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet_model.html
    """

    # config wrapper for Prophet
    PROPHET_FIXED_HYPERPARAMS: Final = {
        "growth": "flat",
        "country_holidays": "US",
        "suppress_stdout_stderror": True,
    }

    _model_class = Prophet
    _model_type = "prophet"
    _required_hyperparams_data = DATA_REQUIRED_HYPERPARAMS
    _required_hyperparams_model = tuple(PROPHET_FIXED_HYPERPARAMS.keys())

    _allowed_variable_hyperparams = MappingProxyType({**DATA_VARIABLE_HYPERPARAMS})
    _fixed_hyperparams = MappingProxyType({**DATA_FIXED_HYPERPARAMS, **PROPHET_FIXED_HYPERPARAMS})

    def __init__(self: "ProphetWrapper", **kwargs: Any) -> None:  # noqa: ANN401
        _allowed_variable_hyperparams_dict = dict(self._allowed_variable_hyperparams)
        # Internally Prophet makes use of components equivalent to
        # day_of_week_frac, time_of_day_frac, and is_holiday,
        # so remove them as potential covariates here.
        allowed_prophet_covariates = [
            _
            for _ in _allowed_variable_hyperparams_dict["covariates"]["allowed"]  # type: ignore[index]
            if _ not in ["day_of_week_frac", "time_of_day_frac", "is_holiday"]
        ]

        for k, v in _allowed_variable_hyperparams_dict["covariates"].items():  # type: ignore[attr-defined]
            _allowed_variable_hyperparams_dict["covariates"][k] = [_ for _ in v if _ in allowed_prophet_covariates]  # type: ignore[index]

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
            self.required_hyperparams_model = list(self._required_hyperparams_model)
            self.allowed_variable_hyperparams = _allowed_variable_hyperparams_dict
            self.variable_hyperparams = kwargs.get("variable_hyperparams", {})
            self.fixed_hyperparams = dict(self._fixed_hyperparams)
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
                required_hyperparams_model=list(self._required_hyperparams_model),
                allowed_variable_hyperparams=_allowed_variable_hyperparams_dict,
                variable_hyperparams=kwargs.get("variable_hyperparams"),
                fixed_hyperparams=dict(self._fixed_hyperparams),
            )
