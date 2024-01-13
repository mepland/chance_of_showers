# pylint: disable=invalid-name
"""Wrapper for Prophet."""
# pylint: enable=invalid-name

from typing import Any, Final

from darts.models import Prophet

from utils.TSModelWrapper import (
    DATA_FIXED_HYPERPARAMS,
    DATA_REQUIRED_HYPERPARAMS,
    DATA_VARIABLE_HYPERPARAMS,
    TSModelWrapper,
)


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
    _is_nn = False
    _required_hyperparams_data = DATA_REQUIRED_HYPERPARAMS
    _required_hyperparams_model = list(PROPHET_FIXED_HYPERPARAMS.keys())

    _allowed_variable_hyperparams = {**DATA_VARIABLE_HYPERPARAMS}
    # Prophet makes "day_of_week_frac", "time_of_day_frac", "is_holiday" equivalent components, so remove as covariates
    _covariates = [
        _
        for _ in _allowed_variable_hyperparams["covariates"]["allowed"]  # type: ignore[index]
        if _ not in ["day_of_week_frac", "time_of_day_frac", "is_holiday"]
    ]

    _allowed_variable_hyperparams["covariates"]["allowed"] = _covariates  # type: ignore[index]
    _allowed_variable_hyperparams["covariates"]["default"] = _covariates  # type: ignore[index]

    _fixed_hyperparams = {**DATA_FIXED_HYPERPARAMS, **PROPHET_FIXED_HYPERPARAMS}

    def __init__(self: "ProphetWrapper", **kwargs: Any) -> None:  # noqa: ANN401
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
            and type(  # noqa: E721 # pylint: disable=unidiomatic-typecheck
                kwargs["TSModelWrapper"].__class__
            )
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
