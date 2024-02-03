# pylint: disable=invalid-name,duplicate-code
"""Wrapper for AutoARIMA."""
# pylint: enable=invalid-name

from typing import Any

from darts.models import AutoARIMA

from utils.TSModelWrapper import (
    DATA_FIXED_HYPERPARAMS,
    DATA_REQUIRED_HYPERPARAMS,
    DATA_VARIABLE_HYPERPARAMS,
    TSModelWrapper,
)


class AutoARIMAWrapper(TSModelWrapper):
    """AutoARIMA wrapper.

    https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html
    """

    # config wrapper for AutoARIMA
    _model_class = AutoARIMA
    _is_nn = False
    _required_hyperparams_data = DATA_REQUIRED_HYPERPARAMS
    _required_hyperparams_model = [
        "m_AutoARIMA",
        "stationary",
        "max_p",
        "max_q",
        "max_P",
        "max_D",
        "max_Q",
        "maxiter",
        "random_state",
    ]

    # Set m_AutoARIMA as either a variable or fixed hyperparameter
    _variable_hyperparams_AutoARIMA = {
        "m_AutoARIMA": {
            "min": 1,  # 24 hours, set in _assemble_hyperparams() - Runs extremely slow...
            "max": 1,  # Default
            "default": 1,
            "type": int,
        },
    }

    _allowed_variable_hyperparams = {
        **DATA_VARIABLE_HYPERPARAMS,
        # **_variable_hyperparams_AutoARIMA,
    }

    _fixed_hyperparams_AutoARIMA = {
        # "m_AutoARIMA": 0,  # 24 hours, set in _assemble_hyperparams() - Runs extremely slow...
        "m_AutoARIMA": 1,  # Default
        "stationary": True,
        # Increase max values
        "max_p": 15,
        "max_q": 15,
        "max_P": 5,
        "max_D": 3,
        "max_Q": 5,
        "maxiter": 100,
    }

    # leave the following hyperparameters at their default values:
    # max_d ~ 2
    # start_p ~ 2
    # d ~ None
    # start_q ~ 2
    # start_P ~ 1
    # D ~ None
    # start_Q ~ 1
    # max_order ~ 5
    # seasonal ~ True
    # information_criterion ~ 'aic'
    # alpha ~ 0.05
    # test ~ 'kpss'
    # seasonal_test ~ 'ocsb'
    # stepwise ~ True
    # n_jobs ~ 1
    # start_params ~ None
    # trend ~ None
    # method ~ 'lbfgs'
    # offset_test_args ~ None
    # seasonal_test_args ~ None
    # suppress_warnings ~ True
    # error_action ~ 'trace'
    # trace ~ False
    # random ~ False
    # n_fits ~ 10
    # out_of_sample_size ~ 0
    # scoring ~ 'mse'
    # scoring_args ~ None
    # with_intercept ~ 'auto'

    _fixed_hyperparams = {
        **DATA_FIXED_HYPERPARAMS,
        **_fixed_hyperparams_AutoARIMA,
    }

    def __init__(self: "AutoARIMAWrapper", **kwargs: Any) -> None:  # noqa: ANN401
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
