# pylint: disable=invalid-name,duplicate-code
"""Wrapper for Croston."""
# pylint: enable=invalid-name

from typing import Any

from darts.models import Croston
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
)

from TSModelWrappers.TSModelWrapper import (
    DATA_FIXED_HYPERPARAMS,
    DATA_REQUIRED_HYPERPARAMS,
    DATA_VARIABLE_HYPERPARAMS,
    TSModelWrapper,
)

__all__ = ["CrostonWrapper"]


class CrostonWrapper(TSModelWrapper):
    """Croston wrapper.

    https://unit8co.github.io/darts/generated_api/darts.models.forecasting.croston.html
    """

    # config wrapper for Croston
    _model_class = Croston
    _model_type = "statistical"
    _required_hyperparams_data = DATA_REQUIRED_HYPERPARAMS
    _required_hyperparams_model = ["version"]
    _allowed_variable_hyperparams = DATA_VARIABLE_HYPERPARAMS
    _fixed_hyperparams = DATA_FIXED_HYPERPARAMS

    _valid_versions = [
        "classic",
        "optimized",
        "sba",
        # Do not use tsb as alpha_d and alpha_p must be set
    ]

    def __init__(self: "CrostonWrapper", **kwargs: Any) -> None:  # noqa: ANN401
        # setup the version parameter correctly
        if "version" in kwargs:
            version = kwargs["version"]
            # check validity of version, and set model_name_tag appropriately
            if version in self._valid_versions:
                if "model_name_tag" in kwargs and len(kwargs["model_name_tag"]):
                    kwargs["model_name_tag"] = f'{version}_{kwargs["model_name_tag"]}'
                else:
                    kwargs["model_name_tag"] = version

            elif isinstance(version, type) and issubclass(
                version, FutureCovariatesLocalForecastingModel
            ):
                if "model_name_tag" not in kwargs:
                    raise ValueError(
                        "Require a descriptive model_name_tag in kwargs when using FutureCovariatesLocalForecastingModel for version parameter!"
                    )

            else:
                valid_versions_str = ", ".join([f"{_!r}" for _ in self._valid_versions])
                raise ValueError(
                    f"{version = } must be in {valid_versions_str} or be a subclass of FutureCovariatesLocalForecastingModel"
                )

            self._fixed_hyperparams["version"] = version
            # remove version from kwargs so it does not cause later complications
            del kwargs["version"]
        else:
            raise ValueError("'version' is required in kwargs for CrostonWrapper!")

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
            )
