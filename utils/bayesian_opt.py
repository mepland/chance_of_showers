"""Setup Bayesian optimization."""

import datetime
import gc
import json
import pathlib
import platform
import pprint
import signal
import traceback
from types import FrameType  # noqa: TC003
from typing import TYPE_CHECKING, Final, TypeAlias

import bayes_opt
import humanize
import numpy as np
import pandas as pd
import psutil
import torch
from bayes_opt.event import DEFAULT_EVENTS, Events
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.util import load_logs

# isort: off
from utils.TSModelWrapper import TSModelWrapper  # noqa: TC001

# Prophet
from utils.ProphetWrapper import ProphetWrapper  # noqa: TC001

# PyTorch NN Models
from utils.NBEATSModelWrapper import NBEATSModelWrapper  # noqa: TC001
from utils.NHiTSModelWrapper import NHiTSModelWrapper  # noqa: TC001
from utils.TCNModelWrapper import TCNModelWrapper  # noqa: TC001
from utils.TransformerModelWrapper import TransformerModelWrapper  # noqa: TC001
from utils.TFTModelWrapper import TFTModelWrapper  # noqa: TC001
from utils.DLinearModelWrapper import DLinearModelWrapper  # noqa: TC001
from utils.NLinearModelWrapper import NLinearModelWrapper  # noqa: TC001
from utils.TiDEModelWrapper import TiDEModelWrapper  # noqa: TC001
from utils.RNNModelWrapper import RNNModelWrapper  # noqa: TC001
from utils.BlockRNNModelWrapper import BlockRNNModelWrapper  # noqa: TC001

# Statistical Models
from utils.AutoARIMAWrapper import AutoARIMAWrapper  # noqa: TC001
from utils.BATSWrapper import BATSWrapper  # noqa: TC001
from utils.TBATSWrapper import TBATSWrapper  # noqa: TC001
from utils.FourThetaWrapper import FourThetaWrapper  # noqa: TC001
from utils.StatsForecastAutoThetaWrapper import StatsForecastAutoThetaWrapper  # noqa: TC001
from utils.FFTWrapper import FFTWrapper  # noqa: TC001
from utils.KalmanForecasterWrapper import KalmanForecasterWrapper  # noqa: TC001
from utils.CrostonWrapper import CrostonWrapper  # noqa: TC001

# Regression Models
from utils.LinearRegressionModelWrapper import LinearRegressionModelWrapper  # noqa: TC001
from utils.RandomForestWrapper import RandomForestWrapper  # noqa: TC001
from utils.LightGBMModelWrapper import LightGBMModelWrapper  # noqa: TC001
from utils.XGBModelWrapper import XGBModelWrapper  # noqa: TC001
from utils.CatBoostModelWrapper import CatBoostModelWrapper  # noqa: TC001

# Naive Models
from utils.NaiveMeanWrapper import NaiveMeanWrapper  # noqa: TC001
from utils.NaiveSeasonalWrapper import NaiveSeasonalWrapper  # noqa: TC001
from utils.NaiveDriftWrapper import NaiveDriftWrapper  # noqa: TC001
from utils.NaiveMovingAverageWrapper import NaiveMovingAverageWrapper  # noqa: TC001

# isort: on

WrapperTypes: TypeAlias = type[
    # Prophet
    ProphetWrapper
    # PyTorch NN Models
    | NBEATSModelWrapper
    | NHiTSModelWrapper
    | TCNModelWrapper
    | TransformerModelWrapper
    | TFTModelWrapper
    | DLinearModelWrapper
    | NLinearModelWrapper
    | TiDEModelWrapper
    | RNNModelWrapper
    | BlockRNNModelWrapper
    # Statistical Models
    | AutoARIMAWrapper
    | BATSWrapper
    | TBATSWrapper
    | FourThetaWrapper
    | StatsForecastAutoThetaWrapper
    | FFTWrapper
    | KalmanForecasterWrapper
    | CrostonWrapper
    # Regression Models
    | LinearRegressionModelWrapper
    | RandomForestWrapper
    | LightGBMModelWrapper
    | XGBModelWrapper
    | CatBoostModelWrapper
    # Naive Models
    | NaiveMeanWrapper
    | NaiveSeasonalWrapper
    | NaiveDriftWrapper
    | NaiveMovingAverageWrapper
]


def load_json_log_to_dfp(f_path: pathlib.Path) -> None | pd.DataFrame:
    """Load prior bayes_opt log from json file as a pandas dataframe.

    Args:
        f_path: Full path to json log file.

    Returns:
        Log as pandas dataframe.
    """
    # Adapted from:
    # https://github.com/bayesian-optimization/BayesianOptimization/blob/129caac02177b146ce315e177d4d88950b75253a/bayes_opt/util.py#L214-L241
    with open(str(f_path), encoding="utf-8") as f_json:
        rows = []
        while True:
            try:
                iteration = next(f_json)
            except StopIteration:
                break

            row = {}
            for _k0, _v0 in dict(sorted(json.loads(iteration).items())).items():
                if isinstance(_v0, dict):
                    for _k1, _v1 in dict(sorted(_v0.items())).items():
                        row[f"{_k0}_{_k1}"] = _v1
                else:
                    row[_k0] = _v0
            rows.append(row)
        f_json.close()

        if rows:
            dfp = pd.DataFrame(rows)
            dfp["i_point"] = dfp.index

            cols_fixed = ["i_point", "datetime_datetime"]
            return dfp[cols_fixed + [_ for _ in dfp.columns if _ not in cols_fixed]]
        return None


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


n_points = 0  # # pylint: disable=invalid-name


def run_bayesian_opt(  # noqa: C901 # pylint: disable=too-many-statements,too-many-locals
    parent_wrapper: TSModelWrapper,
    model_wrapper_class: WrapperTypes,
    *,
    model_wrapper_kwargs: dict | None = None,
    hyperparams_to_opt: list[str] | None = None,
    n_iter: int = 100,
    allow_duplicate_points: bool = False,
    utility_kind: str = "ucb",
    utility_kappa: float = 2.576,
    verbose: int = 2,
    model_verbose: int = -1,
    display_memory_usage: bool = False,
    enable_progress_bar: bool = False,
    max_time_per_model: datetime.timedelta | None = None,
    accelerator: str | None = "auto",
    fixed_hyperparams_to_alter: dict | None = None,
    enable_json_logging: bool = True,
    enable_reloading: bool = True,
    enable_model_saves: bool = False,
    bayesian_opt_work_dir_name: str = "bayesian_optimization",
) -> tuple[dict, bayes_opt.BayesianOptimization]:
    """Run Bayesian optimization for this model wrapper.

    Args:
        parent_wrapper: TSModelWrapper object containing all parent configs.
        model_wrapper_class: TSModelWrapper class to optimize.
        model_wrapper_kwargs: kwargs to passs to model_wrapper.
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
        model_verbose: Verbose level of model_wrapper, default is -1 to silence LightGBMModel.
        display_memory_usage: Print memory usage at each training iteration.
        enable_progress_bar: Enable torch progress bar during training.
        max_time_per_model: Set the maximum amount of training time for each iteration.
            Torch models will use max_time_per_model as the max time per epoch,
            while non-torch models will use it for the whole iteration if signal is avaliable e.g. Linux, Darwin.
        accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")
        fixed_hyperparams_to_alter: Dict of fixed hyperparameters to alter, but not optimize.
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

    # Set a finite, but horrible, loss for when the training fails to complete.
    # np.finfo(np.float64).min + 1 does not work, sklearn errors
    bad_target = -999

    if model_wrapper_kwargs is None:
        model_wrapper_kwargs = {}

    # Setup hyperparameters
    _model_wrapper = model_wrapper_class(TSModelWrapper=parent_wrapper, **model_wrapper_kwargs)
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
    bayesian_opt_work_dir: Final = pathlib.Path(
        _model_wrapper.work_dir_base, bayesian_opt_work_dir_name, generic_model_name
    ).expanduser()
    fname_json_log: Final = bayesian_opt_work_dir / f"bayesian_opt_{generic_model_name}.json"

    # Reload prior points, must be done before json_logger is recreated to avoid duplicating past runs
    n_points = 0
    if enable_reloading and fname_json_log.is_file():
        print(f"Resuming Bayesian optimization from:\n{fname_json_log}\n")
        optimizer.dispatch(Events.OPTIMIZATION_START)
        load_logs(optimizer, logs=str(fname_json_log))
        n_points = len(optimizer.space)
        print(f"Loaded {n_points} existing points.\n")

    # Continue to setup logging
    if enable_json_logging:
        bayesian_opt_work_dir.mkdir(parents=True, exist_ok=True)
        json_logger = JSONLogger(path=str(fname_json_log), reset=False)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, json_logger)

    if 0 < verbose:
        screen_logger = ScreenLogger(verbose=verbose)
        for event in DEFAULT_EVENTS:
            optimizer.subscribe(event, screen_logger)

    # Define function to complete an iteration
    def complete_iter(
        i_iter: int,
        model_wrapper: TSModelWrapper,
        target: float,
        point_to_probe: dict,
        *,
        probed_point: dict | None = None,
    ) -> None:
        """Complete this iteration, register point(s) and clean up.

        Args:
            i_iter: Index of this iteration.
            model_wrapper: Model wrapper object to rest.
            target: Target value to register.
            point_to_probe: Raw point to probe.
            probed_point: Point that was actually probed.
        """
        global n_points
        optimizer.register(params=point_to_probe, target=target)
        n_points += 1
        if probed_point and not (
            # point_to_probe is exactly the same as probed_point on this iter,
            # i.e. the optimizer chose a point that required no cleanup in _assemble_hyperparams().
            # Do not register the probed_point.
            np.array_equiv(
                optimizer.space.params_to_array(point_to_probe),
                optimizer.space.params_to_array(probed_point),
            )
        ):
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

    # Setup signal_handler to kill iteration if it runs too long
    def signal_handler(
        dummy_signal: int,  # noqa: U100
        dummy_frame: FrameType | None,  # noqa: U100 # pylint: disable=used-before-assignment
    ) -> None:
        """Stop iteration gracefuly.

        https://medium.com/@chamilad/timing-out-of-long-running-methods-in-python-818b3582eed6

        Args:
            dummy_signal: signal number.
            dummy_frame: Frame object.

        Raises:
            Exception: Out of Time!
        """
        raise Exception("Out of Time!")  # pylint: disable=broad-exception-raised

    max_time_per_model_flag = (
        max_time_per_model is not None
        and not parent_wrapper.is_nn
        and platform.system() in ["Linux", "Darwin"]
    )
    if max_time_per_model_flag:
        signal.signal(signal.SIGALRM, signal_handler)

    # Run Bayesian optimization iterations
    try:
        for i_iter in range(n_iter):
            if i_iter == 0:
                optimizer.dispatch(Events.OPTIMIZATION_START)
            print(f"\nStarting {i_iter = }, with {n_points = }")
            next_point_to_probe = optimizer.suggest(utility)

            # Create a fresh model_wrapper object to try to avoid GPU memory leaks
            # This may not be necessary, but as it is already coded, just be safe and leave it
            model_wrapper = model_wrapper_class(
                TSModelWrapper=parent_wrapper, **model_wrapper_kwargs
            )
            model_wrapper.alter_fixed_hyperparams(
                fixed_hyperparams_to_alter=fixed_hyperparams_to_alter
            )
            model_wrapper.set_work_dir(work_dir_absolute=bayesian_opt_work_dir)
            model_wrapper.set_enable_progress_bar(enable_progress_bar=enable_progress_bar)
            model_wrapper.set_max_time(max_time=max_time_per_model)
            model_wrapper.set_accelerator(accelerator=accelerator)
            model_wrapper.verbose = model_verbose

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
                    complete_iter(i_iter, model_wrapper, target, next_point_to_probe)
                    is_duplicate_point = True
                    break
            if is_duplicate_point:
                continue

            # set model_name_tag for this iteration
            model_wrapper.set_model_name_tag(model_name_tag=f"iteration_{n_points}")

            # Setup iteration kill timer
            if max_time_per_model_flag:
                if TYPE_CHECKING:
                    assert isinstance(  # noqa: SCS108 # nosec assert_used
                        max_time_per_model, datetime.timedelta
                    )
                signal.alarm(max_time_per_model.seconds)

            # train the model
            try:
                target = model_wrapper.train_model(**next_point_to_probe)
                # Put a lower bound on target at bad_target.
                # This is in case a NN is interrupted mid-epoch and returns a loss of -float("inf").
                target = max(target, bad_target)
            except KeyboardInterrupt:
                print("KeyboardInterrupt: Returning with current objects.")
                optimizer.dispatch(Events.OPTIMIZATION_END)

                return optimizer.max, optimizer
            except RuntimeError as error:
                if "out of memory" in str(error):
                    print(f"Ran out of memory, returning {bad_target:.3g} as loss")
                    complete_iter(
                        i_iter,
                        model_wrapper,
                        bad_target,
                        next_point_to_probe,
                        probed_point=next_point_to_probe_cleaned,
                    )
                    continue
                raise error
            except Exception as error:
                if "Out of Time!" in str(error):
                    print(f"Ran out of time, returning {bad_target:.3g} as loss")
                    complete_iter(
                        i_iter,
                        model_wrapper,
                        bad_target,
                        next_point_to_probe,
                        probed_point=next_point_to_probe_cleaned,
                    )
                    continue
                raise error
            finally:
                if max_time_per_model_flag:
                    signal.alarm(0)

            if enable_model_saves:
                fname_model = (
                    bayesian_opt_work_dir / f"iteration_{n_points}_{generic_model_name}.pt"
                )
                model_wrapper.get_model().save(fname_model)

            # Register the point
            complete_iter(
                i_iter,
                model_wrapper,
                target,
                next_point_to_probe,
                probed_point=next_point_to_probe_cleaned,
            )

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Returning with current objects.")
    except bayes_opt.util.NotUniqueError as error:
        print(
            str(error).replace(
                '. You can set "allow_duplicate_points=True" to avoid this error', ""
            )
            + ", stopping optimization here."
        )
    except Exception as error:
        print(
            f"Unexpected error in run_bayesian_opt():\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nnext_point_to_probe = {pprint.pformat(next_point_to_probe)}\n\nReturning with current objects."
        )

    optimizer.dispatch(Events.OPTIMIZATION_END)

    return optimizer.max, optimizer
