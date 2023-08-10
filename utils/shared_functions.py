"""Shared functions."""

################################################################################
# python imports
import datetime
import os
import socket
import sys
from typing import TYPE_CHECKING

import pandas as pd


################################################################################
def get_lock(process_name: str) -> None:
    """Lock script via abstract socket, only works in Linux!

    Adapted from https://stackoverflow.com/a/7758075

    Args:
        process_name: The process_name to use for the locking socket.
    """
    # Without holding a reference to our socket somewhere it gets garbage collected when the function exits
    # pylint: disable=protected-access
    get_lock._lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)  # type: ignore[attr-defined]
    # See https://github.com/python/mypy/issues/2087 for ongoing mypy discussion on this attr-defined false positive
    try:
        # The null byte (\0) means the socket is created in the abstract namespace instead of being created on the file system itself.
        get_lock._lock_socket.bind(  # type: ignore[attr-defined]
            "\0" + process_name  # noqa: ESC101
        )
    # pylint: enable=protected-access
    except OSError:
        print(f"Lock for {process_name} exists, exiting!")
        sys.exit()


################################################################################
def get_SoC_temp() -> float:  # pylint: disable=invalid-name
    """Get SoC's temperature.

    Returns:
        SoC temperature as a float.
    """
    res = os.popen("vcgencmd measure_temp").readline()  # noqa: SCS110 # nosec: B605, B607

    return float(res.replace("temp=", "").replace("'C\n", ""))


################################################################################
def normalize_pressure_value(
    pressure_value: int,
    observed_pressure_min: float,
    observed_pressure_max: float,
    *,
    clip: bool = False,
) -> float:
    """Normalize raw ADC pressure_value.

    Args:
        pressure_value: The raw ADC pressure_value to be normalized.
        observed_pressure_min: The minimum observed ADC pressure value.
        observed_pressure_max: The maximum observed ADC pressure value.
        clip: Clip output between 0.0 and 1.0.

    Returns:
        (pressure_value-observed_pressure_min)/(observed_pressure_max-observed_pressure_min),
        i.e. observed_pressure_min (observed_pressure_max) maps to 0 (1).
    """
    normalized_pressure_value = (pressure_value - observed_pressure_min) / (
        observed_pressure_max - observed_pressure_min
    )

    # could use np.clip(normalized_pressure_value, a_min=0., a_max=1.), but let's avoid the dependency
    if clip:
        if normalized_pressure_value < 0.0:  # noqa: R505 pylint: disable=no-else-return
            return 0.0
        elif 1.0 < normalized_pressure_value:
            return 1.0
    return normalized_pressure_value


################################################################################
def rebin_chance_of_showers_time_series(
    dfp_in: pd.DataFrame,
    time_col: str,
    y_col: str,
    *,
    time_bin_size: datetime.timedelta | None = None,
    other_cols_to_agg_dict: dict | None = None,
    y_bin_edges: list[float] | None = None,
) -> pd.DataFrame:
    """Rebin the chance of showers time_series in time and y prior to modeling.

    Args:
        dfp_in: The input dataframe to rebin.
        time_col: The time column.
        y_col: The y column.
        time_bin_size: The size of time bins, must be less than 1 hour with the current implementation. This is not an issue in the chance of showers context.
        other_cols_to_agg_dict: Other columns to aggregate during time rebinning, and their aggregation function(s).
        y_bin_edges: The left bin edges for y.

    Returns:
        The rebinned dataframe.

    Raises:
        ValueError: Bad configuration.
    """
    rebin_time = time_bin_size is not None and time_col is not None
    rebin_y = y_bin_edges is not None and y_col is not None

    if rebin_time:
        if TYPE_CHECKING:
            assert isinstance(time_bin_size, datetime.timedelta)  # noqa: SCS108 # nosec assert_used

        time_bin_size_minutes = time_bin_size.seconds // 60
        if not 0 < time_bin_size_minutes < 60:
            raise ValueError(f"Invalid {time_bin_size = }, {time_bin_size_minutes = }")

    cols = [time_col, y_col]
    if rebin_time:
        if other_cols_to_agg_dict is None:
            other_cols_to_agg_dict = {}
        cols += other_cols_to_agg_dict.keys()

        cols_to_agg_dict = {y_col: "mean", **other_cols_to_agg_dict}

    dfp = dfp_in[cols].copy()

    if rebin_time:
        dfp[time_col] = dfp.apply(
            lambda row: row[time_col].replace(
                minute=time_bin_size_minutes * (row[time_col].minute // time_bin_size_minutes),
                second=0,
            ),
            axis=1,
        )
        dfp = dfp.groupby(time_col).agg(cols_to_agg_dict).reset_index()

    if rebin_y:
        if TYPE_CHECKING:
            assert isinstance(y_bin_edges, list)  # noqa: SCS108 # nosec assert_used

        dfp[y_col] = pd.cut(dfp[y_col], bins=y_bin_edges, right=True, labels=y_bin_edges[1:])

    return dfp
