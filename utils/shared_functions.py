"""Shared TEST othe transfered anouncement TEST functions."""

################################################################################
# python imports
import datetime
import hashlib
import hmac
import os
import pathlib
import pickle  # nosec B403
import platform
import socket
import sys
from typing import TYPE_CHECKING, Any

import holidays
import pandas as pd

__all__ = [
    "create_datetime_component_cols",
    "get_SoC_temp",
    "get_lock",
    "normalize_pressure_value",
    "read_secure_pickle",
    "rebin_chance_of_showers_time_series",
    "write_secure_pickle",
]


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
    res = os.popen("vcgencmd measure_temp").readline()  # noqa: DUO106, SCS110 # nosec: B605, B607

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
    retain_DateTimeIndex: bool = True,
    other_cols_to_agg_dict: dict | None = None,
    y_bin_edges: list[float] | None = None,
) -> pd.DataFrame:
    """Rebin the chance of showers time_series in time and y prior to modeling.

    Args:
        dfp_in: The input dataframe to rebin.
        time_col: The time column.
        y_col: The y column.
        time_bin_size: The size of time bins.
            Must be less than 1 hour and a divisor of 60 minutes, e.g. 60 % time_bin_size_in_minutes == 0, with the current implementation. This is not an issue in the chance of showers context, but may need refactoring if this code is reused elsewhere.
        retain_DateTimeIndex: Keep the rebinned time_col as a pandas DateTimeIndex of the dataframe, with a regular time_col as well, or drop it for a normal RangeIndex.
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
            raise ValueError(
                f"Invalid {time_bin_size = }, {time_bin_size_minutes = }, should be between 0 and 60!"
            )

        if 60 % time_bin_size_minutes != 0:
            raise ValueError(
                f"Invalid {time_bin_size = }, {time_bin_size_minutes = }, {60 % time_bin_size_minutes = } should be 0!"
            )

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
        dfp = dfp.groupby(time_col).agg(cols_to_agg_dict)
        if retain_DateTimeIndex:
            dfp[time_col] = dfp.index
        else:
            dfp = dfp.reset_index()

    if rebin_y:
        if TYPE_CHECKING:
            assert isinstance(y_bin_edges, list)  # noqa: SCS108 # nosec assert_used

        dfp[y_col] = pd.cut(dfp[y_col], bins=y_bin_edges, right=True, labels=y_bin_edges[1:])

    return dfp


################################################################################
def create_datetime_component_cols(
    dfp: pd.DataFrame,
    datetime_col: str,
    date_fmt: str,
    time_fmt: str,
    *,
    country_code: str = "US",
    prov: str | None = None,
    state: str | None = "NY",
    language: str | None = "en_US",
) -> pd.DataFrame:
    """Add columns for day of week, time of day, holidays, etc based on datetime_col.

    Args:
        dfp: Input dataframe.
        datetime_col: Datetime column.
        date_fmt: String format of dates.
        time_fmt: String format of times.
        country_code: The country ISO code for holidays.
        prov: The province for holidays.
        state: The state for holidays.
        language: The language for holidays.

    Returns:
        Dataframe with additional columns.
    """
    dfp = dfp.copy()

    dfp["day_of_week_int"] = dfp[datetime_col].dt.dayofweek
    dfp["day_of_week_frac"] = dfp["day_of_week_int"] / 6.0
    # day_of_week_str -> .dt.day_name()

    dfp["time_of_day"] = dfp[datetime_col].dt.strftime(time_fmt)
    dfp["time_of_day_frac"] = dfp.apply(
        lambda row: pd.to_timedelta(row["time_of_day"]).total_seconds()
        / datetime.timedelta(days=1).total_seconds(),
        axis=1,
    )

    scope = range(
        dfp[datetime_col].min().year, (dfp[datetime_col].max() + pd.Timedelta(days=1)).year
    )
    country_holidays = holidays.country_holidays(
        country_code,
        language=language,
        prov=prov,
        state=state,
        years=scope,
    )

    dfp["is_holiday"] = dfp.apply(
        lambda row: row[datetime_col].strftime(date_fmt) in country_holidays,
        axis=1,
    ).astype(int)

    return dfp


################################################################################
def _get_key_from_platform() -> bytes:
    """Create a key for pickles from platform properties.

    You should really load a private key from somewhere on the system!
    However, this code is only really going to be used to share a local file between programs on the same machine.
    Including a signed header in pickle files is already a bit of security theater for this use case,
    really being more of an exercise in how to do it, so just using the platform's properties as the key should be fine.

    Returns:
        Key made from the platform's properties.
    """
    return bytes(
        hashlib.sha256(
            "-".join(
                [
                    platform.node(),
                    platform.platform(),
                    *platform.python_build(),
                    platform.version(),
                ]
            )
            .replace(" ", "-")
            .encode("utf-8")
        ).hexdigest(),
        sys.stdin.encoding,
    )


################################################################################
def write_secure_pickle(
    data: Any, f_path: pathlib.Path, *, shared_key: None | bytes = None  # noqa: ANN401
) -> None:
    """Write data to pickle file with signed header for security.

    Adapted from:
        https://pycharm-security.readthedocs.io/en/latest/checks/PIC100.html
        https://stackoverflow.com/questions/74638045/getting-invalid-signature-for-hmac-authentication-of-python-pickle-file

    Args:
        data: The object to be pickled.
        f_path: The full path for the output pickle file.
        shared_key: The shared key to sign the file.

    Raises:
        ValueError: Bad configuration.
    """
    if shared_key is None:
        shared_key = _get_key_from_platform()

    pickle_data = pickle.dumps(data)
    digest = hmac.new(shared_key, pickle_data, hashlib.blake2b).hexdigest()

    if f_path.suffix != ".pickle":
        raise ValueError(f"f_path ends in {f_path.suffix}, must be .pickle!")

    f_path.parent.mkdir(parents=True, exist_ok=True)

    with f_path.open("wb") as f_pickle:
        f_pickle.write(bytes(digest, sys.stdin.encoding) + b"\n" + pickle_data)


################################################################################
def read_secure_pickle(
    f_path: pathlib.Path, *, shared_key: None | bytes = None
) -> Any:  # noqa: ANN401
    """Read data from pickle file with signed header for security.

    Adapted from:
        https://pycharm-security.readthedocs.io/en/latest/checks/PIC100.html
        https://stackoverflow.com/questions/74638045/getting-invalid-signature-for-hmac-authentication-of-python-pickle-file

    Args:
        f_path: The full path for the output pickle file.
        shared_key: The shared key to sign the file.

    Returns:
        Pickled data.

    Raises:
        OSError: Could not load file.
        ValueError: Bad configuration.
    """
    if shared_key is None:
        shared_key = _get_key_from_platform()

    if f_path.suffix != ".pickle":
        raise ValueError(f"f_path ends in {f_path.suffix}, must be .pickle!")

    digest = None
    pickle_data = None
    with f_path.open("rb") as f_pickle:
        digest = f_pickle.readline().rstrip()
        pickle_data = f_pickle.read()

    if digest is None or pickle_data is None:
        raise OSError(filename=str(f_path))

    recomputed = hmac.new(shared_key, pickle_data, hashlib.blake2b).hexdigest()
    if not hmac.compare_digest(digest, bytes(recomputed, sys.stdin.encoding)):
        raise ValueError("Invalid signature!!")

    return pickle.loads(pickle_data)  # noqa: DUO103, SCS113 # nosec B301
