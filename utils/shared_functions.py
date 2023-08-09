"""Shared functions."""

################################################################################
# python imports
import os
import socket
import sys


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
