"""Run DAQ script."""

from __future__ import annotations

################################################################################
# python imports
import datetime
import logging
import pathlib
import signal
import socket
import sys
import threading
import time
import traceback
import zoneinfo
from csv import writer
from typing import TYPE_CHECKING, Final

import humanize
import hydra
import numpy as np
import pause
import psutil
from omegaconf import DictConfig  # noqa: TC002

__all__: list[str] = []


if TYPE_CHECKING:
    from types import FrameType

# pylint: disable=import-outside-toplevel

################################################################################
# Global variables
# pylint: disable=invalid-name
thread_daq_loop = None
running_daq_loop = True
had_flow = 0
new_connection = False
# pylint: enable=invalid-name


@hydra.main(version_base=None, config_path="..", config_name="config")
def daq(  # noqa: C901 # pylint: disable=too-many-statements, too-many-locals
    cfg: DictConfig,
) -> None:
    """Run the DAQ script.

    Args:
        cfg: Hydra configuration.

    Raises:
        ValueError: Bad configuration.
    """
    global thread_daq_loop
    global running_daq_loop
    global had_flow
    global new_connection
    ################################################################################
    # Setup variables
    # pylint: disable=invalid-name
    LOG_TO_FILE: Final = cfg["daq"]["log_to_file"]
    DISPLAY_TERMINAL: Final = cfg["daq"]["display_terminal"]
    DISPLAY_TERMINAL_OVERWRITE: Final = cfg["daq"]["display_terminal_overwrite"]
    DISPLAY_OLED: Final = cfg["daq"]["display_oled"]
    DISPLAY_WEB: Final = cfg["daq"]["display_web"]
    DISPLAY_WEB_LOGGING_TERMINAL: Final = cfg["daq"]["display_web_logging_terminal"]
    VERBOSITY: Final = cfg["daq"]["verbosity"]

    STARTING_TIME_MINUTES_MOD: Final = cfg["daq"]["starting_time_minutes_mod"]
    AVERAGING_PERIOD_SECONDS: Final = cfg["daq"]["averaging_period_seconds"]
    POLLING_PERIOD_SECONDS: Final = cfg["daq"]["polling_period_seconds"]

    DATE_FMT: Final = cfg["general"]["date_fmt"]
    TIME_FMT: Final = cfg["general"]["time_fmt"]
    DATETIME_FMT: Final = f"{DATE_FMT} {TIME_FMT}"
    FNAME_DATETIME_FMT: Final = cfg["general"]["fname_datetime_fmt"]
    LOCAL_TIMEZONE_STR: Final = cfg["general"]["local_timezone"]

    if LOCAL_TIMEZONE_STR not in zoneinfo.available_timezones():
        AVAILABLE_TIMEZONES: Final = "\n".join(list(zoneinfo.available_timezones()))
        raise ValueError(f"Unknown {LOCAL_TIMEZONE_STR = }, choose from:\n{AVAILABLE_TIMEZONES}")

    UTC_TIMEZONE: Final = zoneinfo.ZoneInfo("UTC")
    LOCAL_TIMEZONE: Final = zoneinfo.ZoneInfo(LOCAL_TIMEZONE_STR)

    PACKAGE_PATH: Final = pathlib.Path(cfg["general"]["package_path"]).expanduser()

    RAW_DATA_RELATIVE_PATH: Final = cfg["daq"]["raw_data_relative_path"]
    LOGS_RELATIVE_PATH: Final = cfg["daq"]["logs_relative_path"]

    LOG_MEMORY_USAGE: Final = cfg["daq"]["log_memory_usage"]
    LOG_MEMORY_USAGE_MINUTES_MOD: Final = cfg["daq"]["log_memory_usage_minutes_mod"]

    N_LAST_POINTS_WEB: Final = cfg["daq"]["n_last_points_web"]

    OBSERVED_PRESSURE_MIN: Final = cfg["general"]["observed_pressure_min"]
    OBSERVED_PRESSURE_MAX: Final = cfg["general"]["observed_pressure_max"]

    # DAQ variables
    N_POLLING: Final = int(np.ceil(AVERAGING_PERIOD_SECONDS / POLLING_PERIOD_SECONDS))
    # pylint: enable=invalid-name

    # Defining these arrays here first saves memory
    polling_pressure_samples = np.empty(N_POLLING)
    polling_pressure_samples.fill(np.nan)
    polling_flow_samples = np.zeros(N_POLLING)  # noqa: F841 # pylint: disable=unused-variable

    ################################################################################
    # Lock script, avoid launching duplicates
    sys.path.append(str(pathlib.Path.cwd().parent))
    from utils.shared_functions import get_lock, get_SoC_temp, normalize_pressure_value

    get_lock("daq")

    ################################################################################
    # Paths
    RAW_DATA_FULL_PATH: Final = (  # pylint: disable=invalid-name
        PACKAGE_PATH / RAW_DATA_RELATIVE_PATH
    )
    LOGS_FULL_PATH: Final = PACKAGE_PATH / LOGS_RELATIVE_PATH  # pylint: disable=invalid-name
    RAW_DATA_FULL_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_FULL_PATH.mkdir(parents=True, exist_ok=True)

    ################################################################################
    # Logging
    logger_daq = logging.getLogger("daq")
    logger_daq.setLevel(logging.INFO)
    if 0 < VERBOSITY:
        logger_daq.setLevel(logging.DEBUG)

    if LOG_TO_FILE:
        LOG_DATETIME: Final = datetime.datetime.now(  # pylint: disable=invalid-name
            UTC_TIMEZONE
        ).strftime(FNAME_DATETIME_FMT)

        FNAME_LOG: Final = f"daq_{LOG_DATETIME}.log"  # pylint: disable=invalid-name
        logging_fh = logging.FileHandler(LOGS_FULL_PATH / FNAME_LOG)
        logging_fh.setLevel(logging.INFO)
        if 0 < VERBOSITY:
            logging_fh.setLevel(logging.DEBUG)

        logging_formatter = logging.Formatter(
            "%(asctime)s [%(name)-8.8s] [%(threadName)-10.10s] [%(levelname)-8.8s] %(message)s",
            f"{DATETIME_FMT} UTC",
        )
        # https://docs.python.org/3/library/logging.html#logging.Formatter.formatTime
        # https://docs.python.org/3/library/time.html#time.gmtime
        logging_formatter.converter = time.gmtime  # GMT = UTC

        logging_fh.setFormatter(logging_formatter)

        logger_daq.addHandler(logging_fh)

    def my_print(
        line: str,
        *,
        logger_level: int | None = logging.INFO,
        use_print: bool = DISPLAY_TERMINAL,
        print_prefix: str = "",
        print_postfix: str = "",
        use_stdout_overwrite: bool = False,
    ) -> None:
        """Custom print function to print to screen and log files.

        We could just add a StreamHandler to logger,
        but as we also want to erase lines on stdout we define our own print function instead.

        Args:
            line: The line to be printed.
            logger_level: The level to use for logging, None to disable logging.
            use_print: Flag for printing to stdout.
            print_prefix: String to prepend to `line` before printing to stdout.
            print_postfix: String to append to `line` before printing to stdout.
            use_stdout_overwrite: Flag for overwriting the previous line on stdout.
        """
        if not LOG_TO_FILE or logger_level is None:
            pass
        elif logger_level == logging.CRITICAL:
            logger_daq.critical(line)
        elif logger_level == logging.ERROR:
            logger_daq.error(line)
        elif logger_level == logging.WARNING:
            logger_daq.warning(line)
        elif logger_level == logging.INFO:
            logger_daq.info(line)
        elif logger_level == logging.DEBUG:
            logger_daq.debug(line)
        else:
            logger_daq.critical(
                f"Unknown {logger_level = }, {type(logger_level) = } in my_print, logging as {logging.CRITICAL = }"
            )
            logger_daq.critical(line)

        if use_print:
            print(f"{print_prefix}{line}{print_postfix}")
        elif use_stdout_overwrite:
            # https://stackoverflow.com/a/39177802
            sys.stdout.write("\x1b[1A\x1b[2K" + line + "\r")
            sys.stdout.flush()

    ################################################################################
    # Helper variables and functions

    def normalize_pressure_value_safe(pressure_value: int) -> float:
        """Normalize raw ADC pressure_value.

        Args:
            pressure_value: The raw ADC pressure_value to be normalized.

        Returns:
            (pressure_value-observed_pressure_min)/(observed_pressure_max-observed_pressure_min),
            i.e. observed_pressure_min (observed_pressure_max) maps to 0 (1).
        """
        normalize_pressure_value_float = -1.0
        try:
            normalize_pressure_value_float = normalize_pressure_value(
                pressure_value, OBSERVED_PRESSURE_MIN, OBSERVED_PRESSURE_MAX
            )
        except Exception as error:
            # don't want to kill the DAQ just because of a display problem
            # Note normalize_pressure_value() is only used to populate the displays, not save the raw data
            my_print(
                f"Unexpected error in normalize_pressure_value():\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nContinuing",
                logger_level=logging.DEBUG,
                use_print=False,
            )

        return normalize_pressure_value_float

    def get_SoC_temp_safe() -> float:  # pylint: disable=invalid-name
        """Get SoC's temperature.

        Returns:
            SoC temperature as a float.
        """
        temp = -1.0
        try:
            temp = get_SoC_temp()
        except Exception as error:
            # don't want to kill the DAQ just because of a problem reading the SoC temp
            my_print(
                f"Unexpected error in get_SoC_temp():\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nContinuing",
                logger_level=logging.DEBUG,
                use_print=False,
            )
            pass

        return temp

    def signal_handler(
        dummy_signal: int,  # noqa: U100 # pylint: disable=unused-argument
        dummy_frame: FrameType | None,  # noqa: U100 # pylint: disable=unused-argument
    ) -> None:
        """Catch ctrl+c and kill, and shut down gracefully.

        https://stackoverflow.com/a/38665760
        Use the running_daq_loop variable and a pause of 2 * polling_period_seconds seconds to end the daq_loop() thread gracefully

        Args:
            dummy_signal: signal number.
            dummy_frame: Frame object.
        """
        global running_daq_loop
        my_print(
            f"DAQ loop will exit gracefully in {2 * POLLING_PERIOD_SECONDS} seconds",
            print_prefix="\n",
        )
        running_daq_loop = False
        time.sleep(2 * POLLING_PERIOD_SECONDS)
        my_print("DAQ exiting gracefully", print_prefix="\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    ################################################################################
    # Setup connection to MCP3008 ADC
    # https://docs.circuitpython.org/projects/mcp3xxx/en/stable/index.html#mcp3008-single-ended
    # https://docs.circuitpython.org/projects/mcp3xxx/en/stable/api.html#adafruit_mcp3xxx.analog_in.AnalogIn
    # chan_0.value = Returns the value of an ADC pin as an integer. Due to 10-bit accuracy of the chip, the returned values range [0, 65472].
    # chan_0.voltage = Returns the voltage from the ADC pin as a floating point value. Due to the 10-bit accuracy of the chip, returned values range from 0 to (reference_voltage * 65472 / 65535)
    # DAQ max value is 65472

    import adafruit_mcp3xxx.mcp3008 as MCP
    import board
    import busio
    import digitalio
    from adafruit_mcp3xxx.analog_in import AnalogIn

    spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
    cs = digitalio.DigitalInOut(board.D5)  # GPIO pin 5
    mcp = MCP.MCP3008(spi, cs, ref_voltage=5)  # 5 Volts
    chan_0 = AnalogIn(mcp, MCP.P0)  # MCP3008 pin 0

    ################################################################################
    # Setup connection for reading the flow sensor as a switch
    # https://gpiozero.readthedocs.io/en/stable/api_input.html?highlight=Button#gpiozero.Button
    # Note, I would prefer to read the pulses per minute with RPi.GPIO as in fan_control.py,
    # but my flow sensor only produces a constant Vcc while flow is occurring, no pulses.
    from gpiozero import Button
    from gpiozero.pins.rpigpio import RPiGPIOFactory

    def rise() -> None:
        """Flow sensor rise action."""
        global had_flow
        had_flow = 1

    def fall() -> None:
        """Flow sensor fall action."""
        global had_flow
        had_flow = 0

    # bounce_time and hold_time are in seconds
    factory = RPiGPIOFactory()
    flow_switch = Button(pin=19, pull_up=False, bounce_time=0.1, hold_time=1, pin_factory=factory)
    flow_switch.when_held = rise
    flow_switch.when_released = fall

    ################################################################################
    # Setup connection to i2c display
    # https://luma-oled.readthedocs.io/en/latest
    if DISPLAY_OLED:
        from luma.core.error import DeviceNotFoundError
        from luma.core.interface.serial import i2c
        from luma.core.render import canvas
        from luma.oled.device import sh1106
        from PIL import ImageFont

        i2c_device = sh1106(i2c(port=1, address=0x3C), rotate=0)

        try:
            OLED_FONT_SIZE = 14  # pylint: disable=invalid-name
            OLED_FONT = ImageFont.truetype(  # pylint: disable=invalid-name
                "DejaVuSans.ttf", size=OLED_FONT_SIZE
            )
        except OSError:
            OLED_FONT_SIZE = 12  # pylint: disable=invalid-name
            # ImageFont and FreeTypeFont behave the same in draw.text()
            OLED_FONT = ImageFont.load_default()  # type: ignore[assignment] # pylint: disable=invalid-name
        except Exception as error_oled_font:
            my_print(
                f"Unexpected error in ImageFont:\n{error_oled_font = }\n{type(error_oled_font) = }\n{traceback.format_exc()}\nExiting!",
                logger_level=logging.ERROR,
            )
            sys.exit(1)

        def paint_oled(
            lines: list[str],
            *,
            lpad: float = 4.0,
            vpad: float = 0.0,
            line_height: int = OLED_FONT_SIZE,
            bounding_box: bool = False,
        ) -> None:
            """Function to paint OLED display.

            Args:
                lines: List of strings to be printed.
                lpad: Screen left pad.
                vpad: Screen top vertical pad.
                line_height: Line height.
                bounding_box: Flag for including an outline bounding box.
            """
            try:
                with canvas(i2c_device) as draw:
                    if bounding_box:
                        draw.rectangle(i2c_device.bounding_box, outline="white", fill="black")

                    for i_line, line in enumerate(lines):
                        draw.text(
                            (lpad, vpad + i_line * line_height),
                            line,
                            fill="white",
                            font=OLED_FONT,
                        )
            except (OSError, DeviceNotFoundError, TypeError):
                # do not log device not connected errors, OLED power is probably just off
                my_print(
                    line="",
                    logger_level=None,
                    use_print=False,
                )
                pass
            except Exception as error:
                # don't want to kill the DAQ just because of an OLED problem
                my_print(
                    f"Unexpected error in paint_oled():\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nContinuing",
                    logger_level=logging.DEBUG,
                    use_print=False,
                )
                pass

    ################################################################################
    # Setup web page
    # following https://github.com/donskytech/dht22-weather-station-python-flask-socketio
    if DISPLAY_WEB:
        import json

        import python_arptable
        from flask import Flask, render_template, request
        from flask_socketio import SocketIO

        flask_app = Flask(
            __name__,
            static_url_path="",
            static_folder=str(pathlib.Path("web", "static")),
            template_folder=str(pathlib.Path("web", "templates")),
        )

        # pragma: allowlist nextline secret
        flask_app.config["SECRET_KEY"] = "test"  # nosec: B105
        flask_app.config["TEMPLATES_AUTO_RELOAD"] = True

        logger_sio = logging.getLogger("sio")
        logger_sio.setLevel(logging.WARNING)
        if 1 < VERBOSITY:
            logger_sio.setLevel(logging.DEBUG)

        if LOG_TO_FILE:
            logger_sio.addHandler(logging_fh)

        sio = SocketIO(
            flask_app,
            cors_allowed_origins="*",
            logger=logger_sio,
            engineio_logger=logger_sio,
        )

        @flask_app.route("/")
        def index() -> str:
            """Serve index.html.

            Returns:
                Rendered template.
            """
            try:
                return render_template("index.html")
            except Exception as error:
                # don't want to kill the DAQ just because of a web problem
                my_print(
                    f"Unexpected error in index():\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nContinuing",
                    logger_level=logging.DEBUG,
                    use_print=False,
                )
                return ""

        def conn_details() -> str:
            """Get connection details.

            Session ID, IP address and MAC address.

            Returns:
                Connection details as a string.
            """
            try:
                ip_address = request.remote_addr
                mac_address = "Unknown"
                for _ in python_arptable.get_arp_table():
                    if _.get("IP address") == ip_address:
                        mac_address = _.get("HW address", mac_address)
                        break

                conn_details_str = (
                    f"sid: {request.sid}"  # type: ignore[attr-defined]
                    + f", IP address: {ip_address}"
                    + f", MAC address: {mac_address}"
                )

            except Exception as error:
                # don't want to kill the DAQ just because of a web problem
                my_print(
                    f"Unexpected error in conn_details():\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nContinuing",
                    logger_level=logging.DEBUG,
                    use_print=False,
                )
                conn_details_str = "ERROR"

            return conn_details_str

        @sio.on("connect")
        def connect() -> None:
            """Decorator for connect."""
            try:
                global new_connection
                new_connection = True
                my_print(
                    f"Client connected {conn_details()}",
                    use_print=DISPLAY_TERMINAL and DISPLAY_WEB_LOGGING_TERMINAL,
                )
            except Exception as error:
                # don't want to kill the DAQ just because of a web problem
                my_print(
                    f"Unexpected error in connect():\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nContinuing",
                    logger_level=logging.DEBUG,
                    use_print=False,
                )
                pass

        @sio.on("disconnect")
        def disconnect() -> None:
            """Decorator for disconnect."""
            try:
                my_print(
                    f"Client disconnected {conn_details()}",
                    use_print=DISPLAY_TERMINAL and DISPLAY_WEB_LOGGING_TERMINAL,
                )
            except Exception as error:
                # don't want to kill the DAQ just because of a web problem
                my_print(
                    f"Unexpected error in disconnect():\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nContinuing",
                    logger_level=logging.DEBUG,
                    use_print=False,
                )
                pass

        if not (0 < VERBOSITY or DISPLAY_WEB_LOGGING_TERMINAL):
            # No messages in terminal
            import flask.cli

            flask.cli.show_server_banner = lambda *args: None  # noqa: U100

        # never write werkzeug logs to terminal
        log_werkzeug = logging.getLogger("werkzeug")
        log_werkzeug.setLevel(logging.WARNING)
        if 0 < VERBOSITY:
            log_werkzeug.setLevel(logging.DEBUG)

        if LOG_TO_FILE:
            log_werkzeug.addHandler(logging_fh)

        t_local_str_n_last: list[str] = []
        mean_pressure_value_normalized_n_last: list[float] = []
        past_had_flow_n_last: list[int] = []

        def get_ip_address() -> str:
            """Get IP address of host machine.

            https://stackoverflow.com/a/28950776

            Returns:
                IP address of host as a string.
            """
            m_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            m_socket.settimeout(0)
            try:
                # doesn't even have to be reachable
                m_socket.connect(("10.254.254.254", 1))
                ip_address = m_socket.getsockname()[0]
            except Exception:
                ip_address = "127.0.0.1"
            finally:
                m_socket.close()

            return ip_address

        PORT_NUMBER: Final = 5000  # pylint: disable=invalid-name
        host_ip_address = get_ip_address()

    ################################################################################
    # Wait until UTC minutes is mod STARTING_TIME_MINUTES_MOD
    # Then if the script is interrupted, we can resume on the same cadence
    t_start = datetime.datetime.now(UTC_TIMEZONE)
    t_start_minute = (
        t_start.minute - (t_start.minute % STARTING_TIME_MINUTES_MOD) + STARTING_TIME_MINUTES_MOD
    ) % 60

    t_start = t_start.replace(minute=t_start_minute, second=0, microsecond=0)

    t_utc_str = t_start.astimezone(UTC_TIMEZONE).strftime(DATETIME_FMT)
    t_local_str = t_start.astimezone(LOCAL_TIMEZONE).strftime(DATETIME_FMT)

    if LOG_TO_FILE:
        my_print(f"Logging to {FNAME_LOG}", print_prefix="\n")

    if DISPLAY_WEB:
        my_print(
            f"Live dashboard hosted at: http://{host_ip_address}:{PORT_NUMBER}",
            print_prefix="\n",
        )

    my_print(
        f"Starting DAQ at {t_utc_str} UTC, {t_local_str} {LOCAL_TIMEZONE_STR}",
        print_prefix="\n       ",
    )

    if DISPLAY_OLED:
        # write to OLED display
        t_local_str_short = t_start.astimezone(LOCAL_TIMEZONE).strftime(TIME_FMT)
        paint_oled(
            ["Will start at:", t_local_str_short, f"SoC: {get_SoC_temp_safe()}"],
            bounding_box=True,
        )

    pause.until(t_start)

    t_start = datetime.datetime.now(UTC_TIMEZONE)
    t_utc_str = t_start.astimezone(UTC_TIMEZONE).strftime(DATETIME_FMT)
    t_local_str = t_start.astimezone(LOCAL_TIMEZONE).strftime(DATETIME_FMT)
    my_print(
        f"Started taking data at {t_utc_str} UTC, {t_local_str} {LOCAL_TIMEZONE_STR}",
        print_prefix="\n",
        print_postfix="\n\n",
    )

    ################################################################################
    def daq_loop(t_utc_str: str, t_local_str: str) -> None:
        """DAQ loop.

        Args:
            t_utc_str: Starting UTC time as a string
            t_local_str: Starting local time as a string
        """
        global new_connection
        global had_flow
        mean_pressure_value = -1
        mean_pressure_value_normalized = -1.0
        past_had_flow = -1
        while running_daq_loop:
            # Set seconds to 0 to avoid drift over multiple hours / days
            t_start = datetime.datetime.now(UTC_TIMEZONE).replace(second=0, microsecond=0)
            t_stop = t_start

            # average over AVERAGING_PERIOD_SECONDS
            i_polling = 0
            # reset variables
            had_flow = 0  # avoid sticking high if we lose pressure while flowing
            polling_pressure_samples.fill(np.nan)
            polling_flow_samples = np.zeros(N_POLLING)
            while running_daq_loop and t_stop - t_start < datetime.timedelta(
                seconds=AVERAGING_PERIOD_SECONDS
            ):
                # sample pressure and flow
                pressure_value = int(chan_0.value)
                flow_value = int(had_flow)

                # save data point to array
                polling_pressure_samples[i_polling] = pressure_value
                polling_flow_samples[i_polling] = flow_value

                # display
                pressure_value_normalized = normalize_pressure_value_safe(pressure_value)

                line1 = f"{t_utc_str} UTC Mean Pressure: {mean_pressure_value:5d}, {mean_pressure_value_normalized:4.0%}, Flow: {past_had_flow}"
                line2 = f"i = {i_polling:3d}              Current Pressure: {pressure_value:5.0f}, {pressure_value_normalized:4.0%}, Flow: {flow_value}"
                my_print(
                    f"{line1}\n{line2}",
                    logger_level=logging.DEBUG,
                    use_print=DISPLAY_TERMINAL and not DISPLAY_TERMINAL_OVERWRITE,
                    use_stdout_overwrite=DISPLAY_TERMINAL and DISPLAY_TERMINAL_OVERWRITE,
                )

                if DISPLAY_OLED:
                    # write to OLED display
                    paint_oled(
                        [
                            f"Pressure: {pressure_value_normalized:4.0%}",
                            f"Pressure: {pressure_value:5.0f}",
                            f"Flow: {flow_value}",
                            f"SoC: {get_SoC_temp_safe()}",
                        ]
                    )

                if DISPLAY_WEB:
                    try:
                        # send data to socket
                        _data = {
                            # time
                            "tLocalStr": t_local_str,
                            "iPolling": i_polling,
                            # live values
                            "pressureValue": pressure_value,
                            "pressureValueNormalized": pressure_value_normalized,
                            "hadFlow": flow_value,
                        }
                        # N_LAST_POINTS_WEB mean values
                        if i_polling == 0 or new_connection:
                            new_connection = False
                            _data["tLocalStrNLast"] = t_local_str_n_last
                            _data["meanPressureValueNormalizedNLast"] = (
                                mean_pressure_value_normalized_n_last
                            )
                            _data["pastHadFlowNLast"] = past_had_flow_n_last

                        sio.emit("emitData", json.dumps(_data))
                    except Exception as error:
                        # don't want to kill the DAQ just because of a web problem
                        my_print(
                            f"Unexpected error in sio.emit():\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nContinuing",
                            logger_level=logging.DEBUG,
                            use_print=False,
                        )
                        pass

                # wait POLLING_PERIOD_SECONDS between data points to average
                while datetime.datetime.now(UTC_TIMEZONE) - t_stop < datetime.timedelta(
                    seconds=POLLING_PERIOD_SECONDS
                ):
                    pass

                i_polling += 1
                t_stop = datetime.datetime.now(UTC_TIMEZONE)

            # process polling results if DAQ is still running
            if running_daq_loop:
                # take mean and save data point to csv
                t_utc_str = t_stop.astimezone(UTC_TIMEZONE).strftime(DATETIME_FMT)
                if DISPLAY_WEB:
                    t_local_str = t_start.astimezone(LOCAL_TIMEZONE).strftime(DATETIME_FMT)

                mean_pressure_value = int(np.nanmean(polling_pressure_samples))
                mean_pressure_value_normalized = normalize_pressure_value_safe(mean_pressure_value)
                past_had_flow = int(np.max(polling_flow_samples))
                new_row = [t_utc_str, mean_pressure_value, past_had_flow]

                fname_date_utc = t_stop.astimezone(UTC_TIMEZONE).strftime(DATE_FMT)
                with (RAW_DATA_FULL_PATH / f"date_{fname_date_utc}.csv").open(
                    "a", encoding="utf-8"
                ) as f_csv:
                    m_writer = writer(f_csv)
                    if f_csv.tell() == 0:
                        # empty file, create header
                        m_writer.writerow(["datetime_utc", "mean_pressure_value", "had_flow"])

                    m_writer.writerow(new_row)
                    f_csv.close()

                if DISPLAY_WEB:
                    try:
                        # save N_LAST_POINTS_WEB mean values
                        t_local_str_n_last.append(t_local_str)
                        mean_pressure_value_normalized_n_last.append(mean_pressure_value_normalized)
                        past_had_flow_n_last.append(past_had_flow)

                        if N_LAST_POINTS_WEB < len(t_local_str_n_last):
                            del t_local_str_n_last[0]
                            del mean_pressure_value_normalized_n_last[0]
                            del past_had_flow_n_last[0]

                    except Exception as error:
                        # don't want to kill the DAQ just because of a web problem
                        my_print(
                            f"Unexpected error updating _n_last lists:\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nContinuing",
                            logger_level=logging.DEBUG,
                            use_print=False,
                        )
                        pass

                if LOG_MEMORY_USAGE:
                    try:
                        if t_start.minute % LOG_MEMORY_USAGE_MINUTES_MOD == 0:
                            ram_info = psutil.virtual_memory()
                            my_print(
                                f"RAM Available: {humanize.naturalsize(ram_info.available)}, "
                                + f"Used: {humanize.naturalsize(ram_info.used)}, "
                                + f"Percent: {ram_info.percent:.2f}%",
                                logger_level=logging.INFO,
                                use_print=False,
                            )

                    except Exception as error:
                        # don't want to kill the DAQ just because of a memory logging problem
                        my_print(
                            f"Unexpected error logging memory usage:\n{error = }\n{type(error) = }\n{traceback.format_exc()}\nContinuing",
                            logger_level=logging.DEBUG,
                            use_print=False,
                        )
                        pass

        my_print(f"Exiting daq_loop() via {running_daq_loop = }")

    ################################################################################
    # start daq_loop()
    if thread_daq_loop is None:
        # kill gracefully via running_daq_loop
        thread_daq_loop = threading.Thread(target=daq_loop, args=(t_utc_str, t_local_str))
        thread_daq_loop.start()

    ################################################################################
    # serve index.html
    if DISPLAY_WEB:
        try:
            # wait until 0 < len(t_local_str_n_last) before serving the website to avoid crashes
            while len(t_local_str_n_last) < 1:
                # check len(t_local_str_n_last) every ~ 6 seconds
                time.sleep(0.1 * AVERAGING_PERIOD_SECONDS)
                my_print(
                    "Waiting to start web server",
                    logger_level=logging.DEBUG,
                    use_print=False,
                )

            my_print(
                "Starting web server with sio.run()",
                logger_level=logging.DEBUG,
                use_print=False,
            )
            sio.run(
                flask_app,
                port=PORT_NUMBER,
                host="0.0.0.0",  # nosec: B104
                # debug must be false to avoid duplicate threads of the entire script!
                debug=False,
            )
        except Exception as error_sio_run:
            # don't want to kill the DAQ just because of a web problem
            my_print(
                f"Unexpected error in sio.run():\n{error_sio_run = }\n{type(error_sio_run) = }\n{traceback.format_exc()}\nContinuing",
                logger_level=logging.DEBUG,
                use_print=False,
            )
            pass

    ################################################################################
    # run daq_loop() until we exit the main thread
    if thread_daq_loop is not None:
        thread_daq_loop.join()


if __name__ == "__main__":
    daq()  # pylint: disable=no-value-for-parameter
