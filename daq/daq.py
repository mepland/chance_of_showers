"""Run DAQ script."""
################################################################################
# display options
# TEMPORARY pylint: disable=C0103
log_to_file = True
display_terminal = True
display_terminal_overwrite = True
display_oled = True
display_web = True
display_web_logging_terminal = False
verbose = 0

################################################################################
# DAQ parameters
starting_time_minutes_mod = 1
averaging_period_seconds = starting_time_minutes_mod * 60
polling_period_seconds = 1
# DAQ_max_value = 65472

date_fmt = "%Y-%m-%d"
time_fmt = "%H:%M:%S"
datetime_fmt = f"{date_fmt} {time_fmt}"

m_path = "~/chance_of_showers/daq"

# pressure variables and functions
observed_pressure_min = 6400
observed_pressure_max = 14000
# TEMPORARY pylint: enable=C0103

################################################################################
# python imports
import datetime
import logging
import os
import signal
import sys
import threading
import time
import traceback
from csv import writer
from types import FrameType
from typing import List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pause

################################################################################
# paths
m_path = os.path.expanduser(m_path)
os.makedirs(f"{m_path}/raw_data", exist_ok=True)
os.makedirs(f"{m_path}/logs", exist_ok=True)

################################################################################
# logging
logger_daq = logging.getLogger("daq")
logger_daq.setLevel(logging.INFO)
if 0 < verbose:
    logger_daq.setLevel(logging.DEBUG)

if log_to_file:
    log_datetime = datetime.datetime.now().astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d-%H-%M-%S")

    fname_log = f"daq_{log_datetime}.log"
    logging_fh = logging.FileHandler(f"{m_path}/logs/{fname_log}")
    logging_fh.setLevel(logging.INFO)
    if 0 < verbose:
        logging_fh.setLevel(logging.DEBUG)

    logging_formatter = logging.Formatter(
        "%(asctime)s [%(name)-8.8s] [%(threadName)-10.10s] [%(levelname)-8.8s] %(message)s",
        datetime_fmt,
    )

    logging_fh.setFormatter(logging_formatter)

    logger_daq.addHandler(logging_fh)


def my_print(
    line: str,
    logger_level: Optional[int] = logging.INFO,
    use_print: bool = display_terminal,
    print_prefix: str = "",
    print_postfix: str = "",
    use_stdout_overwrite: bool = False,
) -> None:
    """Custom print function to print to screen, possibly with previous line overwriting, and log files.

    We could just add a StreamHandler to logger,
    but as we also want to erase lines on stdout we define our own print function instead.
    """
    if not log_to_file or logger_level is None:
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
            f"Unknown {logger_level=}, {type(logger_level)=} in my_print, logging as {logging.CRITICAL=}"
        )
        logger_daq.critical(line)

    if use_print:
        print(f"{print_prefix}{line}{print_postfix}")

    if use_stdout_overwrite:
        # https://stackoverflow.com/a/39177802
        sys.stdout.write("\x1b[1A\x1b[2K" + line + "\r")
        sys.stdout.flush()


################################################################################
# helper variables and functions


def normalize_pressure_value(pressure_value: int) -> float:
    """Normalize raw ADC pressure_value.

    observed_pressure_min (observed_pressure_max) maps to 0 (1).
    """
    try:
        normalize_pressure_value_float = (pressure_value - observed_pressure_min) / (
            observed_pressure_max - observed_pressure_min
        )
    except Exception as error:
        # don't want to kill the DAQ just because of a display problem
        # Note normalize_pressure_value() is only used to populate the displays, not save the raw data
        my_print(
            f"Unexpected error in normalize_pressure_value():\n{error=}\n{type(error)=}\n{traceback.format_exc()}\nContinuing",
            logger_level=logging.DEBUG,
            use_print=False,
        )
        normalize_pressure_value_float = -1
    return normalize_pressure_value_float


# DAQ variables
n_polling = int(np.ceil(averaging_period_seconds / polling_period_seconds))
# test if defining here first saves memory?
polling_pressure_samples = np.empty(n_polling)
polling_pressure_samples.fill(np.nan)
polling_flow_samples = np.zeros(n_polling)


def get_SoC_temp() -> float:  # pylint: disable=invalid-name
    """Get SoC's temperature."""
    try:
        res = os.popen("vcgencmd measure_temp").readline()  # nosec B605, B607
        temp = float(res.replace("temp=", "").replace("'C\n", ""))
    except Exception as error:
        # don't want to kill the DAQ just because of a problem reading the SoC temp
        my_print(
            f"Unexpected error in get_SoC_temp():\n{error=}\n{type(error)=}\n{traceback.format_exc()}\nContinuing",
            logger_level=logging.DEBUG,
            use_print=False,
        )
        pass

    return temp


thread_daq_loop = None  # pylint: disable=C0103
running_daq_loop = True  # pylint: disable=C0103


def signal_handler(
    signal: int, frame: Optional[FrameType]  # pylint: disable=unused-argument,redefined-outer-name
) -> None:
    """Catch ctrl+c and kill, and shut down gracefully.

    https://stackoverflow.com/a/38665760
    Use the running_daq_loop variable and a pause of 2 * polling_period_seconds seconds to end the daq_loop() thread gracefully
    """
    global running_daq_loop
    my_print(
        f"DAQ loop will exit gracefully in {2 * polling_period_seconds} seconds",
        print_prefix="\n",
    )
    running_daq_loop = False
    time.sleep(2 * polling_period_seconds)
    my_print("DAQ exiting gracefully", print_prefix="\n")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

################################################################################
# Setup connection to MCP3008 ADC
# https://docs.circuitpython.org/projects/mcp3xxx/en/stable/index.html#mcp3008-single-ended
# https://docs.circuitpython.org/projects/mcp3xxx/en/stable/api.html#adafruit_mcp3xxx.analog_in.AnalogIn
# chan_0.value = Returns the value of an ADC pin as an integer. Due to 10-bit accuracy of the chip, the returned values range [0, 65472].
# chan_0.voltage = Returns the voltage from the ADC pin as a floating point value. Due to the 10-bit accuracy of the chip, returned values range from 0 to (reference_voltage * 65472 / 65535)

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

had_flow = 0  # pylint: disable=C0103


def rise() -> None:
    """Flow sensor rise action."""
    global had_flow
    had_flow = 1


def fall() -> None:
    """Flow sensor fall action."""
    global had_flow
    had_flow = 0


# bounce_time and hold_time are in seconds
flow_switch = Button(pin=19, pull_up=False, bounce_time=0.1, hold_time=1)
flow_switch.when_held = rise
flow_switch.when_released = fall

################################################################################
# Setup connection to i2c display
# https://luma-oled.readthedocs.io/en/latest
if display_oled:
    from luma.core.error import DeviceNotFoundError
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import sh1106
    from PIL import ImageFont

    i2c_device = sh1106(i2c(port=1, address=0x3C), rotate=0)

    try:
        OLED_FONT_SIZE = 14
        OLED_FONT = ImageFont.truetype("DejaVuSans.ttf", size=OLED_FONT_SIZE)
    except OSError:
        OLED_FONT_SIZE = 12
        # ImageFont and FreeTypeFont behave the same in draw.text()
        OLED_FONT = ImageFont.load_default()  # type: ignore[assignment]
    except Exception as error_oled_font:
        my_print(
            f"Unexpected error in ImageFont:\n{error_oled_font=}\n{type(error_oled_font)=}\n{traceback.format_exc()}\nExiting!",
            logger_level=logging.ERROR,
        )
        sys.exit(1)

    def paint_oled(
        lines: List[str],
        lpad: float = 4.0,
        vpad: float = 0.0,
        line_height: int = OLED_FONT_SIZE,
        bounding_box: bool = False,
    ) -> None:
        """Function to paint OLED display."""
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
                # f"Expected error in paint_oled():\n{error=}\n{type(error)=}\n{traceback.format_exc()}\nContinuing",
                # f"Expected error in paint_oled(): {error=}, Continuing",
                # logger_level=logging.DEBUG,
                line="",
                logger_level=None,
                use_print=False,
            )
            pass
        except Exception as error:
            # don't want to kill the DAQ just because of an OLED problem
            my_print(
                f"Unexpected error in paint_oled():\n{error=}\n{type(error)=}\n{traceback.format_exc()}\nContinuing",
                logger_level=logging.DEBUG,
                use_print=False,
            )
            pass


################################################################################
# Setup web page
# following https://github.com/donskytech/dht22-weather-station-python-flask-socketio
new_connection = False  # pylint: disable=C0103
if display_web:  # noqa: C901
    import json
    import socket

    import python_arptable
    from flask import Flask, render_template, request
    from flask_socketio import SocketIO

    flask_app = Flask(
        __name__,
        static_url_path="",
        static_folder="web/static",
        template_folder="web/templates",
    )

    flask_app.config["SECRET_KEY"] = "test"  # nosec B105
    flask_app.config["TEMPLATES_AUTO_RELOAD"] = True

    logger_sio = logging.getLogger("sio")
    logger_sio.setLevel(logging.WARNING)
    if 1 < verbose:
        logger_sio.setLevel(logging.DEBUG)
    if log_to_file:
        logger_sio.addHandler(logging_fh)

    sio = SocketIO(
        flask_app,
        cors_allowed_origins="*",
        logger=logger_sio,
        engineio_logger=logger_sio,
    )

    @flask_app.route("/")
    def index() -> str:
        """Serve index.html."""
        try:
            return render_template("index.html")
        except Exception as error:
            # don't want to kill the DAQ just because of a web problem
            my_print(
                f"Unexpected error in index():\n{error=}\n{type(error)=}\n{traceback.format_exc()}\nContinuing",
                logger_level=logging.DEBUG,
                use_print=False,
            )
            return ""

    def conn_details() -> str:
        """Get connection details.

        ip address and mac address
        """
        try:
            ip_address = request.remote_addr
            mac_address = "Unknown"
            for _ in python_arptable.get_arp_table():
                if _.get("IP address", None) == ip_address:
                    mac_address = _.get("HW address", mac_address)
                    break
            conn_details_str = (
                f"sid: {request.sid}"  # type: ignore[attr-defined]
                + ", IP address: {ip_address}"
                + ", MAC address: {mac_address}"
            )

        except Exception as error:
            # don't want to kill the DAQ just because of a web problem
            my_print(
                f"Unexpected error in conn_details():\n{error=}\n{type(error)=}\n{traceback.format_exc()}\nContinuing",
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
                use_print=(display_terminal and display_web_logging_terminal),
            )
        except Exception as error:
            # don't want to kill the DAQ just because of a web problem
            my_print(
                f"Unexpected error in connect():\n{error=}\n{type(error)=}\n{traceback.format_exc()}\nContinuing",
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
                use_print=(display_terminal and display_web_logging_terminal),
            )
        except Exception as error:
            # don't want to kill the DAQ just because of a web problem
            my_print(
                f"Unexpected error in disconnect():\n{error=}\n{type(error)=}\n{traceback.format_exc()}\nContinuing",
                logger_level=logging.DEBUG,
                use_print=False,
            )
            pass

    if not (0 < verbose or display_web_logging_terminal):
        # No messages in terminal
        import flask.cli

        flask.cli.show_server_banner = lambda *args: None

    # never write werkzeug logs to terminal
    log_werkzeug = logging.getLogger("werkzeug")
    log_werkzeug.setLevel(logging.WARNING)
    if 0 < verbose:
        log_werkzeug.setLevel(logging.DEBUG)
    if log_to_file:
        log_werkzeug.addHandler(logging_fh)

    N_LAST = 15
    t_est_str_n_last: List[str] = []
    mean_pressure_value_normalized_n_last: List[float] = []
    past_had_flow_n_last: List[int] = []

    def get_ip_address() -> str:
        """Get ip address of host machine.

        https://stackoverflow.com/a/28950776
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

    PORT_NUMBER = 5000
    host_ip_address = get_ip_address()

################################################################################
# Wait until UTC minutes is mod starting_time_minutes_mod
# Then if the script is interrupted, we can resume on the same cadence
t_start = datetime.datetime.now()
t_start_minute = (
    t_start.minute - (t_start.minute % starting_time_minutes_mod) + starting_time_minutes_mod
) % 60

t_start = t_start.replace(minute=t_start_minute, second=0, microsecond=0)

t_utc_str = t_start.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
t_est_str = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(datetime_fmt)

if log_to_file:
    my_print(f"Logging to {fname_log}", print_prefix="\n")
if display_web:
    my_print(
        f"Live dashboard hosted at: http://{host_ip_address}:{PORT_NUMBER}",
        print_prefix="\n",
    )
my_print(f"Starting DAQ at {t_utc_str} UTC, {t_est_str} EST", print_prefix="\n       ")

if display_oled:
    # write to OLED display
    t_est_str_short = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(time_fmt)
    paint_oled(
        ["Will start at:", t_est_str_short, f"SoC: {get_SoC_temp()}"],
        bounding_box=True,
    )

pause.until(t_start)

t_start = datetime.datetime.now()
t_utc_str = t_start.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
t_est_str = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(datetime_fmt)
my_print(
    f"Started taking data at {t_utc_str} UTC, {t_est_str} EST",
    print_prefix="\n",
    print_postfix="\n\n",
)


################################################################################
def daq_loop() -> None:
    """DAQ loop."""
    global new_connection
    global t_utc_str
    global t_est_str
    global t_start
    global had_flow
    global polling_pressure_samples  # pylint: disable=global-variable-not-assigned
    global polling_flow_samples
    mean_pressure_value = -1
    mean_pressure_value_normalized = -1.0
    past_had_flow = -1
    while running_daq_loop:
        # Set seconds to 0 to avoid drift over multiple hours / days
        t_start = datetime.datetime.now().replace(second=0, microsecond=0)
        t_stop = t_start

        # average over averaging_period_seconds
        i_polling = 0
        # reset variables
        had_flow = 0  # avoid sticking high if we lose pressure while flowing
        polling_pressure_samples.fill(np.nan)
        polling_flow_samples = np.zeros(n_polling)
        while running_daq_loop and t_stop - t_start < datetime.timedelta(
            seconds=averaging_period_seconds
        ):
            # sample pressure and flow
            pressure_value = int(chan_0.value)
            flow_value = int(had_flow)

            # save data point to array
            polling_pressure_samples[i_polling] = pressure_value
            polling_flow_samples[i_polling] = flow_value

            # display
            pressure_value_normalized = normalize_pressure_value(pressure_value)

            line1 = f"{t_utc_str} UTC Mean Pressure: {mean_pressure_value:5d}, {mean_pressure_value_normalized:4.0%}, Flow: {past_had_flow}"
            line2 = f"i = {i_polling:3d}              Current Pressure: {pressure_value:5.0f}, {pressure_value_normalized:4.0%}, Flow: {flow_value}"
            my_print(
                f"{line1}\n{line2}",
                logger_level=logging.DEBUG,
                use_print=(display_terminal and not display_terminal_overwrite),
                use_stdout_overwrite=(display_terminal and display_terminal_overwrite),
            )

            if display_oled:
                # write to OLED display
                paint_oled(
                    [
                        f"Pressure: {pressure_value_normalized:4.0%}",
                        f"Pressure: {pressure_value:5.0f}",
                        f"Flow: {flow_value}",
                        f"SoC: {get_SoC_temp()}",
                    ]
                )

            if display_web:
                try:
                    # send data to socket
                    _data = {
                        # time
                        # "t_utc_str": t_utc_str,
                        "t_est_str": t_est_str,
                        "i_polling": i_polling,
                        # live values
                        "pressure_value": pressure_value,
                        "pressure_value_normalized": pressure_value_normalized,
                        "had_flow": flow_value,
                    }
                    # N_LAST mean values
                    if i_polling == 0 or new_connection:
                        new_connection = False
                        _data["t_est_str_n_last"] = t_est_str_n_last
                        _data[
                            "mean_pressure_value_normalized_n_last"
                        ] = mean_pressure_value_normalized_n_last
                        _data["past_had_flow_n_last"] = past_had_flow_n_last

                    sio.emit("emit_data", json.dumps(_data))
                except Exception as error:
                    # don't want to kill the DAQ just because of a web problem
                    my_print(
                        f"Unexpected error in sio.emit():\n{error=}\n{type(error)=}\n{traceback.format_exc()}\nContinuing",
                        logger_level=logging.DEBUG,
                        use_print=False,
                    )
                    pass

            # wait polling_period_seconds between data points to average
            while datetime.datetime.now() - t_stop < datetime.timedelta(
                seconds=polling_period_seconds
            ):
                pass

            i_polling += 1
            t_stop = datetime.datetime.now()

        # process polling results if DAQ is still running
        if running_daq_loop:
            # take mean and save data point to csv
            t_utc_str = t_stop.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
            if display_web:
                t_est_str = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(datetime_fmt)
            mean_pressure_value = int(np.nanmean(polling_pressure_samples))
            mean_pressure_value_normalized = normalize_pressure_value(mean_pressure_value)
            past_had_flow = int(np.max(polling_flow_samples))
            new_row = [t_utc_str, mean_pressure_value, past_had_flow]

            fname_date = t_stop.astimezone(ZoneInfo("UTC")).strftime(date_fmt)
            with open(f"{m_path}/raw_data/date_{fname_date}.csv", "a", encoding="utf-8") as f_csv:
                m_writer = writer(f_csv)
                if f_csv.tell() == 0:
                    # empty file, create header
                    m_writer.writerow(["datetime_utc", "mean_pressure_value", "had_flow"])
                m_writer.writerow(new_row)
                f_csv.close()

            if display_web:
                try:
                    # save N_LAST mean values
                    t_est_str_n_last.append(t_est_str)
                    mean_pressure_value_normalized_n_last.append(mean_pressure_value_normalized)
                    past_had_flow_n_last.append(past_had_flow)

                    if N_LAST < len(t_est_str_n_last):
                        del t_est_str_n_last[0]
                        del mean_pressure_value_normalized_n_last[0]
                        del past_had_flow_n_last[0]
                except Exception as error:
                    # don't want to kill the DAQ just because of a web problem
                    my_print(
                        f"Unexpected error updating _n_last lists:\n{error=}\n{type(error)=}\n{traceback.format_exc()}\nContinuing",
                        logger_level=logging.DEBUG,
                        use_print=False,
                    )
                    pass

    my_print(f"Exiting daq_loop() via {running_daq_loop=}")


################################################################################
# start daq_loop()
if thread_daq_loop is None:
    # kill gracefully via running_daq_loop
    thread_daq_loop = threading.Thread(target=daq_loop)
    thread_daq_loop.start()

################################################################################
# serve index.html
if display_web:
    try:
        sio.run(
            flask_app,
            port=PORT_NUMBER,
            host="0.0.0.0",  # nosec B104
            # debug must be false to avoid duplicate threads of the entire script!
            debug=False,
        )
    except Exception as error_sio_run:
        # don't want to kill the DAQ just because of a web problem
        my_print(
            f"Unexpected error in sio.run():\n{error_sio_run=}\n{type(error_sio_run)=}\n{traceback.format_exc()}\nContinuing",
            logger_level=logging.DEBUG,
            use_print=False,
        )
        pass

################################################################################
# run daq_loop() until we exit the main thread
if thread_daq_loop is not None:
    thread_daq_loop.join()
