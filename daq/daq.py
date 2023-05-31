################################################################################
# display options
display_terminal = True
display_terminal_overwrite = True
display_oled = True
display_web = True  # MUST BE TRUE FOR NOW as we are using socketio to run our DAQ thread TODO disconnect these!
display_web_logging_terminal = False

################################################################################
# DAQ parameters
starting_time_minutes_mod = 1
averaging_period_seconds = starting_time_minutes_mod * 60
polling_period_seconds = 1
# DAQ_max_value = 65472

date_fmt = "%Y-%m-%d"
time_fmt = "%H:%M:%S"
datetime_fmt = f"{date_fmt} {time_fmt}"

m_path = "~/chance_of_showers/daq/raw_data"

# pressure variables and functions
observed_pressure_min = 6400
observed_pressure_max = 14000

################################################################################
# python imports
import os
import sys
import signal
import datetime
from zoneinfo import ZoneInfo
import pause
import numpy as np
from csv import writer


# catch ctrl+c and kill, shut down gracefully https://stackoverflow.com/a/38665760
def signal_handler(signal, frame):
    if display_terminal:
        print("\nDAQ exiting gracefully")

    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

################################################################################
# Setup connection to MCP3008 ADC
# https://docs.circuitpython.org/projects/mcp3xxx/en/stable/index.html#mcp3008-single-ended
# https://docs.circuitpython.org/projects/mcp3xxx/en/stable/api.html#adafruit_mcp3xxx.analog_in.AnalogIn
# chan_0.value = Returns the value of an ADC pin as an integer. Due to 10-bit accuracy of the chip, the returned values range [0, 65472].
# chan_0.voltage = Returns the voltage from the ADC pin as a floating point value. Due to the 10-bit accuracy of the chip, returned values range from 0 to (reference_voltage * 65472 / 65535)

import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
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

had_flow = int(0)


def rise(n):
    global had_flow
    had_flow = 1


def fall(n):
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
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import sh1106
    from PIL import ImageFont

    i2c_device = sh1106(i2c(port=1, address=0x3C), rotate=0)

    try:
        font_size = 14
        oled_font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except OSError:
        font_size = 12
        oled_font = ImageFont.load_default()

    def paint_oled(lines, lpad=4, vpad=0, line_height=font_size, bounding_box=False):
        try:
            with canvas(i2c_device) as draw:
                if bounding_box:
                    draw.rectangle(
                        i2c_device.bounding_box, outline="white", fill="black"
                    )
                for i_line, line in enumerate(lines):
                    draw.text(
                        (lpad, vpad + i_line * line_height),
                        line,
                        fill="white",
                        font=oled_font,
                    )
        except:
            # don't want to kill the daq just because of an OLED problem
            pass


################################################################################
# Setup web page
# following https://github.com/donskytech/dht22-weather-station-python-flask-socketio
new_connection = bool(False)
if display_web:
    import json
    from flask import Flask, render_template, request
    from flask_socketio import SocketIO

    # from threading import Lock

    # thread = None
    # thread_lock = Lock()

    app = Flask(
        __name__,
        static_url_path="",
        static_folder="web/static",
        template_folder="web/templates",
    )

    app.config["SECRET_KEY"] = "test"
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    sio = SocketIO(app, cors_allowed_origins="*")

    @app.route("/")
    def index():
        return render_template("index.html")

    # Decorator for connect
    @sio.on("connect")
    def connect():
        global new_connection
        new_connection = True
        if display_web_logging_terminal:
            print("Client connected")

    if display_web_logging_terminal:
        # Decorator for disconnect
        @sio.on("disconnect")
        def disconnect():
            print("Client disconnected", request.sid)

    else:
        # No messages in terminal
        import flask.cli

        flask.cli.show_server_banner = lambda *args: None

        import logging

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

    n_last = 15
    t_est_str_n_last = []
    mean_pressure_value_normalized_n_last = []
    past_had_flow_n_last = []


################################################################################
# helper variables and functions
def normalize_pressure_value(pressure_value):
    return (pressure_value - observed_pressure_min) / (
        observed_pressure_max - observed_pressure_min
    )


m_path = os.path.expanduser(m_path)
os.makedirs(m_path, exist_ok=True)

n_polling = int(np.ceil(averaging_period_seconds / polling_period_seconds))

# TODo test if defining here first saves memory?
polling_pressure_samples = np.empty(n_polling)
polling_pressure_samples.fill(np.nan)
polling_flow_samples = np.zeros(n_polling)


# Get SoC's temperature
def get_SoC_temp():
    res = os.popen("vcgencmd measure_temp").readline()
    temp = float(res.replace("temp=", "").replace("'C\n", ""))

    return temp


################################################################################
# Wait until UTC minutes is mod starting_time_minutes_mod
# Then if the script is interrupted, we can resume on the same cadence

t_start = datetime.datetime.now()
t_start_minute = (
    t_start.minute
    - (t_start.minute % starting_time_minutes_mod)
    + starting_time_minutes_mod
) % 60

t_start = t_start.replace(minute=t_start_minute, second=0, microsecond=0)

t_utc_str = t_start.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
t_est_str = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(datetime_fmt)
if display_terminal:
    print(f"       Starting DAQ at {t_utc_str} UTC, {t_est_str} EST")

if display_oled:
    # write to OLED display
    t_est_str_short = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(time_fmt)
    paint_oled(
        [f"Will start at:", t_est_str_short, f"SoC: {get_SoC_temp()}"],
        bounding_box=True,
    )

pause.until(t_start)

t_start = datetime.datetime.now()
t_utc_str = t_start.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
if display_terminal:
    t_est_str = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(datetime_fmt)
    print(f"\nStarted taking data at {t_utc_str} UTC, {t_est_str} EST\n\n")


################################################################################
# daq loop
def daq_loop():
    global new_connection
    global t_utc_str
    global t_est_str
    global had_flow
    mean_pressure_value = -1
    mean_pressure_value_normalized = -1
    past_had_flow = -1
    while True:
        # Set seconds to 0 to avoid drift over multiple hours / days
        t_start = datetime.datetime.now().replace(second=0, microsecond=0)
        t_stop = t_start

        # average over averaging_period_seconds
        i_polling = 0
        # reset variables
        had_flow = 0  # avoid sticking high if we lose pressure while flowing
        polling_pressure_samples.fill(np.nan)
        polling_flow_samples = np.zeros(n_polling)
        while t_stop - t_start < datetime.timedelta(seconds=averaging_period_seconds):
            # sample pressure and flow
            pressure_value = int(chan_0.value)
            flow_value = int(had_flow)

            # save data point to array
            polling_pressure_samples[i_polling] = pressure_value
            polling_flow_samples[i_polling] = flow_value

            # display
            pressure_value_normalized = normalize_pressure_value(pressure_value)

            if display_terminal:
                line1 = f"{t_utc_str} UTC Mean Pressure: {mean_pressure_value:5d}, {mean_pressure_value_normalized:4.0%}, Flow: {past_had_flow}"
                line2 = f"i = {i_polling:3d}              Current Pressure: {pressure_value:5.0f}, {pressure_value_normalized:4.0%}, Flow: {flow_value}\r"
                if display_terminal_overwrite:
                    # https://stackoverflow.com/a/39177802
                    sys.stdout.write("\x1b[1A\x1b[2K" + line1 + "\n" + line2 + "\r")
                    sys.stdout.flush()
                else:
                    # print(line1)
                    print(line2)

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
                # n_last mean values
                if i_polling == 0 or new_connection:
                    new_connection = False
                    _data["t_est_str_n_last"] = t_est_str_n_last
                    _data[
                        "mean_pressure_value_normalized_n_last"
                    ] = mean_pressure_value_normalized_n_last
                    _data["past_had_flow_n_last"] = past_had_flow_n_last

                sio.emit("emit_data", json.dumps(_data))

            # wait polling_period_seconds between data points to average
            while datetime.datetime.now() - t_stop < datetime.timedelta(
                seconds=polling_period_seconds
            ):
                pass

            i_polling += 1
            t_stop = datetime.datetime.now()

        # take mean and save data point to csv in m_path
        t_utc_str = t_stop.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
        if display_web:
            t_est_str = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(
                datetime_fmt
            )
        mean_pressure_value = int(np.nanmean(polling_pressure_samples))
        mean_pressure_value_normalized = normalize_pressure_value(mean_pressure_value)
        past_had_flow = int(np.max(polling_flow_samples))
        new_row = [t_utc_str, mean_pressure_value, past_had_flow]

        fname_date = t_stop.astimezone(ZoneInfo("UTC")).strftime(date_fmt)
        with open(f"{m_path}/date_{fname_date}.csv", "a") as f:
            w = writer(f)
            if f.tell() == 0:
                # empty file, create header
                w.writerow(["datetime_utc", "mean_pressure_value", "had_flow"])
            w.writerow(new_row)
            f.close()

        if display_web:
            # save n_last mean values
            t_est_str_n_last.append(t_est_str)
            mean_pressure_value_normalized_n_last.append(mean_pressure_value_normalized)
            past_had_flow_n_last.append(past_had_flow)

            if n_last < len(t_est_str_n_last):
                del t_est_str_n_last[0]
                del mean_pressure_value_normalized_n_last[0]
                del past_had_flow_n_last[0]


################################################################################
# run daq loop
# if display_web:
# TODO sometimes multiple threads start, even with thread_lock!
# with thread_lock:
#    if thread is None:
# TODO get control C working on threads
#        thread = sio.start_background_task(daq_loop)
sio.start_background_task(daq_loop)

################################################################################
# serve index.html
if display_web:
    sio.run(app, port=5000, host="0.0.0.0", debug=display_web_logging_terminal)
