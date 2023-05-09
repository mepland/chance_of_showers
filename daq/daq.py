# python imports
import os
import sys
import signal
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pause
import numpy as np
from csv import writer


# catch ctrl+c and kill, shut down gracefully https://stackoverflow.com/a/38665760
def signal_handler(signal, frame):
    print("\nDAQ exiting gracefully")

    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

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

# Setup connection for reading the flow sensor as a switch
# https://gpiozero.readthedocs.io/en/stable/api_input.html?highlight=Button#gpiozero.Button
# Note, I would prefer to read the pulses per minute with RPi.GPIO as in fan_control.py,
# but my flow sensor only produces a constant Vcc while flow is occurring, no pulses.
from gpiozero import Button

had_flow = 0


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


# Setup connection to i2c display
# https://luma-oled.readthedocs.io/en/latest
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
    with canvas(i2c_device) as draw:
        if bounding_box:
            draw.rectangle(i2c_device.bounding_box, outline="white", fill="black")
        for i_line, line in enumerate(lines):
            draw.text(
                (lpad, vpad + i_line * line_height), line, fill="white", font=oled_font
            )


# DAQ parameters
starting_time_minutes_mod = 1
averaging_period_seconds = starting_time_minutes_mod * 60
polling_period_seconds = 1
# DAQ_max_value = 65472

date_fmt = "%Y-%m-%d"
time_fmt = "%H:%M:%S"
datetime_fmt = f"{date_fmt} {time_fmt}"

m_path = os.path.expanduser("~/chance_of_showers/daq/raw_data")

# pressure variables and functions
observed_pressure_min = 5000
observed_pressure_max = 20000


def normalize_pressure_value(pressure_value):
    return (pressure_value - observed_pressure_min) / (
        observed_pressure_max - observed_pressure_min
    )


# other variables
os.makedirs(m_path, exist_ok=True)

n_polling = int(np.ceil(averaging_period_seconds / polling_period_seconds))

# TODo test if defining here first saves memory?
polling_pressure_samples = np.empty(n_polling)
polling_pressure_samples.fill(np.nan)
polling_flow_samples = np.zeros(n_polling)


# Get CPU's temperature
def get_SoC_temp():
    res = os.popen("vcgencmd measure_temp").readline()
    temp = res.replace("temp=", "").replace("'C\n", " C")
    return f"SoC: {temp}"


# Wait until UTC minutes is mod starting_time_minutes_mod
# Then if the script is interrupted, we can resume on the same cadence

t_start = datetime.now()
t_start_minute = (
    t_start.minute
    - (t_start.minute % starting_time_minutes_mod)
    + starting_time_minutes_mod
)
t_start = t_start.replace(minute=t_start_minute, second=0, microsecond=0)
t_utc_str = t_start.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
t_est_str = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(datetime_fmt)
t_est_str_short = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(time_fmt)
print(f"Will start taking data at {t_utc_str} UTC, {t_est_str} EST")

# write to OLED display
paint_oled([f"Will start at:", t_est_str_short, get_SoC_temp()], bounding_box=True)

pause.until(t_start)

t_start = datetime.now()
t_utc_str = t_start.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
t_est_str = t_start.astimezone(ZoneInfo("US/Eastern")).strftime(datetime_fmt)
print(f"\nStarted taking data at {t_utc_str} UTC, {t_est_str} EST\n\n")

# start main loop
mean_pressure_value = -1
past_had_flow = -1
while True:
    # Set seconds to 0 to avoid drift over multiple hours / days
    t_start = datetime.now().replace(second=0, microsecond=0)
    t_stop = t_start

    # average over averaging_period_seconds
    _i = 0
    polling_pressure_samples.fill(np.nan)
    polling_flow_samples = np.zeros(n_polling)
    while t_stop - t_start < timedelta(seconds=averaging_period_seconds):
        # save data point to array
        pressure_value = chan_0.value
        polling_pressure_samples[_i] = pressure_value

        polling_flow_samples[_i] = had_flow

        # continually display the current value and most recent mean https://stackoverflow.com/a/39177802
        current_pressure_value_percent_str = (
            f"{normalize_pressure_value(pressure_value):4.0%}"
        )
        current_pressure_value_str = f"Current Pressure: {pressure_value:5.0f}, {current_pressure_value_percent_str}"
        sys.stdout.write(
            "\x1b[1A\x1b[2K"
            + f"{t_utc_str} UTC Mean Pressure: {mean_pressure_value:5d}, {normalize_pressure_value(mean_pressure_value):4.0%}, Flow: {past_had_flow}\n"
        )
        sys.stdout.write(
            f"i = {_i:3d}              {current_pressure_value_str}, Flow: {had_flow}\r"
        )
        sys.stdout.flush()

        # write to OLED display
        paint_oled(
            [
                f"Pressure: {current_pressure_value_percent_str}",
                f"Pressure: {pressure_value:5.0f}",
                f"Flow: {had_flow}",
                get_SoC_temp(),
            ]
        )

        # wait polling_period_seconds between data points to average
        while datetime.now() - t_stop < timedelta(seconds=polling_period_seconds):
            pass

        _i += 1
        t_stop = datetime.now()

    # take mean and save data point to csv in tmp_data
    t_utc_str = t_stop.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
    mean_pressure_value = int(np.nanmean(polling_pressure_samples))
    past_had_flow = int(np.max(polling_flow_samples))
    new_row = [t_utc_str, mean_pressure_value, past_had_flow]

    fname_date = t_stop.astimezone(ZoneInfo("UTC")).strftime(date_fmt)
    with open(f"{m_path}/date_{fname_date}.csv", "a") as f:
        w = writer(f)
        if f.tell() == 0:
            # empty file, create header
            w.writerow(["time_utc", "mean_pressure_value", "had_flow"])
        w.writerow(new_row)
        f.close()
