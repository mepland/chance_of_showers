# python imports
import os, sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
from csv import writer

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
cs = digitalio.DigitalInOut(board.D5)
mcp = MCP.MCP3008(spi, cs, ref_voltage=5)  # 5 Volts
chan_0 = AnalogIn(mcp, MCP.P0)  # pin 0

# Setup connection to i2c display
# https://luma-oled.readthedocs.io/en/latest
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import sh1106

i2c_device = sh1106(i2c(port=1, address=0x3C), rotate=0)

# DAQ parameters
starting_time_minutes_mod = 1
averaging_period_seconds = starting_time_minutes_mod * 60
polling_period_seconds = 1

DAQ_max_value = 65472

datetime_fmt = "%Y-%m-%d %H:%M:%S"
m_path = os.path.expanduser("~/chance_of_showers/daq/tmp_data")

# other variables
os.makedirs(m_path, exist_ok=True)

n_polling = int(np.ceil(averaging_period_seconds / polling_period_seconds))

# TODo test if defining here first saves memory?
polling_samples = np.ones(n_polling)
polling_samples.fill(np.nan)


# Get CPU's temperature
def getCpuTemperature():
    res = os.popen("vcgencmd measure_temp").readline()
    # temp = float(res.replace('temp=','').replace("'C\n",''))
    # return temp
    return res


# write to OLED display
with canvas(i2c_device) as draw:
    draw.rectangle(i2c_device.bounding_box, outline="white", fill="black")
    draw.text((4, 0), "Waiting to start!", fill="white")
    draw.text((4, 12), f"CPU {getCpuTemperature()}", fill="white")

# Wait until UTC minutes is mod starting_time_minutes_mod
# Then if the script is interrupted, we can resume on the same cadence
while True:
    t_now = datetime.now()
    if (t_now.second == 0) & (t_now.minute % starting_time_minutes_mod == 0):
        t_utc_str = t_now.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
        t_est_str = t_now.astimezone(ZoneInfo("US/Eastern")).strftime(datetime_fmt)
        print(f"Started taking data at {t_utc_str} UTC, {t_est_str} EST")
        break

# start main loop
mean_value = -1
while True:
    t_start = datetime.now()
    t_stop = t_start
    # average over averaging_period_seconds
    _i = 0
    polling_samples.fill(np.nan)
    while t_stop - t_start < timedelta(seconds=averaging_period_seconds):
        # save data point to array
        polling_samples[_i] = chan_0.value

        # continually display the current value and most recent mean https://stackoverflow.com/a/39177802
        current_value_per_str = f"{polling_samples[_i]/DAQ_max_value:4.0%}"
        current_value_str = f"Current Value Pressure: {polling_samples[_i]:5.0f}, {current_value_per_str}"
        sys.stdout.write(
            "\x1b[1A\x1b[2K"
            + f"{t_utc_str} UTC Mean Value: {mean_value:5d}, {mean_value/DAQ_max_value:4.0%}\n"
        )
        sys.stdout.write(f"    i = {_i:3d} {current_value_str}\r")
        sys.stdout.flush()

        # write to OLED display
        with canvas(i2c_device) as draw:
            draw.text((4, 0), f"Pressure: {current_value_per_str}", fill="white")
            draw.text((4, 12), f"CPU {getCpuTemperature()}", fill="white")

        # wait polling_period_seconds between data points to average
        while datetime.now() - t_stop < timedelta(seconds=polling_period_seconds):
            pass
        _i += 1
        t_stop = datetime.now()

    # take mean and save data point to csv in tmp_data
    mean_value = int(np.nanmean(polling_samples))
    t_utc_str = t_stop.astimezone(ZoneInfo("UTC")).strftime(datetime_fmt)
    new_row = [t_utc_str, mean_value]

    fname = f"data_{t_stop.astimezone(ZoneInfo('UTC')).strftime('%Y-%m-%d')}"
    with open(f"{m_path}/{fname}.csv", "a") as f:
        w = writer(f)
        w.writerow(new_row)
        f.close()
# TODO exit gracefully on KeyboardInterrupt
