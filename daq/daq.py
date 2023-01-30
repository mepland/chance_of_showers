import time
import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

# Setup connection
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(board.D5)
mcp = MCP.MCP3008(spi, cs)
chan = AnalogIn(mcp, MCP.P0)

# DAQ parameters
averaging_period_seconds = 5*60
polling_period_seconds = 5
verbosity = 2

n_polling = int(np.ceil(averaging_period_seconds/polling_period_seconds))

# test if defining here first saves memory?
polling_samples = np.ones(n_polling).fill(np.nan)

# start main loop
while True:
    # TODO wait until UTC seconds is mod averaging_period_seconds
    start = time.time()
    stop = time.time()

    # average over averaging_period_seconds
    _i = 0
    polling_samples = np.ones(n_polling).fill(np.nan)
    while (stop-start) < averaging_period_seconds:
        print (stop-start) # TODO delete
        # wait polling_period_seconds between data points to average
        time.sleep(polling_period_seconds)
        # save data point to array
        if 2 <= verbosity:
            print(f'Raw ADC Value: {chan.value:5}')
            print(f'ADC Voltage: {chan.voltage:5} V')
        polling_samples[_i] = chan.voltage
        _i += 1
        stop = time.time()
    # take average
    avg = np.nanmean(polling_samples)
    if 1 <= verbose:
        print(f'Average Voltage: {avg:5} V')

    # save data point to csv in tmp_data
