# Adapted from https://blog.driftking.tw/en/2019/11/Using-Raspberry-Pi-to-Control-a-PWM-Fan-and-Monitor-its-Speed/

import sys
import os
import time
import RPi.GPIO as GPIO

# Configuration
verbose = True  # print temp, PWM, RPM
FAN_PIN = 18  # BCM pin used to drive PWM fan
TACH_PIN = 23  # Fan's tachometer output pin
WAIT_TIME = 5  # [s] Time to wait between each refresh
PULSE = 2  # Noctua fans puts out two pluses per revolution
PWM_FREQ = 25  # [kHz] 25kHz for Noctua PWM control

# Configurable temperature and fan speed
MIN_TEMP = 35
MAX_TEMP = 50
FAN_LOW = 30
FAN_HIGH = 100
FAN_OFF = 0
FAN_MAX = 100

# Setup variables
t = time.time()
rpm = 0


# Get SoC's temperature
def get_SoC_temp():
    res = os.popen("vcgencmd measure_temp").readline()
    temp = float(res.replace("temp=", "").replace("'C\n", ""))

    return temp


# Set fan speed
def setFanSpeed(speed):
    fan.start(speed)

    return None


# Caculate pulse frequency and RPM
def fell(n):
    global t
    global rpm

    dt = time.time() - t
    if dt < 0.005:
        return None  # Reject spuriously short pulses

    freq = 1 / dt
    rpm = (freq / PULSE) * 60
    t = time.time()


# Handle fan speed
def handleFanSpeed():
    global rpm

    temp = get_SoC_temp()
    fan_commanded_speed = FAN_MAX

    # Turn off the fan if temperature is below MIN_TEMP
    if temp < MIN_TEMP:
        fan_commanded_speed = FAN_OFF

    # Set fan speed to MAXIMUM if the temperature is above MAX_TEMP
    elif temp > MAX_TEMP:
        fan_commanded_speed = FAN_MAX

    # Caculate dynamic fan speed
    else:
        step = (FAN_HIGH - FAN_LOW) / (MAX_TEMP - MIN_TEMP)
        fan_commanded_speed = FAN_LOW + step * round(temp - MIN_TEMP)

    # Set speed and display parameters
    setFanSpeed(fan_commanded_speed)

    if verbose:
        sys.stdout.write(
            "\x1b[2K" + f"Temp: {temp:.1f}, PWM {0.01*fan_commanded_speed:.0%}, RPM {rpm:.0f}\r"
        )
        sys.stdout.flush()

    rpm = 0

    return None


try:
    # Setup GPIO pins
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(TACH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull up to 3.3V
    GPIO.setup(FAN_PIN, GPIO.OUT, initial=GPIO.LOW)

    fan = GPIO.PWM(FAN_PIN, PWM_FREQ)
    setFanSpeed(FAN_OFF)

    # Add event to detect
    GPIO.add_event_detect(TACH_PIN, GPIO.FALLING, fell)

    # Handle fan speed every WAIT_TIME sec
    while True:
        handleFanSpeed()
        time.sleep(WAIT_TIME)

except KeyboardInterrupt:  # trap a CTRL+C keyboard interrupt
    setFanSpeed(FAN_HIGH)
    GPIO.cleanup()  # resets all GPIO ports used by this function
