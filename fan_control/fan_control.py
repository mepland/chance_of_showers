"""PWM fan control for Raspberry Pi.

Adapted from https://blog.driftking.tw/en/2019/11/Using-Raspberry-Pi-to-Control-a-PWM-Fan-and-Monitor-its-Speed/
"""

import os
import sys
import time

# pylint: disable=no-member
from RPi import GPIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.shared_functions import (  # noqa: E402 # pylint: disable=import-error
    get_lock,
    get_SoC_temp,
)

get_lock("fan_control")


# Configuration
verbose = True  # print temp, PWM, RPM pylint: disable=C0103
FAN_PIN = 18  # BCM pin used to drive PWM fan
TACH_PIN = 23  # Fan's tachometer output pin
WAIT_TIME = 5.0  # [s] Time to wait between each refresh
PULSE = 2  # Noctua fans puts out two pluses per revolution
PWM_FREQ = 25.0  # [kHz] 25kHz for Noctua PWM control

# Configurable temperature and fan speed
MIN_TEMP = 35.0
MAX_TEMP = 50.0
FAN_LOW = 30.0
FAN_HIGH = 100.0
FAN_OFF = 0.0
FAN_MAX = 100.0

# Setup variables
current_time = time.time()
rpm = 0.0  # pylint: disable=C0103


def set_fan_speed(speed: float) -> None:
    """Set fan speed.

    Args:
        speed: Desired fan speed, 0 to 100.
    """
    fan.start(speed)


def fell(pin: int) -> None:  # noqa: U100 # pylint: disable=unused-argument
    """Fell action.

    Caculate pulse frequency and RPM.

    Args:
        pin: Unused, but needed to type annotation the callback of GPIO.add_event_detect().
    """
    global current_time
    global rpm

    delta_t = time.time() - current_time
    # Reject spuriously short pulses
    if 0.005 < delta_t:
        freq = 1 / delta_t
        rpm = (freq / PULSE) * 60
        current_time = time.time()


def handle_fan_speed() -> None:
    """Handle fan speed."""
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
    set_fan_speed(fan_commanded_speed)

    if verbose:
        sys.stdout.write(
            "\x1b[2K" + f"Temp: {temp:.1f}, PWM {0.01*fan_commanded_speed:.0%}, RPM {rpm:.0f}\r"
        )
        sys.stdout.flush()

    rpm = 0


try:
    # Setup GPIO pins
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(TACH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull up to 3.3V
    GPIO.setup(FAN_PIN, GPIO.OUT, initial=GPIO.LOW)

    fan = GPIO.PWM(FAN_PIN, PWM_FREQ)
    set_fan_speed(FAN_OFF)

    # Add event to detect
    GPIO.add_event_detect(TACH_PIN, GPIO.FALLING, fell)

    # Handle fan speed every WAIT_TIME sec
    while True:
        handle_fan_speed()
        time.sleep(WAIT_TIME)

except KeyboardInterrupt:  # trap a CTRL+C keyboard interrupt
    set_fan_speed(FAN_HIGH)
    GPIO.cleanup()  # resets all GPIO ports used by this function
