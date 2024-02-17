"""Quick script to test reading pulses via GPIO."""

import time

# pylint: disable=no-member
from RPi import GPIO

__all__ = []


# Configuration
GPIO_PIN = 19
WAIT_TIME = 5  # [s] Time to wait between each refresh

# Setup variables
n_pulse = 0  # pylint: disable=invalid-name


def fell(pin: int) -> None:  # noqa: U100 # pylint: disable=unused-argument
    """Fell action.

    Args:
        pin: Unused, but needed to type annotation the callback of GPIO.add_event_detect().
    """
    global n_pulse
    n_pulse += 1


try:
    # Setup GPIO pins
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(GPIO_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull up to 3.3V

    GPIO.add_event_detect(GPIO_PIN, GPIO.FALLING, fell)

    while True:
        time.sleep(WAIT_TIME)
        print(f"n_pulse = {n_pulse}")
        n_pulse = 0  # pylint: disable=invalid-name

except KeyboardInterrupt:  # trap a CTRL+C keyboard interrupt
    GPIO.cleanup()  # resets all GPIO ports used by this function
