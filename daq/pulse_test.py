"""Quick script to test reading pulses via GPIO."""
import time

# pylint: disable=no-member
from RPi import GPIO

# Configuration
GPIO_PIN_FLOW = 19
WAIT_TIME = 5  # [s] Time to wait between each refresh

# Setup variables
n_pulse = 0  # pylint: disable=C0103


def fell() -> None:
    """Fell action."""
    global n_pulse
    n_pulse += 1


try:
    # Setup GPIO pins
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(GPIO_PIN_FLOW, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull up to 3.3V

    # Add event to detect
    GPIO.add_event_detect(GPIO_PIN_FLOW, GPIO.FALLING, fell)

    while True:
        time.sleep(WAIT_TIME)
        print(f"n_pulse = {n_pulse}")
        n_pulse = 0  # pylint: disable=C0103

except KeyboardInterrupt:  # trap a CTRL+C keyboard interrupt
    GPIO.cleanup()  # resets all GPIO ports used by this function
