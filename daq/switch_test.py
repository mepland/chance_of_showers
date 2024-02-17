"""Quick script to test reading switch via GPIO."""

import time

from gpiozero import Button

__all__ = []


# Configuration
GPIO_PIN = 19
WAIT_TIME = 5  # [s] Time to wait between each refresh

# Setup variables
had_switch_on = False  # pylint: disable=invalid-name


def rise() -> None:
    """Switch rise action."""
    global had_switch_on
    had_switch_on = True


def fall() -> None:
    """Switch fall action."""
    global had_switch_on
    had_switch_on = False


try:
    # Setup GPIO pin, pull up to 3.3V, bounce_time and hold_time are in seconds
    switch = Button(pin=GPIO_PIN, pull_up=False, bounce_time=0.1, hold_time=1)
    switch.when_held = rise
    switch.when_released = fall

    while True:
        time.sleep(WAIT_TIME)
        print(f"had_switch_on = {had_switch_on}")

except KeyboardInterrupt:  # trap a CTRL+C keyboard interrupt
    pass
