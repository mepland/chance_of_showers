"""Quick script to test reading pulses via GPIO."""
import time
import RPi.GPIO as GPIO

# Configuration
gpio_pin_flow = 19
WAIT_TIME = 5  # [s] Time to wait between each refresh

# Setup variables
n_pulse = 0


def fell(n):
    """Fell action."""
    global n_pulse
    n_pulse += 1


try:
    # Setup GPIO pins
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(gpio_pin_flow, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull up to 3.3V

    # Add event to detect
    GPIO.add_event_detect(gpio_pin_flow, GPIO.FALLING, fell)

    while True:
        time.sleep(WAIT_TIME)
        print(f"n_pulse = {n_pulse}")
        n_pulse = 0

except KeyboardInterrupt:  # trap a CTRL+C keyboard interrupt
    GPIO.cleanup()  # resets all GPIO ports used by this function
