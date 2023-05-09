import time
from gpiozero import Button

# Configuration
gpio_pin_flow = 19
WAIT_TIME = 5  # [s] Time to wait between each refresh

# Setup variables
had_switch_on = False


def rise(n):
    global had_switch_on
    had_switch_on = True


def fall(n):
    global had_switch_on
    had_switch_on = False


try:
    # Setup GPIO pin, pull up to 3.3V, bounce_time and hold_time are in seconds
    flow_switch = Button(pin=gpio_pin_flow, pull_up=False, bounce_time=0.1, hold_time=1)
    # flow_switch.when_pressed = rise
    flow_switch.when_held = rise
    flow_switch.when_released = fall

    while True:
        time.sleep(WAIT_TIME)
        print(f"had_switch_on = {had_switch_on}")
        # print(f"is_pressed = {flow_switch.is_pressed}")

except KeyboardInterrupt:  # trap a CTRL+C keyboard interrupt
    pass
