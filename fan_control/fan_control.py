"""PWM fan control for Raspberry Pi.

Adapted from https://blog.driftking.tw/en/2019/11/Using-Raspberry-Pi-to-Control-a-PWM-Fan-and-Monitor-its-Speed/
"""

import pathlib
import sys
import time as tm
from typing import Final

import hydra
from omegaconf import DictConfig  # noqa: TC002

# pylint: disable=no-member
from RPi import GPIO

__all__: list[str] = []


# Setup global variables
current_time = tm.time()
rpm = 0.0  # pylint: disable=invalid-name


@hydra.main(version_base=None, config_path="..", config_name="config")
def fan_control(cfg: DictConfig) -> None:
    """Run the fan control script.

    Args:
        cfg (DictConfig): Hydra configuration.
    """
    # Configuration
    # pylint: disable=invalid-name
    VERBOSE: Final = cfg["fan_control"]["verbose"]  # print temperature, PWM, and RPM

    FAN_PIN: Final = cfg["fan_control"]["fan_pin"]  # BCM pin used to drive PWM fan
    TACH_PIN: Final = cfg["fan_control"]["tach_pin"]  # Pin used to read the fan's tachometer signal

    REFRESH_PERIOD_SECONDS: Final = cfg["fan_control"][
        "refresh_period_seconds"
    ]  # Time to wait between each refresh

    PULSES_PER_REVOLUTION: Final = cfg["fan_control"][
        "pulses_per_revolution"
    ]  # Noctua fans puts out two pluses per revolution
    PWM_FREQ_KHZ: Final = cfg["fan_control"]["pwm_freq_khz"]  # 25kHz for Noctua PWM control

    # Fan curve parameters
    MIN_TEMP: Final = cfg["fan_control"]["min_temp"]
    MAX_TEMP: Final = cfg["fan_control"]["max_temp"]
    FAN_LOW: Final = cfg["fan_control"]["fan_low"]
    FAN_HIGH: Final = cfg["fan_control"]["fan_high"]
    FAN_MAX: Final = 100.0
    # pylint: enable=invalid-name

    # Lock script, avoid launching duplicates
    sys.path.append(str(pathlib.Path.cwd().parent))
    from utils.shared_functions import (  # pylint: disable=import-outside-toplevel
        get_lock,
        get_SoC_temp,
    )

    get_lock("fan_control")

    # Functions

    def set_fan_speed(speed: float) -> None:
        """Set fan speed.

        Args:
            speed (float): Desired fan speed, 0 to 100.
        """
        fan.start(speed)
        # Note that fan is defined later as GPIO.PWM(FAN_PIN, PWM_FREQ_KHZ)

    def fell(_pin: int) -> None:
        """Fell action.

        Calculate pulse frequency and RPM.

        Args:
            _pin (int): Unused, but needed to type annotation the callback of GPIO.add_event_detect().
        """
        global current_time
        global rpm

        delta_t = tm.time() - current_time
        # Reject spuriously short pulses
        if 0.005 < delta_t:
            freq = 1 / delta_t
            rpm = (freq / PULSES_PER_REVOLUTION) * 60
            current_time = tm.time()

    def handle_fan_speed() -> None:
        """Handle fan speed."""
        global rpm

        temp = get_SoC_temp()
        fan_commanded_speed = FAN_MAX

        # Set fan speed to FAN_LOW if the temperature is below MIN_TEMP
        if temp < MIN_TEMP:
            fan_commanded_speed = FAN_LOW

        # Set fan speed to FAN_MAX if the temperature is above MAX_TEMP
        elif temp > MAX_TEMP:
            fan_commanded_speed = FAN_MAX

        # Calculate dynamic fan speed
        else:
            step = (FAN_HIGH - FAN_LOW) / (MAX_TEMP - MIN_TEMP)
            fan_commanded_speed = FAN_LOW + step * round(temp - MIN_TEMP)

        # Set speed and display parameters
        set_fan_speed(fan_commanded_speed)

        if VERBOSE:
            sys.stdout.write(
                "\x1b[2K"  # noqa: ISC003
                + f"Temp: {temp:.1f}, PWM {0.01*fan_commanded_speed:.0%}, RPM {rpm:.0f}\r"
            )
            sys.stdout.flush()

        rpm = 0

    try:
        # Setup GPIO pins
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)

        GPIO.setup(TACH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull up to 3.3V
        GPIO.setup(FAN_PIN, GPIO.OUT, initial=GPIO.LOW)

        fan = GPIO.PWM(FAN_PIN, PWM_FREQ_KHZ)
        set_fan_speed(FAN_MAX)

        # Add event to detect
        GPIO.add_event_detect(TACH_PIN, GPIO.FALLING, fell)

        # Handle fan speed every REFRESH_PERIOD_SECONDS sec
        while True:
            handle_fan_speed()
            tm.sleep(REFRESH_PERIOD_SECONDS)

    except KeyboardInterrupt:  # trap a CTRL+C keyboard interrupt
        set_fan_speed(FAN_MAX)
        GPIO.cleanup()  # resets all GPIO ports used by this function


if __name__ == "__main__":
    fan_control()  # pylint: disable=no-value-for-parameter
