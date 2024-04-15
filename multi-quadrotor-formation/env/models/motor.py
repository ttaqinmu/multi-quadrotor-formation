import numpy as np
from typing import Optional, Tuple


class Motor:
    def __init__(
        self,
        max_rpm: float,
        thrust_coef: float,
        torque_coef: float,
        physics_hz: int = 240,
        np_random: Optional[np.random.RandomState] = None,
        tau: float = 0.01,
        noise_ratio: float = 0.02,
    ):
        self.physics_period = 1 / physics_hz

        self.throttle: float = 0
        self.rpm: float = 0

        self.max_rpm = max_rpm
        self.thrust_coef = thrust_coef
        self.torque_coef = torque_coef
        self.tau = tau
        self.noise_ratio = noise_ratio

        self.np_random = np_random if np_random else np.random.RandomState()

    def reset(self):
        self.throttle = 0
        self.rpm = 0

    def get_states(self) -> Tuple[float, float]:
        """
        Return: throtlle, rpm
        """
        return self.throttle, self.rpm

    def run(self, pwm: float) -> Tuple[float, float]:
        """
        Args: pwm -> -1<= val >= 1
        Return: thrust, torque
        """

        pwm = -1.0 if pwm < -1 else pwm
        pwm = 1.0 if pwm > 1 else pwm

        self.throttle += (self.physics_period / self.tau) * (pwm - self.throttle)
        self.throttle += self.np_random.randn() * self.throttle * self.noise_ratio

        self.rpm = self.throttle * self.max_rpm

        # rpm to thrust and torque
        rpm_const = (self.rpm**2) * np.sign(self.rpm)
        thrust = rpm_const * self.thrust_coef
        torque = rpm_const * self.torque_coef

        return thrust, torque
