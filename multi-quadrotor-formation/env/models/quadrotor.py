import math
import numpy as np
import os
import yaml
from pybullet_utils import bullet_client
from typing import Optional, List

from .body import Body
from .motor import Motor
from .pid import PID


class Quadrotor:
    """
    DRONE CONTROL
        -------------------------
        |  (cw)2  0(ccw)        |
        |      \\//           x |
        |      //\\         y_| |
        | (ccw)1  3(cw)         |
        -------------------------
        using the ENU convention (East-North-Up)
        control commands are in the form of roll-pitch-yaw-thrust
    """
    NUM_MOTOR = 4

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        control_hz: int = 120,
        physics_hz: int = 240,
        np_random: Optional[np.random.RandomState] = None,
    ):
        self.p = p

        self.np_random = np.random.RandomState() if np_random is None else np_random
        self.physics_control_ratio = int(physics_hz / control_hz)
        self.physics_period = 1.0 / physics_hz
        self.control_period = 1.0 / control_hz

        model_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "env/data/"
        )
        self.drone_path = os.path.join(model_dir, "cf2x.urdf")
        self.param_path = os.path.join(model_dir, "cf2x.yaml")

        self.start_pos = start_pos
        self.start_orn = self.p.getQuaternionFromEuler(start_orn)
        self.Id = self.p.loadURDF(
            self.drone_path,
            basePosition=self.start_pos,
            baseOrientation=self.start_orn,
            useFixedBase=False,
            flags=self.p.URDF_USE_INERTIA_FROM_FILE,
        )

        self.state: np.ndarray
        self.aux_state: np.ndarray
        self.setpoint: np.ndarray

        self.motors: List[Motor] = []
        self.PID: PID

        # motor mapping from command to individual motors
        self.motor_map = np.array(
            [
                [-1.0, -1.0, -1.0, +1.0],
                [+1.0, +1.0, -1.0, +1.0],
                [+1.0, -1.0, +1.0, +1.0],
                [-1.0, +1.0, +1.0, +1.0],
            ]
        )

        # All the params for the drone
        with open(self.param_path, "rb") as f:
            # load all params from yaml
            all_params = yaml.safe_load(f)
            motor_params = all_params["motor_params"]
            drag_params = all_params["drag_params"]
            ctrl_params = all_params["control_params"]

            tau = motor_params["tau"]
            thrust_coef = motor_params["thrust_coef"]
            torque_coef = motor_params["torque_coef"]
            noise_ratio = motor_params["noise_ratio"]

            total_thrust = motor_params["total_thrust"]
            max_rpm = np.sqrt(total_thrust / thrust_coef)

            torque_coef = np.array(
                [
                    -motor_params["torque_coef"],
                    -motor_params["torque_coef"],
                    +motor_params["torque_coef"],
                    +motor_params["torque_coef"],
                ]
            )
            
            self.motors.append(Motor(
                max_rpm=max_rpm,
                thrust_coef=thrust_coef,
                torque_coef=torque_coef,
                physics_hz=physics_hz,
                np_random=self.np_random,
                tau=tau,
                noise_ratio=noise_ratio,
            ))

            # pseudo drag coef
            self.drag_coef_pqr = drag_params["drag_coef_pqr"]

            # simulate the drag on the main body
            self.body = Body(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                body_ids=np.array([4]),
                drag_coefs=np.array([[drag_params["drag_coef_xyz"]] * 3]),
                normal_areas=np.array([[drag_params["drag_area_xyz"]] * 3]),
            )

            # input: angular velocity command
            # outputs: normalized body torque command
            self.Kp_ang_vel = np.array(ctrl_params["ang_vel"]["kp"])
            self.Ki_ang_vel = np.array(ctrl_params["ang_vel"]["ki"])
            self.Kd_ang_vel = np.array(ctrl_params["ang_vel"]["kd"])
            self.lim_ang_vel = np.array(ctrl_params["ang_vel"]["lim"])

    def _disable_artificial_damping(self):
        for idx in range(-1, self.p.getNumJoints(self.Id)):
            self.p.changeDynamics(self.Id, idx, linearDamping=0.0, angularDamping=0.0)

    def reset(self):
        """Resets the vehicle to the initial state."""
        self.set_mode(0)
        self.setpoint = np.zeros(4)
        self.pwm = np.zeros(4)

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self._disable_artificial_damping()
        self.body.reset()
        self.motors.reset()

    def set_mode(self, mode: int):
        self.mode = mode
        self.setpoint = np.array([0.0, 0.0, 0.0, -1.0])

        ang_vel_PID = PID(
            self.Kp_ang_vel,
            self.Ki_ang_vel,
            self.Kd_ang_vel,
            self.lim_ang_vel,
            self.control_period,
        )
        self.PID = ang_vel_PID
        self.PID.reset()

    def update_control(self):
        """Runs through controllers."""
        a_output = self.setpoint[:3].copy()
        z_output = self.setpoint[-1].copy()
        
        mode = self.mode
        if mode == -1:
            self.pwm = np.array([*a_output, z_output])
            return

        # base level controllers
        a_output = self.PID.step(self.state[0], a_output)

        # height controllers
        z_output = np.clip(z_output, 0.0, 1.0).flatten()

        # mix the commands according to motor mix
        cmd = np.array([*a_output, *z_output])
        self.pwm = np.matmul(self.motor_map, cmd)

        high, low = np.max(self.pwm), np.min(self.pwm)
        if high != low:
            pwm_max, pwm_min = min(high, 1.0), max(low, 0.05)
            add = (pwm_min - low) / (pwm_max - low) * (pwm_max - self.pwm)
            sub = (high - pwm_max) / (high - pwm_min) * (self.pwm - pwm_min)
            self.pwm += add - sub
        self.pwm = np.clip(self.pwm, 0.05, 1.0)

    def update_physics(self):
        """Updates the physics of the vehicle."""
        self.body.physics_update()

        for index, motor in enumerate(self.motors):
            thrust, torque = motor.run(self.pwm)
            self.p.applyExternalForce(
                self.Id, index, thrust, [0.0, 0.0, 0.0], self.p.LINK_FRAME
            )
            self.p.applyExternalTorque(self.Id, index, torque, self.p.LINK_FRAME)

        # simulate rotational damping
        drag_pqr = -self.drag_coef_pqr * (np.array(self.state[0]) ** 2)

        # warning, the physics is funky for bounces
        if len(self.p.getContactPoints()) == 0:
            self.p.applyExternalTorque(self.Id, -1, drag_pqr, self.p.LINK_FRAME)

    def update_state(self):
        """Updates the current state of the UAV.

        This includes: ang_vel, ang_pos, lin_vel, lin_pos.
        """
        lin_pos, ang_pos = self.p.getBasePositionAndOrientation(self.Id)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.Id)

        # express vels in local frame
        rotation = np.array(self.p.getMatrixFromQuaternion(ang_pos)).reshape(3, 3).T
        lin_vel = np.matmul(rotation, lin_vel)
        ang_vel = np.matmul(rotation, ang_vel)

        # ang_pos in euler form
        ang_pos = self.p.getEulerFromQuaternion(ang_pos)

        # update the main body
        self.body.state_update(rotation)

        # create the state
        self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

        # update auxiliary information
        self.aux_state = self.motors.get_states()

    def update_last(self):
        """Updates things only at the end of `Aviary.step()`."""
        pass
