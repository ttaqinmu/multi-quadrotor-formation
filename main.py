from PyFlyt.core import Aviary
import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import math

start_pos = np.array([[0.0, 0.0, 0.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

env = Aviary(start_pos=start_pos, start_orn=start_orn, render=False, drone_type="quadx")

env.set_mode(7)
env.set_setpoint(0, np.array([1, 3, 5]))

NUM_POINTS = 100
time_offsets = np.random.rand(NUM_POINTS)

rr.init("rerun_example_my_data", spawn=True)
rr.set_time("stable_time", duration=0)
asset = rr.Asset3D(path="drone.obj")
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
rr.log("world/drone", asset)

for i in range(1000):
    time = i * 0.01
    times = np.repeat(time, NUM_POINTS) + time_offsets
    env.step()
    pos = env.state(0)[3,:]
    rot = env.state(0)[1,:]
    rr.set_time("stable_time", duration=time)
    rr.log("world/drone", rr.Transform3D(quaternion=[1, 0, 0, 1], scale=0.1, translation=pos))

# env.set_setpoint(0, np.array([0, 0, 5]))
#
# for i in range(1000):
#     time = i * 0.01
#     times = np.repeat(time, NUM_POINTS) + time_offsets
#     env.step()
#     pos = env.state(0)[3,:]
#     # print(pos)
#     rr.set_time("stable_time", duration=time)
#     rr.log("sensor/points", asset, rr.Points3D(pos))
