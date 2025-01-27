import zmq
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
## init xArm
import os
import sys
import time
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Slerp

context = zmq.Context()
# talk to server windows machine
print("Connecting to windows server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://172.25.104.11:5555")

i = 0
# EE_np_init = np.array([0,0,0])
EE_left_init = None
EE_right_init = None

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

ip_left = "192.168.1.232"
ip_right = "192.168.1.208"

arm_left = XArmAPI(ip_left)
arm_right = XArmAPI(ip_right)

arm_left.motion_enable(enable=True)
arm_right.motion_enable(enable=True)

arm_left.set_mode(0)
arm_right.set_mode(0)

arm_left.set_state(state=0)
arm_right.set_state(state=0)

arm_left.set_position(x=200, y=0, z=350, roll=-180, pitch=0, yaw=0, speed=100, is_radian=False, wait=True)
arm_right.set_position(x=200, y=0, z=350, roll=-180, pitch=0, yaw=0, speed=100, is_radian=False, wait=True)
input()
xarm_left_init_pos = [200, 0, 350]
xarm_right_init_pos = [200, 0, 350]

# set mode: cartesian online trajectory planning mode
# the running command will be interrupted when the next command is received
arm_left.set_mode(7)
arm_right.set_mode(7)

arm_left.set_state(0)
arm_right.set_state(0)

time.sleep(1)

speed = 100
scale = 1.0


# init gripper 
arm_left.set_gripper_mode(0)
arm_right.set_gripper_mode(0)

arm_left.set_gripper_enable(True)
arm_right.set_gripper_enable(True)

arm_left.set_gripper_speed(5000)
arm_right.set_gripper_speed(5000)

arm_left.set_gripper_position(850, wait=False)
arm_right.set_gripper_position(850, wait=True)

xarm_left_target_position = np.zeros(3)
xarm_left_target_position[0] = 430
xarm_left_target_position[1] = 80
xarm_left_target_position[2] = 140 # then 80
#-180,0,-90

xarm_right_target_position = np.zeros(3)
xarm_right_target_position[0] = 430
xarm_right_target_position[1] = -80
xarm_right_target_position[2] = 140
#

#step0: pre pose
arm_left.set_position(x=xarm_left_target_position[0], y=xarm_left_target_position[1], z=xarm_left_target_position[2], roll=-180, pitch=0, yaw=-90, speed=100, is_radian=False, wait=False)
arm_right.set_position(x=xarm_right_target_position[0], y=xarm_right_target_position[1], z=xarm_right_target_position[2], roll=-180, pitch=0, yaw=-90, speed=100, is_radian=False, wait=False)

print('here xarm_left_target_position:',xarm_left_target_position)
print('here xarm_right_target_position:',xarm_right_target_position)
time.sleep(1)
#step1: trg pose
xarm_left_target_position[2]=80
xarm_right_target_position[2]=80

arm_left.set_mode(0)
arm_right.set_mode(0)

arm_left.set_state(state=0)
arm_right.set_state(state=0)
print('xarm_left_target_position:',xarm_left_target_position)
print('xarm_right_target_position:',xarm_right_target_position)
arm_left.set_position(x=xarm_left_target_position[0], y=xarm_left_target_position[1], z=xarm_left_target_position[2], roll=-180, pitch=0, yaw=-90, speed=100, is_radian=False, wait=False)
arm_right.set_position(x=xarm_right_target_position[0], y=xarm_right_target_position[1], z=xarm_right_target_position[2], roll=-180, pitch=0, yaw=-90, speed=100, is_radian=False, wait=False)
time.sleep(1)
#close gripper

arm_left.set_gripper_position(210, wait=False)
arm_right.set_gripper_position(210, wait=True)
time.sleep(1)

#step2: back to pre pose
xarm_left_target_position[2]=180
xarm_right_target_position[2]=180

arm_left.set_position(x=xarm_left_target_position[0], y=xarm_left_target_position[1], z=xarm_left_target_position[2], roll=-180, pitch=0, yaw=-90, speed=100, is_radian=False, wait=False)
arm_right.set_position(x=xarm_right_target_position[0], y=xarm_right_target_position[1], z=xarm_right_target_position[2], roll=-180, pitch=0, yaw=-90, speed=100, is_radian=False, wait=False)
time.sleep(1)

# arm_left.set_gripper_position(210, wait=False)
# arm_right.set_gripper_position(210, wait=True)
# time.sleep(1)
#-180,0,-90

# # set cartesian velocity control mode
# arm_left.set_mode(5)
# arm_left.set_state(0)
# arm_right.set_mode(5)
# arm_right.set_state(0)
# time.sleep(1)

# while True:
    
#     arm_left.vc_set_cartesian_velocity([0, 100, 0, 0, 0, 0])
#     arm_right.vc_set_cartesian_velocity([0, -100, 0, 0, 0, 0])
#     time.sleep(1)
#     arm_left.vc_set_cartesian_velocity([0, 0, 100, 0, 0, 0])
#     arm_right.vc_set_cartesian_velocity([0, 0, 100, 0, 0, 0])
#     time.sleep(1)
#     arm_left.vc_set_cartesian_velocity([0, -100, 0, 0, 0, 0])
#     arm_right.vc_set_cartesian_velocity([0, 100, 0, 0, 0, 0])
#     time.sleep(1)
#     arm_left.vc_set_cartesian_velocity([0, 0, -100, 0, 0, 0])
#     arm_right.vc_set_cartesian_velocity([0, 0, -100, 0, 0, 0])
#     time.sleep(1)

# while True:
#     socket.send(b"request")
#     print(f"request sent")
#     # Get from server
#     pose24_byte = socket.recv()
#     pose24_np = np.frombuffer(pose24_byte, dtype=np.float64)
#     pose24_lwrist = np.reshape(pose24_np[0:16], [4, 4])
#     pose24_rwrist = np.reshape(pose24_np[16:], [4, 4])

#     EE_left_position = pose24_lwrist[:3, 3]
#     EE_left_rot_np = pose24_lwrist[:3, :3]

#     EE_right_position = pose24_rwrist[:3, 3]
#     EE_right_rot_np = pose24_rwrist[:3, :3]

#     if i == 0:
#         EE_left_init = EE_left_position
#         EE_right_init = EE_right_position
#         i = 1
#         continue
#     else:
#         EE_left_rel = EE_left_position - EE_left_init
#         EE_right_rel = EE_right_position - EE_right_init

#         EE_left_rot = EE_left_rot_np.copy()
#         EE_right_rot = EE_right_rot_np.copy()

#         EE_left_euler = R.from_matrix(EE_left_rot).as_euler('xyz', degrees=True)
#         EE_right_euler = R.from_matrix(EE_right_rot).as_euler('xyz', degrees=True)

#     xarm_left_target_position = np.zeros(3)
#     xarm_left_target_position[0] = xarm_left_init_pos[0] - scale * EE_left_rel[1] * 1000
#     xarm_left_target_position[1] = xarm_left_init_pos[1] + scale * EE_left_rel[0] * 1000
#     xarm_left_target_position[2] = xarm_left_init_pos[2] + scale * EE_left_rel[2] * 1000

#     xarm_right_target_position = np.zeros(3)
#     xarm_right_target_position[0] = xarm_right_init_pos[0] - scale * EE_right_rel[1] * 1000
#     xarm_right_target_position[1] = xarm_right_init_pos[1] + scale * EE_right_rel[0] * 1000
#     xarm_right_target_position[2] = xarm_right_init_pos[2] + scale * EE_right_rel[2] * 1000

#     arm_left.set_position(x=xarm_left_target_position[0],
#                           y=xarm_left_target_position[1],
#                           z=xarm_left_target_position[2],
#                           roll=EE_left_euler[0] + 90,
#                           pitch=EE_left_euler[1],
#                           yaw=EE_left_euler[2] + 60,
#                           speed=speed,
#                           wait=False)

#     arm_right.set_position(x=xarm_right_target_position[0], y=xarm_right_target_position[1], z=xarm_right_target_position[2],
#                       roll=EE_right_euler[0]+90, pitch=EE_right_euler[1], yaw=EE_right_euler[2]-90,
#                       speed=speed, wait=False)
