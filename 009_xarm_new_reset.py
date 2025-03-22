from xarm6_interface.arm_rw import XArm6RealWorld


if __name__ == "__main__":
    
    xarm = XArm6RealWorld()
    
    import time 
    while True:
        time.sleep(1)