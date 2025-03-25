import pyrealsense2 as rs
import numpy as np
import os

# 创建 RealSense 管道
pipeline = rs.pipeline()

# 创建配置对象
config = rs.config()

# 启用颜色流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 10)

# 启动管道
profile = pipeline.start(config)

# 获取颜色传感器的内参
color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

# 构建 3x3 内参矩阵
K = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
])

# 打印内参矩阵
print("Color Intrinsics Matrix (K):")
print(K)

# 获取相机的串口号
device = profile.get_device()
serial_number = device.get_info(rs.camera_info.serial_number)
print(f"Camera serial number: {serial_number}")

# 保存为 .npy 文件
current_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_path, f"data/camera/{serial_number}/K.npy")
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建目录
np.save(save_path, K)
print(f"Saved intrinsic matrix to {save_path}")

# 停止管道
pipeline.stop()