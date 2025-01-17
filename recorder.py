import pyrealsense2 as rs
import cv2
import numpy as np
from datetime import datetime

# 指定 RealSense 摄像头序列号
serial_number = "317222073552"  # 替换为你的设备序列号

# 配置 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 绑定指定的序列号
config.enable_device(serial_number)

# 配置 RGB 视频流（1280x720 分辨率，30 帧率）
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 开启管道
pipeline.start(config)

# 动态生成文件名（根据当前时间）
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"realsense_video_{current_time}.mp4"

# 设置视频保存参数
fps = 30
frame_width = 1280  # 与 RGB 流一致
frame_height = 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码格式
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# 初始状态
is_paused = False  # 用于暂停
is_recording = True  # 标记是否在录制

try:
    print(f"开始录制，视频将保存为: {output_file}")
    print("按 'p' 键暂停/恢复录制，按 'q' 键停止并退出...")
    while True:
        if not is_paused:
            # 获取帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # 将 RGB 帧转换为 Numpy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 显示实时视频
            cv2.imshow('RealSense Video', color_image)

            if is_recording:
                # 保存到 MP4 文件
                video_writer.write(color_image)

        # 键盘控制逻辑
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):  # 暂停或恢复
            is_paused = not is_paused
            print("录制已暂停" if is_paused else "录制已恢复")
        elif key == ord('q'):  # 退出程序
            break

finally:
    # 停止管道并释放资源
    pipeline.stop()
    video_writer.release()
    cv2.destroyAllWindows()
    print("视频已保存为:", output_file)
