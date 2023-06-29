import asyncio
import cv2
import threading
import base64
import time
import websockets
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from queue import Queue

from ultralytics import YOLO

# 定义全局变量

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')
# rtsp_path = "videos/example.mp4"
# rtsp_path ="https://open.ys7.com/v3/openlive/K10170061_1_1.m3u8?expire=1717206565&id=586491899410382848&t=e862caf9959031a4c7b74c33388f2dcbf7bf5f358c5ed4022861c20aa1766100&ev=100"
rtsp_path = 0
speed = 0.1
frame_queue = asyncio.Queue()
flag = False


# 在OpenCV图像上添加中文文字
def put_chinese_text(image, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv2


# 视频获取和处理
def vedioCapture_thread2(n):
    global camera1

    camera1 = cv2.VideoCapture(rtsp_path)

    while True:
        _, img_bgr = camera1.read()

        if img_bgr is None:
            camera1 = cv2.VideoCapture(rtsp_path)
            print('丢失帧')
        else:
            # Run YOLOv8 inference on the frame
            results = model.predict(img_bgr, conf=0.3, device='cuda:0', classes=0)
            annotated_frame = results[0].plot(conf=True, line_width=1, font_size=2)

            # 计算检测到的人数
            person_count = len(results[0])

            # 添加人数中文文本
            text = f"人数: {person_count}"
            position = (10, 10)
            font_path = "microsoft-yahei/chinese.msyh.ttf"
            font_size = 50
            color = (0, 255, 255)
            annotated_frame = put_chinese_text(annotated_frame, text, position, font_path, font_size, color)

            # 添加帧数（FPS）中文文本
            fps_text = f"帧数: {fps:.2f}"
            fps_position = (10, 70)
            annotated_frame = put_chinese_text(annotated_frame, fps_text, fps_position, font_path, font_size, color)

            # 将帧数据放入队列
            frame_queue.put(annotated_frame)


def vedioSend_thread1(n):
    global flag

    time.sleep(3)
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # 将帧数据转换为base64编码
            if frame_queue.empty():
                continue

            # 从队列中获取帧数据
            frame = frame_queue.get()

            if frame is not None:  # Check if the frame is not empty
                image = cv2.imencode('.jpg', frame)[1]
                base64_data = base64.b64encode(image)
                s = base64_data.decode()
                base64img = 'data:image/jpeg;base64,{}'.format(s)
                flag = True
            time.sleep(speed)

async def sendImg(websocket, path):
    global flag
    while True:
        if flag:
            try:
                await websocket.send(base64img)
            except websockets.exceptions.ConnectionClosedError as e:
                print("connection closed error")
                break
            except Exception as e:
                print(e)
                break
            flag = False

async def handle_command(command):
    if command == "switch_to_camera":
        # 切换到摄像头进行推理
        # 进行相关操作
        print("Switching to camera for inference")
    else:
        # 其他指令的处理逻辑
        print('Received command: ', command)

async def receive_commands(websocket, path):
    async for message in websocket:
        if message == "switch_to_camera":
            await handle_command(message)
        else:
            print('Received message: ', message)
        await sendImg(websocket, path)

async def main():
    thread1 = threading.Thread(target=vedioSend_thread1, args=(1,))
    thread1.start()
    thread2 = threading.Thread(target=vedioCapture_thread2, args=(1,))
    thread2.start()

    async with websockets.serve(receive_commands, "localhost", 8767):
        await asyncio.Future()  # run forever

    thread1.join()
    thread2.join()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Shutting down gracefully...")

