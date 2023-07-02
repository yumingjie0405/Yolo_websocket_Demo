import threading
import base64
import cv2
import asyncio
import time
import numpy as np
import websockets
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import supervision as sv

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')
# 摄像头编号
'''
# Open the video file
# cap = cv2.VideoCapture("https://open.ys7.com/v3/openlive/K10170061_1_1.m3u8?expire=1717206565&id=586491899410382848&t=e862caf9959031a4c7b74c33388f2dcbf7bf5f358c5ed4022861c20aa1766100&ev=100")
# cap = cv2.VideoCapture("MOT16-03.mp4")
'''
# 设置RTSP路径或摄像头索引
# rtsp_path = "videos/example.mp4"
# rtsp_path ="https://open.ys7.com/v3/openlive/K10170061_1_1.m3u8?expire=1717206565&id=586491899410382848&t=e862caf9959031a4c7b74c33388f2dcbf7bf5f358c5ed4022861c20aa1766100&ev=100"
rtsp_path = 0
camera1 = None
frame = None
base64img = ''
flag = True
speed = 0.001  # 视频帧率 表示多少秒一帧


# 在OpenCV图像上添加中文文字
def put_chinese_text(image, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv2


# 一个线程 视频获取和处理

def vedioCapture_thread2(n):
    global camera1
    camera1 = cv2.VideoCapture(rtsp_path)
    global frame

    # Adding FPS calculation
    fps = 0
    last_time = time.time()

    while True:
        _, img_bgr = camera1.read()

        if img_bgr is None:
            camera1 = cv2.VideoCapture(rtsp_path)
            print('丢失帧')
        else:
            # Update FPS calculation
            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time

            # Run YOLOv8 inference on the frame
            results = model(img_bgr, conf=0.3, device='cuda:0', classes=0)
            annotated_frame = results[0].plot(conf=True, line_width=2, font_size=2)

            # 计算检测到的人数
            person_count = len(results[0])

            # 添加人数中文文本
            text = f"人数: {person_count}"
            position = (10, 10)
            font_path = "microsoft-yahei/chinese.msyh.ttf"
            font_size = 30
            color = (0, 255, 255)
            annotated_frame = put_chinese_text(annotated_frame, text, position, font_path, font_size, color)

            # 添加帧数（FPS）中文文本
            fps_text = f"帧数: {fps:.2f}"
            fps_position = (10, 70)
            annotated_frame = put_chinese_text(annotated_frame, fps_text, fps_position, font_path, font_size, color)

            frame = annotated_frame


# 一个线程，用于通过WebSocket连接发送视频帧：
def vedioSend_thread1(n):
    global base64img
    global flag
    print('send')
    time.sleep(3)
    while True:
        if frame is not None:  # Check if the frame is not empty
            image = cv2.imencode('.jpg', frame)[1]
            base64_data = base64.b64encode(image)
            s = base64_data.decode()
            base64img = 'data:image/jpeg;base64,{}'.format(s)
            flag = True
        time.sleep(speed)


# 一个函数，用于启动视频捕获和发送线程
def from_vedio():
    thread1 = threading.Thread(target=vedioSend_thread1, args=(1,))
    thread1.start()
    thread2 = threading.Thread(target=vedioCapture_thread2, args=(1,))
    thread2.start()


# 一个异步WebSocket处理函数，用于发送视频帧
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

# 一个异步WebSocket处理函数，用于接收控制指令
async def receiveSignal(websocket, path):
    async for message in websocket:
        if  message == "switch_to_camera":
            #切换到摄像头
            rtsp_path = 0
            print('切换到摄像头')
        else:
            video_data = message.split(":", 1)[1]
            #切换到RTSP
            rtsp_path = message

            print('切换到RTSP',video_data)
# 主要的异步函数，启动WebSocket服务器
async def main():
    sendvideo = websockets.serve(sendImg, "localhost", 8765)
    receive_signal_server = websockets.serve(receiveSignal, "localhost", 8766)
    await asyncio.gather(sendvideo,receive_signal_server)
    # async with websockets.serve(sendImg, "localhost", 8767):
    #     await asyncio.Future()  # run forever


# 启动视频捕获和WebSocket服务器：
from_vedio()

try:
    asyncio.run(main())
    print('run')
except KeyboardInterrupt:
    print("Shutting down gracefully...")

