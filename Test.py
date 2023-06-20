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
from supervision.draw.color import Color

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')
# 摄像头编号
'''
# Open the video file
# cap = cv2.VideoCapture("https://open.ys7.com/v3/openlive/K10170061_1_1.m3u8?expire=1717206565&id=586491899410382848&t=e862caf9959031a4c7b74c33388f2dcbf7bf5f358c5ed4022861c20aa1766100&ev=100")
# cap = cv2.VideoCapture("MOT16-03.mp4")
'''
rtsp_path = "videos/example.mp4"
# rtsp_path ="https://open.ys7.com/v3/openlive/K10170061_1_1.m3u8?expire=1717206565&id=586491899410382848&t=e862caf9959031a4c7b74c33388f2dcbf7bf5f358c5ed4022861c20aa1766100&ev=100"
# rtsp_path = 2
camera1 = None
frame = None
base64img = ''
flag = True
speed = 0.001  # 视频帧率 表示多少秒一帧
# 在全局变量中添加锁
frame_lock = threading.Lock()

# 越线检测位置
LINE_START = sv.Point(50, 300)
LINE_END = sv.Point(550, 300)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

# 线的可视化配置
line_color = Color(r=224, g=57, b=151)
line_annotator = sv.LineZoneAnnotator(thickness=5, text_thickness=2, text_scale=1, color=line_color)

# 目标检测可视化配置
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)


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
    global camera1, frame, frame_lock
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
            # results = model.track(source=img_bgr, show=False, stream=True, verbose=False, device='cuda:0',classes=0)
            results = list(
                model.track(source=img_bgr, show=False, stream=True, verbose=False, device='cuda:0', classes=0,
                            conf=0.7))

            # results = model(img_bgr, conf=0.2, device=0, classes=0)
            annotated_frame = results[0].plot(conf=True, line_width=2, font_size=2,color={0,255,0})
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

            # 用 supervision 解析预测结果
            detections = sv.Detections.from_yolov8(results[0])

            # detections.tracker_id = results[0].id.cpu().numpy().astype(int)  # 解析追踪ID

            # 越线检测
            line_counter.trigger(detections=detections)

            # 绘制目标检测可视化结果
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=None)

            # 绘制越线检测可视化结果
            annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)

            frame = annotated_frame

            with frame_lock:  # 添加锁
                frame = annotated_frame


def vedioSend_thread1(n):
    global base64img, flag, frame, frame_lock
    print('send')
    time.sleep(3)
    while True:
        if frame is not None:  # Check if the frame is not empty
            with frame_lock:  # 添加锁
                image = cv2.imencode('.jpg', frame)[1]
            base64_data = base64.b64encode(image)
            s = base64_data.decode()
            base64img = 'data:image/jpeg;base64,{}'.format(s)
            flag = True
        time.sleep(speed)


def from_vedio():
    thread1 = threading.Thread(target=vedioSend_thread1, args=(1,))
    thread1.start()
    thread2 = threading.Thread(target=vedioCapture_thread2, args=(1,))
    thread2.start()


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


async def main():
    async with websockets.serve(sendImg, "localhost", 8766):
        await asyncio.Future()  # run forever


from_vedio()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Shutting down gracefully...")

from_vedio()
asyncio.run(main())
