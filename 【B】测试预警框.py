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
rtsp_path = "videos/bridge-short.mp4"
# rtsp_path ="https://open.ys7.com/v3/openlive/K10170061_1_1.m3u8?expire=1717206565&id=586491899410382848&t=e862caf9959031a4c7b74c33388f2dcbf7bf5f358c5ed4022861c20aa1766100&ev=100"
# rtsp_path = 0
camera1 = None
frame = None
base64img = ''
flag = True
speed = 0.01  # 视频帧率 表示多少秒一帧


# 在OpenCV图像上添加中文文字
def put_chinese_text(image, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv2


# 制定绘制的多边形区域
# polygons = [
#     np.array([[600, 680], [927, 680], [851, 950], [42, 950]]),
#     np.array([[987, 680], [1350, 680], [1893, 950], [1015, 950]])
# ]
polygons = [
    np.array([[600, 680], [927, 680], [851, 950], [42, 950]]),
    np.array([[987, 680], [1350, 680], [1893, 950], [1015, 950]])
]


# 视频获取和处理

def vedioCapture_thread2(n):
    global camera1

    camera1 = cv2.VideoCapture(rtsp_path)
    global frame
    play_video = True
    # Adding FPS calculation
    fps = 0
    last_time = time.time()

    while True:
        if play_video:
            _, img_bgr = camera1.read()
            frame = img_bgr
            height, width, _ = img_bgr.shape

            if img_bgr is None:
                camera1 = cv2.VideoCapture(rtsp_path)
                print('丢失帧')
            else:

                # Update FPS calculation
                current_time = time.time()
                fps = 1 / (current_time - last_time)
                last_time = current_time

                zones = [sv.PolygonZone(polygon=polygon, frame_resolution_wh=(width, height)) for polygon in
                         polygons]
                # 配色方案
                colors = sv.ColorPalette.default()
                # 区域可视化，每个区域配一个 PolygonZoneAnnotator
                zone_annotators = [
                    sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(index), thickness=6, text_thickness=12,
                                            text_scale=4)
                    for index, zone in enumerate(zones)
                ]

                # 目标检测可视化，每个区域配一个 BoxAnnotator
                box_annotators = [
                    sv.BoxAnnotator(color=colors.by_idx(index), thickness=2, text_thickness=4, text_scale=2)
                    for index in range(len(polygons))
                ]

                # YOLOV8 推理预测
                results = model(img_bgr, imgsz=640, verbose=False, show=False, device='cuda:0')[0]

                # 用 supervision 解析预测结果
                detections = sv.Detections.from_yolov8(results)

                # 遍历每个区域对应的所有 Annotator
                for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                    # 判断目标是否在区域内
                    mask = zone.trigger(detections=detections)
                    # print(zone.current_count)
                    # 筛选出在区域内的目标+

                    detections_filtered = detections[mask]

                    # 画框
                    frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)

                    # 画区域，并写区域内目标个数
                    frame = zone_annotator.annotate(scene=frame)
            # Check if the video has reached the end
            current_frame = camera1.get(cv2.CAP_PROP_POS_FRAMES)
            frames = camera1.get(cv2.CAP_PROP_FRAME_COUNT)
            # print("current_frame:{}  frames:{}".format(current_frame,frames))
            if current_frame == frames-3:

                if camera1.isOpened():
                    camera1.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Set video position to the beginning
                    print('reset')
                else:
                    print("Invalid video capture object")


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
    async with websockets.serve(sendImg, "localhost", 8767):
        await asyncio.Future()  # run forever


from_vedio()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Shutting down gracefully...")
