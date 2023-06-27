import time
import cv2
from ultralytics import YOLO
import supervision as sv
from supervision.draw.color import Color
import base64
import asyncio
import threading
import websockets
from PIL import Image
from io import BytesIO

# 越线检测位置
LINE_START = sv.Point(260, 790)
LINE_END = sv.Point(1860, 790)
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

# 视频路径
input_path = 0

# 全局变量
frame = None
speed = 0.1


# 视频获取和处理
def videoCaptureThread():
    global frame
    cap = cv2.VideoCapture(input_path)
    print("video loaded {}".format(cap.isOpened() and "successfully" or "unsuccessfully"))
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("video read error")
            # 回到视频的起始位置
            cap = cv2.VideoCapture(input_path)
        time.sleep(speed)


def videoSendThread():
    global frame
    global base64img
    global speed
    global flag
    base64img = ''
    flag = True
    time.sleep(3)

    while True:
        if frame is not None:
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = BytesIO()
            image_pil.save(buffer, format="JPEG")
            base64_data = base64.b64encode(buffer.getvalue())
            base64img = 'data:image/jpeg;base64,{}'.format(base64_data.decode())
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


async def main(websocket, path):
    global frame
    # 加载YOLOv8模型
    print("loading model...")
    try:
        model = YOLO('yolov8l.pt')
        print("model loaded successfully")
    except Exception as e:
        print('Failed to load model', str(e))
        return

    while True:
        if frame is not None:
            result = model.track(source=frame, show=False, stream=True, verbose=False, device='cuda:0')
            try:
                frame_result = next(result)
                frame = frame_result.orig_img

                # 用 supervision 解析预测结果
                detections = sv.Detections.from_yolov8(frame_result)

                ## 过滤掉某些类别
                # detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]
                # 检查 frame_result.boxes 是否为 None
                if frame_result.boxes is not None:
                    # 解析追踪ID
                    detections.tracker_id = frame_result.boxes.id.cpu().numpy().astype(int)

                    # 绘制越线检测线
                    # 获取每个目标的：追踪ID、类别名称、置信度
                    class_ids = detections.class_id  # 类别ID
                    confidences = detections.confidence  # 置信度
                    tracker_ids = detections.tracker_id  # 多目标追踪ID
                    labels = ['#{} {} {:.1f}'.format(tracker_ids[i], model.names[class_ids[i]], confidences[i] * 100)
                              for i in range(len(class_ids))]

                    # 绘制目标检测可视化结果
                    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

                # 越线检测
                line_counter.trigger(detections=detections)
                line_annotator.annotate(frame=frame, line_counter=line_counter)

                # 将帧转换为base64格式
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buffer = BytesIO()
                img_pil.save(buffer, format="JPEG")
                base64_data = base64.b64encode(buffer.getvalue())
                base64_frame = 'data:image/jpeg;base64,{}'.format(base64_data.decode())

                try:
                    await websocket.send(base64_frame)
                except websockets.exceptions.ConnectionClosedError as e:
                    print("connection closed error")
                    break
                except Exception as e:
                    print(e)
                    break
            except StopIteration:
                print("StopIteration: No more frames from model.track")
                # 回到视频的起始位置
                cap = cv2.VideoCapture(input_path)
                print("video loaded {}".format(cap.isOpened() and "successfully" or "unsuccessfully"))
                continue
            await asyncio.sleep(0.01)

        # 关闭连接
        await websocket.close()
        print("websocket 关闭")


thread2 = threading.Thread(target=videoCaptureThread)
thread2.start()
print('------------------')
start_server = websockets.serve(main, "localhost", 8767)
print("WebSocket服务器启动")

asyncio.get_event_loop().run_until_complete(start_server)
print('------------------')
asyncio.get_event_loop().run_forever()
