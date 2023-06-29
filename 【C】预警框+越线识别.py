import asyncio
import websockets
import cv2
import numpy as np
import base64

rtsp_path = 'videos/example.mp4'  # RTSP路径或摄像头索引
speed = 0.1  # 视频帧率，表示多少秒一帧

async def videoCapture():
    camera = cv2.VideoCapture(rtsp_path)
    while True:
        _, img_bgr = camera.read()

        if img_bgr is None:
            camera = cv2.VideoCapture(rtsp_path)
            print('丢失帧')
        else:
            # 进行深度学习推理操作
            inference_result = deep_learning_inference(img_bgr)

            # 将图像转换为base64编码字符串
            _, buffer = cv2.imencode('.jpg', img_bgr)
            base64_data = buffer.tobytes()
            base64img = base64.b64encode(base64_data).decode('utf-8')

            # 发送图像数据给WebSocket客户端
            await sendImg(base64img)

        await asyncio.sleep(speed)

def deep_learning_inference(image):
    # 在这里执行深度学习推理操作
    # 返回推理结果
    return inference_result

async def sendImg(base64img):
    async with websockets.connect('ws://localhost:8767') as websocket:
        await websocket.send(base64img)

async def main():
    await videoCapture()

asyncio.run(main())
