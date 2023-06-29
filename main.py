import threading
import base64
import time
import cv2
import asyncio
import websockets

rtsp_path = 'videos/example.mp4'  # RTSP路径或摄像头索引
camera1 = None
frame = None
base64img = ''
flag = True
speed = 0.01 # 视频帧率 表示多少秒一帧

# 视频获取
def vedioCapture_thread2(n):
	global camera1
	camera1 = cv2.VideoCapture(rtsp_path)
	global frame
	while True:
		_, img_bgr = camera1.read()

		if img_bgr is None:
			camera1 = cv2.VideoCapture(rtsp_path)
			print('丢失帧')
		else:
			frame=img_bgr
		# cv2.imshow('video', frame)
		if cv2.waitKey(50) == 27:
			break

def vedioSend_thread1(n):
	global base64img
	global flag
	print('send')
	time.sleep(3)
	while True:
		image = cv2.imencode('.jpg', frame)[1]
		base64_data = base64.b64encode(image)
		s = base64_data.decode()
		# print('data:image/jpeg;base64,%s'%s)
		base64img = 'data:image/jpeg;base64,{}'.format(s)
		flag = True
		# server.send_message_to_all('data:image/jpeg;base64,%s'%s)
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
    # start a websocket server

    async with websockets.serve(sendImg, "localhost", 8767):
        await asyncio.Future()  # run forever

from_vedio()
asyncio.run(main())