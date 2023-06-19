import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import time

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')
# model = YOLO(r'F:\人群密集检测\yolov5-v7.0\runs\train\exp10\weights\best.pt')

# Open the video file
# cap = cv2.VideoCapture("https://open.ys7.com/v3/openlive/K10170061_1_1.m3u8?expire=1717206565&id=586491899410382848&t=e862caf9959031a4c7b74c33388f2dcbf7bf5f358c5ed4022861c20aa1766100&ev=100")
# cap = cv2.VideoCapture("MOT16-03.mp4")
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('example.mp4')


# 将OpenCV图像转换为PIL图像
def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# 将PIL图像转换为OpenCV图像
def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# 在OpenCV图像上添加中文文字
def put_chinese_text(image, text, position, font_path, font_size, color):
    # 将OpenCV图像转换为PIL图像
    img_pil = cv2_to_pil(image)

    # 创建一个绘画对象，并设置字体
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    # 添加文字
    draw.text(position, text, font=font, fill=color)

    # 将PIL图像转换回OpenCV图像
    img_cv2 = pil_to_cv2(img_pil)
    return img_cv2


while cap.isOpened():
    # Read a frame from the video and start the timer
    start_time = time.time()
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.1, device=0, classes=0)
        # results = model(frame)
                        # Visualize the results on the frame
        annotated_frame = results[0].plot( conf=True, line_width=2, font_size=2)

        # 计算检测到的人数
        person_count = len(results[0])

        # 添加人数中文文本
        text = f"人数: {person_count}"
        position = (10, 10)
        font_path = "microsoft-yahei/chinese.msyh.ttf"
        font_size = 50
        color = (0, 255, 255)
        annotated_frame = put_chinese_text(annotated_frame, text, position, font_path, font_size, color)

        # Calculate and add FPS text
        fps = int(1.0 / (time.time() - start_time))
        fps_text = f"FPS: {fps}"
        fps_position = (10, 80)
        annotated_frame = put_chinese_text(annotated_frame, fps_text, fps_position, font_path, font_size, color)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()