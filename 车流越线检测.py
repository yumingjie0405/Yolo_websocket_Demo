import time
from tqdm import tqdm

import cv2
import numpy as np

from ultralytics import YOLO

import supervision as sv
from supervision.draw.color import Color

model = YOLO('yolov8s.pt')

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
# 车流越线计数-上海漕河泾宜山路隧道
input_path = 'videos/line_count_2.mp4'
filehead = input_path.split('/')[-1]
output_path = "out-" + filehead

# 获取视频总帧数
cap = cv2.VideoCapture(input_path)
frame_count = 0
while (cap.isOpened()):
    success, frame = cap.read()
    frame_count += 1
    if not success:
        break
cap.release()
print('视频总帧数为', frame_count)

cap = cv2.VideoCapture(input_path)
frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

with tqdm(total=frame_count - 1) as pbar:
    for result in model.track(source=input_path, show=False, stream=True, verbose=False, device='cuda:0'):
        frame = result.orig_img

        # 用 supervision 解析预测结果
        detections = sv.Detections.from_yolov8(result)

        ## 过滤掉某些类别
        # detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

        # 解析追踪ID
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        # 获取每个目标的：追踪ID、类别名称、置信度
        class_ids = detections.class_id  # 类别ID
        confidences = detections.confidence  # 置信度
        tracker_ids = detections.tracker_id  # 多目标追踪ID
        labels = ['#{} {} {:.1f}'.format(tracker_ids[i], model.names[class_ids[i]], confidences[i] * 100) for i in
                  range(len(class_ids))]

        # 绘制目标检测可视化结果
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # 越线检测

        line_counter.trigger(detections=detections)

        line_annotator.annotate(frame=frame, line_counter=line_counter)

        out.write(frame)

        pbar.update(1)

cv2.destroyAllWindows()
out.release()
cap.release()
print('视频已保存', output_path)

print('共跨线进入 ', line_counter.in_count)
print('共跨线离开 ', line_counter.out_count)
