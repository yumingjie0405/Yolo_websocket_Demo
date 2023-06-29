import time
from tqdm import tqdm

import cv2
import numpy as np

from ultralytics import YOLO

import supervision as sv

import matplotlib.pyplot as plt

model = YOLO('yolov8l.pt')

VIDEO_PATH = 'videos/bridge-short.mp4'

# 获取视频信息
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
generator = sv.get_video_frames_generator(VIDEO_PATH)
iterator = iter(generator)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
while (cap.isOpened()):
    success, frame = cap.read()
    frame_count += 1
    if not success:
        break
cap.release()
print('视频总帧数为', frame_count)

frame = next(iterator)
polygons = [
    np.array([[600, 680], [927, 680], [851, 950], [42, 950]]),
    np.array([[987, 680], [1350, 680], [1893, 950], [1015, 950]])
]

zones = [sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh) for polygon in polygons]
# 配色方案
colors = sv.ColorPalette.default()

# 区域可视化，每个区域配一个 PolygonZoneAnnotator
zone_annotators = [
    sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(index), thickness=6, text_thickness=12, text_scale=4)
    for index, zone in enumerate(zones)
]

box_annotators = [
    sv.BoxAnnotator(color=colors.by_idx(index), thickness=2, text_thickness=4, text_scale=2)
    for index in range(len(polygons))
]
# YOLOV8 推理预测
results = model(frame, imgsz=640, verbose=False, show=False, device='cuda:0')[0]

# 用 supervision 解析预测结果
detections = sv.Detections.from_yolov8(results)
# 遍历每个区域对应的所有 Annotator
for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
    # 判断目标是否在区域内
    mask = zone.trigger(detections=detections)

    # 筛选出在区域内的目标
    detections_filtered = detections[mask]

    # 画框
    frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)

    # 画区域，并写区域内目标个数
    frame = zone_annotator.annotate(scene=frame)

    # # 显示
    # plt.imshow(frame)
    # plt.show()
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
# cap.release()
# cv2.destroyAllWindows()

# # 视频获取和处理
#
# def vedioCapture_thread2(n):
#     global camera1
#
#     camera1 = cv2.VideoCapture(rtsp_path)
#     camera1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

