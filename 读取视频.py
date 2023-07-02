import cv2

# 视频路径
video_path = 0

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 逐帧读取和显示视频
while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 如果视频帧读取失败，退出循环
    if not ret:
        break

    # 显示当前帧
    cv2.imshow('Video', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭视频流和窗口
cap.release()
cv2.destroyAllWindows()
