<center>

# 实时人流量监测系统

</center>
<br>

<h4>
<center><font face="楷体">
    安徽信息工程学院   芜湖北斗实验室  © </font>
<center>    
</h4>
<h4>
<center><font face="楷体">
余铭杰
<br>
   mjyu5@iflytek.com  </font>
<center>    
</h4>
这是一个实时人数检测和计算帧率的项目，基于Python的OpenCV、YOLOv8模型和WebSocket技术。在运行此代码之前，请确保已经安装了必要的库（如OpenCV、websockets等）以及相关依赖。

以下是一个简短的说明文档：

实时人数检测与帧率计算
这个项目使用YOLOv8模型、OpenCV和WebSocket技术实现实时人数检测和计算视频流的帧率。

### 开始之前
请确保您已经安装了以下库：
> pip install opencv-python
> 
>pip install websockets
> 
>pip install ultralytics

### 运行
将代码保存为main.py。

在终端中，运行以下命令：

```python main.py```

在客户端连接到websocket，地址为：ws://localhost:8765。

### 功能概述
实时视频流处理：从指定路径读取实时视频流。
使用YOLOv8模型进行检测：在每一帧上运行YOLOv8模型，提取目标对象（如人）。
计算并显示当前人数：在每一帧上计算检测到的人数，并在图像上显示。
计算并显示帧率：计算视频流的实时帧率，并在图像上显示。
使用WebSocket发送图像：将处理后的图像以base64编码的形式发送到WebSocket客户端。
### 注意事项
请确保您的计算机上安装了正确版本的Python和相关库。
代码中提供了一个实例视频文件example.mp4，您可以使用自己的视频文件替换它。
调整speed变量可改变视频帧率（表示多少秒一帧）。
如果有任何疑问，请随时联系。祝您使用愉快！

