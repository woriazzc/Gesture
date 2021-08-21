from utils import predict
import cv2
import numpy as np
from data_utils import compute_test_transform


capture = 0
blurValue = 5
bgSubThreshold = 36
threshold = 60
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
x1 = 380
y1 = 60
x2 = 640
y2 = 350
Gesturetype = ['666', 'yech', 'stop', 'punch', 'OK']

if __name__ == '__main__':
    # 读取默认摄像头
    cap = cv2.VideoCapture(capture)
    # 设置捕捉模式
    cap.set(10, 200)
    # 背景减法创建及初始化
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

    transform = compute_test_transform()
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        # 镜像转换
        frame = cv2.flip(frame, 1)

        cv2.imshow('Original', frame)
        # 双边滤波
        frame = cv2.bilateralFilter(frame, 5, 50,100)

        # 绘制矩形，第一个为左上角坐标(x,y)，第二个为右下角坐标
        # rec = cv2.rectangle(frame, (220, 50), (450, 300), (255, 0, 0), 2)
        rec = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 定义roi区域，第一个为y的取值，第2个为x的取值
        # frame = frame[50:300, 220:450]
        frame = frame[y1:y2, x1:x2]
        cv2.imshow('bilateralFilter', frame)
        # 背景减法运动检测
        bg = bgModel.apply(frame, learningRate=0)
        # 显示背景减法的窗口
        cv2.imshow('bg', bg)
        # 图像边缘处理--腐蚀
        fgmask = cv2.erode(bg, skinkernel, iterations=1)
        # 显示边缘处理后的图像
        cv2.imshow('erode', fgmask)
        # 将原始图像与背景减法+腐蚀处理后的蒙版做"与"操作
        bitwise_and = cv2.bitwise_and(frame, frame, mask=fgmask)
        # 显示与操作后的图像
        cv2.imshow('bitwise_and', bitwise_and)
        # 灰度处理
        gray = cv2.cvtColor(bitwise_and, cv2.COLOR_BGR2GRAY)
        # 高斯滤波
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 2)
        cv2.imshow('GaussianBlur', blur)

        # 使用自适应阈值分割(adaptiveThreshold)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imshow('thresh', thresh)

        Ges = cv2.resize(thresh, (100, 100))
        # 图像的阈值处理(采用ostu)
        # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # cv2.imshow('threshold1', thresh)

        Ges = transform(Ges)
        prediction = predict(Ges)
        ges_type = Gesturetype[prediction]

        # print(ges_type)
        cv2.putText(rec, ges_type, (x1, y1), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=2, color=(0, 0, 255))
        cv2.imshow('Original', rec)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break