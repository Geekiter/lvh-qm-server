import json
import cv2
from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
from math import fabs
import numpy as np
from flask_cors import CORS

from img import edge, equal, contour, brightness, smooth, gamma_correction, sharpen

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


def avi2img(avi_path, img_dir):
    cap = cv2.VideoCapture(avi_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(img_dir, str(i) + '.jpg'), frame)
            i += 1
        else:
            break


@app.route("/api/track", methods=['POST'])
def track():
    params = request.get_json()
    file_path = params['file_path']
    # file_path = r"C:\Users\alber\dev\data\lvh\40320484肥厚\Image08.avi"
    model = YOLO("yolo_weight_best.pt")
    results = model.track(file_path, show=False, stream=True)
    upper_thickness_list = []
    lower_thickness_list = []
    for result in results:
        upper_thickness = fabs(result.boxes.xyxy[0][1] - result.boxes.xyxy[1][1])
        lower_thickness = fabs(result.boxes.xyxy[0][3] - result.boxes.xyxy[1][3])
        upper_thickness_list.append(upper_thickness)
        lower_thickness_list.append(lower_thickness)

    # TODO 返回2个数组，分别是上下圈的厚度
    fake_result = {
        'file_path': file_path,
        'up': upper_thickness_list,
        'down': lower_thickness_list
    }
    return jsonify(fake_result)


@app.route("/api/to_webm", methods=['POST'])
def to_webm():
    params = request.get_json()
    file_path = params['file_path']
    # if file_path ends is not webm, convert it to webm
    if file_path.endswith('.webm'):
        return jsonify({'file_path': file_path})
    else:
        # 保存位置为file_path的文件夹下，名字tmp+前一个序号+1
        # 获取保存文件夹下所有tmp开头的文件
        tmp_files = [f for f in os.listdir(os.path.dirname(file_path)) if f.startswith('tmp')]
        # 获取最大的序号
        max_num = 0
        for f in tmp_files:
            num = int(f[3:].split('.')[0])
            if num > max_num:
                max_num = num
        # 保存位置
        save_path = os.path.join(os.path.dirname(file_path), f"tmp{max_num + 1}.webm")
        # save_path = os.path.join(os.path.dirname(file_path), "tmp.webm")
        cap = cv2.VideoCapture(file_path)
        # 获取视频帧率和尺寸
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
        print(f"fps: {fps}, size: {size}")
        # 边缘提取然后保存
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(save_path, fourcc, fps, size, True)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        return jsonify({'file_path': save_path})


@app.route("/api/filters", methods=['POST'])
def filters():
    params = request.get_json()
    file_path = params['file_path']
    # 亮度alpha
    bri_a = float(params.get('bri_a', 1.0))
    # 亮度beta
    bri_b = int(params.get('bri_b', 0))
    # 平滑帧大小
    smooth_frame1 = int(params.get('smooth_frame1', 1))
    equal_switch = params.get('equal_switch', False)
    # gamma校正值
    gamma1 = float(params.get('gamma1', 2.2))
    # 边缘检测开关
    edge_switch = params.get('edge_switch', False)
    # 锐化程度
    sharpen_level = int(params.get('sharpen_level', 0))
    # 轮廓检测开关
    contour_switch = params.get('contour_switch', False)

    # file_path = r".\Image08.avi"
    # save_path = r"./edge.mp4"
    # 保存位置为file_path的文件夹下，名字tmp+前一个序号+1
    # 获取保存文件夹下所有tmp开头的文件
    tmp_files = [f for f in os.listdir(os.path.dirname(file_path)) if f.startswith('tmp')]
    # 获取最大的序号
    max_num = 0
    for f in tmp_files:
        num = int(f[3:].split('.')[0])
        if num > max_num:
            max_num = num
    # 保存位置
    save_path = os.path.join(os.path.dirname(file_path), f"tmp{max_num + 1}.webm")
    # save_path = os.path.join(os.path.dirname(file_path), "tmp.webm")

    cap = cv2.VideoCapture(file_path)
    # 获取视频帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"fps: {fps}, size: {size}")
    # 边缘提取然后保存
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(save_path, fourcc, fps, size, True)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            frame = edge(frame, edge_switch)

            frame = equal(frame, equal_switch)
            frame = contour(frame, contour_switch)

            # change brightness
            frame = brightness(frame, bri_a, bri_b)
            # 平滑处理
            frame = smooth(frame, smooth_frame1)
            # gamma校正
            frame = gamma_correction(frame, gamma1)
            # 锐化
            frame = sharpen(frame, sharpen_level)
            # 转换为rgb
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # print(frame.shape)

            out.write(frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    return jsonify({'file_path': save_path})


if __name__ == '__main__':
    app.run(host='localhost', port=16000, debug=True)
