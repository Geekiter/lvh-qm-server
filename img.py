import cv2
import numpy as np


def edge(frame, edge_switch, t1=100, t2=200):
    if edge_switch:
        # 转换为灰度图像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 边缘检测
        frame = cv2.Canny(frame, t1, t2)
    return frame


def equal(frame, equal_switch):
    if equal_switch:
        # 转换为灰度图像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化
        frame = cv2.equalizeHist(frame)
    return frame


def contour(frame, contour_switch):
    if contour_switch:
        # 转换为灰度图
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 二值化
        ret, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 轮廓检测
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 帧直接变成轮廓图
        frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    return frame


def brightness(frame, a=1, b=0):
    # change brightness
    frame = cv2.convertScaleAbs(frame, alpha=a, beta=b)
    return frame


def smooth(frame, smooth_frame):
    # 平滑处理
    if smooth_frame > 0:
        frame = cv2.blur(frame, (smooth_frame, smooth_frame))
    return frame


def gamma_correction(img, g=2.2, c=1):
    """
    gamma校正
    :param img:
    :param c: c是一个常数，c>0
    :param g: gamma值
    :return:
    """
    out = img.copy().astype(np.float32)
    out /= 255.
    out = (1 / c * out) ** (1 / g)

    out *= 255
    out = out.astype(np.uint8)

    return out


def sharpen(image, level=0):
    # Blur the image
    gauss = cv2.GaussianBlur(image, (7, 7), 0)
    # Apply Unsharp masking
    unsharp_image = cv2.addWeighted(image, 2 + level, gauss, -1 - level, 0)
    return unsharp_image


if __name__ == '__main__':
    test_video_path = r"C:\Users\alber\dev\data\lvh\40320484肥厚\Image08.avi"
    cap = cv2.VideoCapture(test_video_path)
    # 获取视频帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"fps: {fps}, size: {size}")
    # 边缘提取然后保存
    save_path = r"./edge.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 30, size, True)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            frame = edge(frame, True)

            # 转换为rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            print(frame.shape)

            out.write(frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
