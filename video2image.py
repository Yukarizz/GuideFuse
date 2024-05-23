import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from PIL import Image


def Pic2Video():
    imgPath = "youimgPath"  # 读取图片路径
    videoPath = "youvideoPath"  # 保存视频名称，路径默认为当前文件夹下

    images = os.listdir(imgPath)
    # 如果视频出现乱帧的情况，采用如下函数可以改正
    images.sort()


    fps = 25  # 每秒25帧数

    # VideoWriter_fourcc为视频编解码器 ('I', '4', '2', '0') —>(.avi) 、('P', 'I', 'M', 'I')—>(.avi)、('X', 'V', 'I', 'D')—>(.avi)、('T', 'H', 'E', 'O')—>.ogv、('F', 'L', 'V', '1')—>.flv、('m', 'p', '4', 'v')—>.mp4
    fourcc = VideoWriter_fourcc(*"MJPG")

    h, w, _ = cv2.imread(os.path.join(imgPath, images[0])).shape

    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, (w, h))
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])  # 这里的路径只能是英文路径
        # frame = cv2.imdecode(np.fromfile((imgPath + images[im_name]), dtype=np.uint8), 1)  # 此句话的路径可以为中文路径
        print(im_name)
        videoWriter.write(frame)
    print("图片转视频结束！")
    videoWriter.release()
    cv2.destroyAllWindows()


def Video2Pic():
    videoPath = r"E:\dataset\INO_video_analyntic\ino_visitorparking\INO_VisitorParking\INO_VisitorParking\INO_VisitorParking_T.avi"  # 读取视频路径
    imgPath = r"E:\dataset\INO_video_analyntic\ino_visitorparking\INO_VisitorParking\INO_VisitorParking\ir"  # 保存图片路径

    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        # cv2.imwrite(imgPath + str(frame_count).zfill(4), frame)
        cv2.imwrite(imgPath + "\%d.jpg" % frame_count, frame)
        cv2.waitKey(1)
    cap.release()
    print("视频转图片结束！")


if __name__ == '__main__':
    Video2Pic()  # 视频转图像
    # Pic2Video() #图像转视频
