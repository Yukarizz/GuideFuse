import torch
import torchvision.transforms as transform
from torch import nn
from torch.nn import functional as F
import cv2 as cv
import matplotlib.pyplot as plt

def matplotlib_multi_pic1(pyramid):
    i = 0
    for layer in pyramid:
        img = layer
        title = "title" + str(i + 1)
        # 行，列，索引
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(title, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        i += 1
    plt.show()
def get_laplacian_pyramid(gaussian_pyramid, up_times=5):
    laplacian_pyramid = [gaussian_pyramid[-1]]
    Sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    for i in range(up_times, 0, -1):
        # i的取值为5,4,3,2,1,0也就是拉普拉斯金字塔有6层
        temp_pyrUp = Sampler(gaussian_pyramid[i])
        temp_lap = torch.subtract(gaussian_pyramid[i - 1], temp_pyrUp)
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid


def get_gaussian_pyramid(ori_image, down_times=5):
    # 1：添加第一个图像为原始图像
    temp_gau = ori_image
    gaussian_pyramid = [temp_gau]
    h,w = ori_image.shape[2],ori_image.shape[3]
    for i in range(down_times):
        # 2：连续存储5次下采样，这样高斯金字塔就有6层
        temp_gau = F.interpolate(temp_gau, size=(int(h/2), int(w/2)))
        gaussian_pyramid.append(temp_gau)
        h = h/2
        w = w/2
    return gaussian_pyramid




if __name__ == '__main__':
    input = torch.randn(size=(1,3,64,64))
    gaussian = get_gaussian_pyramid(input)
    laplacian = get_laplacian_pyramid(gaussian)
    for layer in laplacian:
        print(layer.shape)
    image = cv.imread("women.png")
    image = image[:,:,[2,1,0]]
    trans = transform.Compose([transform.ToTensor(),
                              transform.Resize(size=(512,512))])
    image = trans(image)
    image = torch.reshape(image,shape=[1,3,512,512])
    gaussian = get_gaussian_pyramid(image)
    laplacian = get_laplacian_pyramid(gaussian)
    image_list = []
    toPIL = transform.ToPILImage()
    for layer in laplacian:
        layer = layer.squeeze(0)
        print(layer.shape)
        layer = toPIL(layer)
        image_list.append(layer)
    matplotlib_multi_pic1(image_list)
