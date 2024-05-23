import torch
from network import *
import cv2 as cv
import time
from torchvision.transforms import transforms
from PIL import Image
import torchvision
import os
from torch.nn import functional as F
from kornia import augmentation
from utils import gradient,guide_filter,lap_enhance,flop_counter,gradient_canny
EPSILON = 1e-10
from skimage.metrics import peak_signal_noise_ratio as psnr

def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors

def get_img_path(path1, path2):
    imglist1 = []
    imglist2 = []
    filenames1 = os.listdir(path1)
    filenames1.sort(key=lambda x: int(x[:-4]))
    filenames2 = os.listdir(path2)
    filenames2.sort(key=lambda x: int(x[:-4]))
    for name in filenames1:
        img_path = path1 + "/" + name
        imglist1.append(img_path)
    for name in filenames2:
        img_path = path2 + "/" + name
        imglist2.append(img_path)
    return imglist1, imglist2


def saveimg_fuse(img1, img2,grad_map, img_fuse,class_path, num=0):
    # img = torchvision.utils.make_grid([img1[0].cpu(), img2[0].cpu(), img_fuse[0].cpu()], nrow=3)
    img = torchvision.utils.make_grid([img_fuse[0].cpu()], nrow=1)
    torchvision.utils.save_image(img, fp=(os.path.join('fusedata/'+class_path+'/fuse1/'+'%d.png' % (num + 1))))


def saveimg_recon(img, img_re, num=0):
    img = torchvision.utils.make_grid([img[0].cpu(), img_re[0].cpu()], nrow=2)
    torchvision.utils.save_image(img, fp=(os.path.join('fusedata/result_%d.jpg' % (num))))


def loadimg(path1, path2, device='cuda', mode='GRAY'):
    img1 = cv.imread(path1)
    img2 = cv.imread(path2)
    h,w = img1.shape[0],img1.shape[1]
    # h,w = 256,256
    transform = transforms.Compose([
        transforms.Resize([h,w]),
        transforms.ToTensor(),
    ])
    if mode =='GRAY':
        img1 = Image.fromarray(cv.cvtColor(img1, cv.COLOR_BGR2GRAY))
        img2 = Image.fromarray(cv.cvtColor(img2, cv.COLOR_BGR2GRAY))
    elif mode =='RGB':
        img1 = Image.fromarray(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    img1 = transform(img1)
    img2 = transform(img2)
    if mode == 'GRAY':
        img1 = torch.reshape(img1, shape=[1, 1, h, w]).to(device)
        img2 = torch.reshape(img2, shape=[1, 1, h, w]).to(device)
    elif mode == 'RGB':
        img1 = torch.reshape(img1, shape=[1, 3, 256, 256]).to(device)
        img2 = torch.reshape(img2, shape=[1, 3, 256, 256]).to(device)
    return img1, img2


def Cross(logits, target):
    logits = torch.sigmoid(logits)
    loss = - target * torch.log(logits) - (1 - target) * torch.log(1 - logits)
    loss = loss.mean()
    return loss.item()


def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)
    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2
    return tensor_f


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


def my_softmax(x, y):
    return torch.exp(x) / (torch.exp(x) + torch.exp(y) + EPSILON)
def get_union_weight(x,y):
    return x / (x+y)
def get_enchancement(g_x,g_y):
    weight = torch.abs(g_x)/(torch.abs(g_x)+torch.abs(g_y)+1e-10)
    return weight
if __name__ == "__main__":

    aug = augmentation.AugmentationSequential(
                augmentation.RandomAffine([-5,5],[0.1,0.1],[0.9,1.1],None, p=1.0,align_corners=True,padding_mode=(2)),
                # augmentation.RandomElasticTransform(p=1.0)
            )

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda'
    # class_path = "Exposure"
    # Class = ['Tno','Exposure','Med','Road']
    Class = ['INO_video_analytics','OSU','Tno','Road',]
    fuse_model = Fusion_network(mode="fuse").to(device)
    total_time = 0.

    state_dict = torch.load('weights/best_fusion.pt')
    fuse_model.load_state_dict(state_dict['weight'])
    encoder = fuse_model.encoder
    decoder = fuse_model.decoder
    grad_branch = fuse_model.grad_branch
    guider = fuse_model.encoder.guide2base2
    changer = fuse_model.encoder.guide2base2.fc
    enchancement = [5,3]
    channel_attn = "auto" # auto or nuclear
    # 加载融合子网络参数
    # fuse = fusion_layer().to(device)
    # fuse_layer_state_dict = torch.load('weights/fusion_layer/best_layer.pt')
    # fuse.load_state_dict(fuse_layer_state_dict['weight'])
    for class_path in Class:
        print("model epoch:%d   "%(state_dict['epoch'])+"Processing Dataclass:%s"%class_path)
        psnr_t = 0
        if class_path == "Exposure" or class_path == "Med":
            path1 = "testdata/" + class_path + "/" + "1"
            path2 = "testdata/" + class_path + "/" + "2"
        elif class_path=="Tno" or class_path == "Road" or class_path== "INO_video_analytics" or class_path == "OSU":
            path1 = "testdata/" + class_path + "/" + "vi"
            path2 = "testdata/" + class_path + "/" + "ir"
        else:
            path1 = "testdata/" + class_path + "/" + "visible"
            path2 = "testdata/" + class_path + "/" + "infrared"
        n = 0
        a, b = get_img_path(path1, path2)
        # img1是可见光图像，img2是红外图像
        with torch.no_grad():
            for img_path1, img_path2 in zip(a, b):

                img1, img2 = loadimg(img_path1, img_path2, device=device, mode='GRAY')
                start = time.perf_counter()
                encoder.eval()
                decoder.eval()
                # ||| get fused gradient map |||
                grad_map1 = gradient_canny((img1))
                grad_map2 = gradient_canny((img2))
                img1_detail = img1 - guide_filter(I=img1, p=img1,window_size=11,eps=0.2)
                img2_detail = img2 - guide_filter(I=img2, p=img2, window_size=11, eps=0.2)
                weight = 5*get_enchancement(img1_detail,img2_detail)


                # ||| fuse strategy |||
                # fmp1,fmp2未经过Encoder最后一层指导
                # flop_counter(encoder,img1)
                fmp1, guide1 = encoder(img1)
                fmp2, guide2 = encoder(img2)
                # fmp1 = aug(fmp1,params=aug._params)
                # fmp2 = aug(fmp2,params=aug._params)

                grad_map1 = enchancement[0]*grad_map1
                grad_map2 = enchancement[1]*grad_map2

                grad_map = torch.cat([grad_map1,  grad_map2], dim=1)
                # |* 手动融合* |
                # 若测试模型的重建能力，应该使用fmps1，fmps2
                fmps1 = guider(fmp1,guide1)
                fmps2 = guider(fmp2,guide2)
                # spatial fusion
                fmps = spatial_fusion(fmps1, fmps2)
                # max decision
                fmp_max = fmps1*(fmps1>fmps2) + fmps2*(fmps2>fmps1)

                # channel attention fusion
                # guide1和guide2分别通过Encoder的最后一层指导分支进行维度的压缩
                if channel_attn =='auto':
                    guidec1 = guide1
                    guidec1 = guidec1.view((guidec1.shape[0], guidec1.shape[1]))
                    guidec1 = changer(guidec1)
                    guidec1 = guidec1.view((guidec1.shape[0], guidec1.shape[1], 1, 1))
                    guidec2 = guide2
                    guidec2 = guidec2.view((guidec2.shape[0], guidec2.shape[1]))
                    guidec2 = changer(guidec2)
                    guidec2 = guidec2.view((guidec2.shape[0], guidec2.shape[1], 1, 1))
                    fmpc = ((get_union_weight(guidec1,guidec2)*fmp1) + (get_union_weight(guidec2,guidec1)*fmp2))
                else:
                    shape = fmp1.size()
                    global_p1 = nuclear_pooling(fmp1)
                    global_p2 = nuclear_pooling(fmp2)
                    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
                    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)
                    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
                    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])
                    fmpc = global_p_w1 * fmp1 + global_p_w2 * fmp2


                # 最终的特征图使用两种策略获得特征图的平均
                fmp = (fmps+fmpc)/2

                # |*** 自动融合 *** |下一行不注释时候需要注释手动融合所有行
                # fmp = fuse(fmp1,fmp2)
                # |*** end ***|

                # ||| fuse end |||
                guide = (guide1+guide2)/2
                # guide = guide1*(guide1>guide2) + guide2*(guide2>guide1)
                # 1 是可见光，2 是红外光
                # 测试模型重建grad_map1 = torch.cat([grad_map1,grad_map1],dim=1)，img_re = decoder(fmps1, guide1, grad_map1)
                # 测试模型融合能力img_re = decoder(fmp, guide, grad_map)
                # flop_counter(decoder, [fmp,guide,grad_map])
                img_re = decoder(fmp, guide, grad_map)
                # img_re = torch.clamp(img_re,min=0,max=1)
                end = time.perf_counter()
                saveimg_fuse((img1), (img2), grad_map1,img_fuse=img_re,class_path=class_path, num=n)
                # saveimg_recon(img1, img_re, num=n)
                n += 1
                psnr1 = psnr(img1.cpu().numpy(), img_re.cpu().numpy())
                psnr2 = psnr(img2.cpu().numpy(), img_re.cpu().numpy())
                psnr_t += ((psnr1 + psnr2) / 2)
                time_one = (end-start)
                total_time += time_one
    print("Down! Total psnr:%.5f"%psnr_t)
    print('Running time: %.5f Seconds per image' % (total_time/n))