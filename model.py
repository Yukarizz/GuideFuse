# -*- coding: utf-8 -*-
import torch.nn as nn
from network import Encoder,Decoder,Fusion_network
# from ssim import *
from ssim_fun import *
import torch
import torch.optim as optim
import torchvision
import os
import torch.nn.functional as F
from contiguous_params import ContiguousParams
import numpy as np
from utils import gradient,gradient_sobel,gradient_Isotropic_sobel,gradient_Prewitt,gradient_canny


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.fusion = Fusion_network()
        self.MSE_fun = nn.MSELoss()
        # self.L1_loss = nn.L1Loss()
        # k（a，b）b大了会有竖向，小了横向
        # self.SSIM_fun = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1, K=(0.01,0.03))
        self.SSIM_fun = SSIM()
        if args.contiguousparams == True:
            print("ContiguousParams---")
            parametersF = ContiguousParams(self.fusion.parameters())
            self.optimizer_G = optim.Adam(parametersF.contiguous(), lr=args.lr)
        else:
            self.optimizer_G = optim.Adam(self.fusion.parameters(), lr=args.lr)

        # self.optimizer_G = optim.Adam(self.fusion.parameters(), lr=args.lr)
        self.loss = torch.zeros(1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.5,
                                                                    patience=2,
                                                                    verbose=False, threshold=0.0001,
                                                                    threshold_mode='rel',
                                                                    cooldown=0, min_lr=0, eps=1e-10)
        self.min_loss = 1000
        self.mean_loss = 0
        self.count_batch = 0
        self.args = args
        if args.multiGPU:
            self.mulgpus()
        self.load()
        # [a*pixel+b*ssim+c*gradient]
        self.loss_hyperparameters = [1, 100, 10]
    def load(self, ):
        start_epoch = 0
        if self.args.load_pt:
            print("=========LOAD WEIGHTS=========")
            print(self.args.weights_path)
            checkpoint = torch.load(self.args.weights_path)
            start_epoch = checkpoint['epoch'] + 1
            try:
                if self.args.multiGPU:
                    print("load G")
                    self.fusion.load_state_dict(checkpoint['weight'])
                else:
                    print("load G single")
                    # 单卡模型读取多卡模型
                    state_dict = checkpoint['weight']
                    # create new OrderedDict that does not contain `module.`
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace('module.', '')  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.fusion.load_state_dict(new_state_dict)
            except:
                model = self.fusion
                print("weights not same ,try to load part of them")
                model_dict = model.state_dict()
                pretrained = torch.load(self.args.weights_path)['weight']
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in model_dict.items() if k in pretrained}
                left_dict = {k for k, v in model_dict.items() if k not in pretrained}
                print(left_dict)
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict)
                print(len(model_dict), len(pretrained_dict))
                # model_dict = self.fusion.state_dict()
                # pretrained_dict = {k: v for k, v in model_dict.items() if k in checkpoint['weight'] }
                # print(len(checkpoint['weight'].items()), len(pretrained_dict), len(model_dict))
                # model_dict.update(pretrained_dict)
                # self.fusion.load_state_dict(model_dict)
            print("start_epoch:", start_epoch)
            print("=========END LOAD WEIGHTS=========")
        print("========START EPOCH: %d=========" % start_epoch)
        self.start_epoch = start_epoch

    def forward(self):
        self.img_re,self.grad_map_re = self.fusion(self.img)

    def backward(self):
        img = self.img
        img_re = self.img_re
        grad_map_re = self.grad_map_re
        # 计算ssim损失
        ssim_loss = 1 - self.SSIM_fun(img_re, img)
        # 计算像素损失
        pixel_loss = self.MSE_fun(img_re, img)
        # 计算梯度损失
        grad_map = gradient_canny(img)
        self.grad_map = grad_map
        # grad_map = torch.repeat_interleave(grad_map,repeats=32,dim=1)
        grd_loss = self.MSE_fun(grad_map_re, grad_map)
        # 损失求和 回传
        loss = self.loss_hyperparameters[0]*pixel_loss + self.loss_hyperparameters[1]*ssim_loss + self.loss_hyperparameters[2]*grd_loss
        self.optimizer_G.zero_grad()

        loss.backward()
        self.loss = loss
        self.ssim_loss = ssim_loss
        self.pixel_loss = pixel_loss
        self.Gloss = grd_loss
        # self.Ploss = perceptual_loss


    def mulgpus(self):
        self.fusion = nn.DataParallel(self.fusion.cuda(), device_ids=self.args.GPUs, output_device=self.args.GPUs[0])
        self.D = nn.DataParallel(self.D.cuda(), device_ids=self.args.GPUs, output_device=self.args.GPUs[0])

    def setdata(self, img):
        img = img.to(self.args.device)
        self.img = img

    def step(self):
        self.forward()
        self.backward()  # calculate gradients for G
        self.optimizer_G.step()  # update G weights
        self.count_batch += 1
        self.print = 'Loss: ALL[%.5lf]mean[%.5f] { pixel[%.5lf]ssim[%.5lf]Gloss[%.5lf]}' % \
                     (self.loss.item(),
                      self.mean_loss / self.count_batch,
                      self.pixel_loss.item()*self.loss_hyperparameters[0],
                      self.ssim_loss.item()*self.loss_hyperparameters[1],
                      self.Gloss.item()*self.loss_hyperparameters[2])
        self.mean_loss += self.loss.item()

    def saveimg(self, epoch, num=0):
        img1 = self.img[0].cpu()
        img2 = self.img_re[0].cpu()
        img3 = self.grad_map_re[0].cpu()
        img4 = self.grad_map[0].cpu()
        img = torchvision.utils.make_grid([img1, img2, img3,img4], nrow=4)
        torchvision.utils.save_image(img,
                                     fp=(os.path.join('output/output_guide_decoder/result_%d_%d.jpg' % (epoch, num))))

    # def saveimgfuse(self,name=''):
    #     # self.img_down = self.downsample(self.img)
    #     # self.img_g = gradient(self.img)
    #
    #     img = torchvision.utils.make_grid(
    #         [self.img[0].cpu(), self.img_g[0].cpu(), ((self.g1+self.g2+self.g3)*1.5)[0].cpu()], nrow=3)
    #     torchvision.utils.save_image(img, fp=(os.path.join(name.replace('Test','demo'))))
    #     # torchvision.utils.save_image(img, fp=(os.path.join('output/epoch/'+str(num)+'.jpg')))

    def save(self, epoch):
        ## 保存模型和最佳模型
        self.mean_loss = self.mean_loss / self.count_batch
        if self.min_loss > self.mean_loss:
            self.min_loss = self.mean_loss
            torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_fusion.pt'))
            # torch.save({'weight': self.D.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_D.pt'))
            print('[%d] - Best model is saved -' % (epoch))

        if epoch % 1 == 0:
            torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                       os.path.join('weights/epoch' + str(epoch) + '_fusion.pt'))
            # torch.save({'weight': self.D.state_dict(), 'epoch': epoch, },os.path.join('weights/epoch' + str(epoch) + '_D.pt'))
        self.mean_loss = 0
        self.count_batch = 0
    # def getimg(self):
    #     return self.g1, self.g2,self.g3,self.s

# 使用gan的方式训练融合策略，效果不太好

# class fuse_model(nn.Module):
#     def __init__(self,args):
#         super(fuse_model, self).__init__()
#         self.fusion = Fusion_network()
#         self.fuse_Layer = fusion_layer()
#         self.args = args
#         self.loss_detail = torch.zeros(1)
#         self.loss_feature = torch.zeros(1)
#         self.total_loss = torch.zeros(1)
#         self.loss_grad = torch.zeros(1)
#         self.mse_fun = nn.MSELoss()
#         self.ssim = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1, K=(0.01,0.03))
#         self.min_loss = 10000
#         self.optimizer_G = optim.Adam(self.fuse_Layer.parameters(), lr=args.lr)
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.5,
#                                                                     patience=2,
#                                                                     verbose=False, threshold=0.0001,
#                                                                     threshold_mode='rel',
#                                                                     cooldown=0, min_lr=0, eps=1e-10)
#         # [detail,feature,grad]
#         self.loss_hyperparameters = [1000,1,10]
#
#     def load(self):
#         print("=========LOAD WEIGHTS=========")
#         print(self.args.weights_path)
#         checkpoint = torch.load(self.args.weights_path)
#         self.fusion.load_state_dict(checkpoint['weight'])
#         self.Encoder = self.fusion.encoder
#         self.Decoder = self.fusion.decoder
#         self.grad_branch = self.fusion.grad_branch
#         if self.args.load_fusion_layer:
#             checkpoint = torch.load(self.args.fusion_layer_path)
#             self.fuse_Layer.load_state_dict(checkpoint['weight'])
#
#     def setdata(self, vi_img,ir_img):
#         vi_img = vi_img.to(self.args.device)
#         ir_img = ir_img.to(self.args.device)
#         self.vi = vi_img
#         self.ir = ir_img
#
#     # 给bn*c*128*128的两组图片
#     def forward(self):
#         grad_vi_map = gradient(self.vi)
#         grad_ir_map = gradient(self.ir)
#         fmp_vi,guide_vi = self.Encoder(self.vi,grad_vi_map)
#         fmp_ir,guide_ir = self.Encoder(self.ir,grad_ir_map)
#         fmp = self.fuse_Layer(fmp_vi,fmp_ir)
#         self.loss_feature = self.mse_fun(fmp, 10*fmp_ir+5*fmp_vi)
#         guide = (guide_vi+guide_ir)/2
#         self.img_re = self.Decoder(fmp, guide, grad_vi_map)
#         self.grad_map = self.grad_branch(fmp)
#         self.loss_grad = self.mse_fun(self.grad_map,gradient(self.vi))+self.mse_fun(self.grad_map,gradient(self.ir))
#
#
#     def backward(self):
#         self.loss_detail = 1-self.ssim(self.img_re, self.vi)
#         loss = self.loss_hyperparameters[0]*self.loss_detail +\
#                self.loss_hyperparameters[1]*self.loss_feature + \
#                self.loss_hyperparameters[2]*self.loss_grad
#         self.optimizer_G.zero_grad()
#         loss.backward()
#         self.total_loss = loss
#
#     def step(self):
#         self.forward()
#         self.backward()  # calculate gradients for G
#         self.optimizer_G.step()  # update G weights
#         self.print = 'Loss: ALL[%.5lf] { detail[%.5lf]feature[%.5lf]grad[%.5lf]}' % \
#                      (self.total_loss.item(),
#                       self.loss_hyperparameters[0]*self.loss_detail.item(),
#                       self.loss_hyperparameters[1]*self.loss_feature.item(),
#                       self.loss_hyperparameters[2]*self.loss_grad.item())
#
#     def saveimg(self, epoch, num=0):
#         img1 = self.vi[0].cpu()
#         img2 = self.ir[0].cpu()
#         img3 = self.img_re[0].cpu()
#         img4 = self.grad_map[0].cpu()
#         img = torchvision.utils.make_grid([img1, img2, img3,img4], nrow=4)
#         torchvision.utils.save_image(img,
#                                      fp=(os.path.join('output/train_fuse/result_%d_%d.jpg' % (epoch, num))))
#     def save(self, epoch):
#         ## 保存模型和最佳模型
#         if self.min_loss > self.total_loss.item():
#             self.min_loss = self.total_loss.item()
#             torch.save({'weight': self.fuse_Layer.state_dict(), 'epoch': epoch, }, os.path.join('weights/fusion_layer/best_layer.pt'))
#             # torch.save({'weight': self.D.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_D.pt'))
#             print('[%d] - Best model is saved -' % (epoch))
#         if epoch % 1 == 0:
#             torch.save({'weight': self.fuse_Layer.state_dict(), 'epoch': epoch, },
#                        os.path.join('weights/fusion_layer/epoch' + str(epoch) + '_layer.pt'))
#             # torch.save({'weight': self.D.state_dict(), 'epoch': epoch, },os.path.join('weights/epoch' + str(epoch) + '_D.pt'))
