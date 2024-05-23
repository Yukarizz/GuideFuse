import torch.nn as nn
from net_utils import conv_norm_lrelu, res_conv_norm_lrelu, conv1x1
import torch
from utils import gradient,gradient_sobel,gradient_Isotropic_sobel,gradient_Prewitt,gradient_canny
from net_utils import res_for_guider

class Encoder(nn.Module):
    def __init__(self, mode="train"):
        super(Encoder, self).__init__()
        channel_0 = 16
        # before input x is 1x256x256
        # Conv input
        self.conv_input = res_conv_norm_lrelu(1, channel_0, kernel_size=(7, 7), padding=(3, 3))  # 16 x 256 x 256
        # base brach
        self.conv0 = res_conv_norm_lrelu(channel_0, channel_0 * 2)  # 32 x 256 x 256
        self.conv1 = res_conv_norm_lrelu(channel_0 * 2, channel_0 * 4)  # 64 x 256 x 256
        self.conv2 = res_conv_norm_lrelu(channel_0 * 4, channel_0 * 8)  # 128 x 256 x 256

        # guide branch
        self.guide_brach16 = get_guidevalues(channel_0)  # 输入16通道特征图，输出16*16=256通道的GV

        # guide to base
        # self.guide2base0 = guide_base(guide_channel=channel_0 * 16, x_channel=channel_0 * 2)
        # self.guide2base1 = guide_base(guide_channel=channel_0 * 16, x_channel=channel_0 * 4)
        self.guide2base2 = guide_base(guide_channel=channel_0 * 16, x_channel=channel_0 * 8)
        self.Mode = mode

    def forward(self, x):
        # ========================================
        # 第一层
        x = self.conv_input(x)
        # print(x.shape)

        # ========================================
        # guide branch
        guide_values16 = self.guide_brach16(x)

        # print(guide_values.shape)

        # ========================================
        # base brach + guide blocks
        x = self.conv0(x)
        # x = self.guide2base0(x, guide_values16)
        x = self.conv1(x)
        # x = self.guide2base1(x, guide_values16)
        x = self.conv2(x)
        if(self.Mode=='train'):
            x = self.guide2base2(x, guide_values16)
        return x, guide_values16


# Encoder 的输出是128x256x256
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        channel_0 = 16
        # Conv out
        self.conv_input = res_conv_norm_lrelu(channel_0 * 8+2 , 64)  # 128-64-32-16-1
        self.conv1 = res_conv_norm_lrelu(64+2, 32)  # 64,32,16进行guide
        self.conv2 = res_conv_norm_lrelu(32+2, 16)
        self.conv3 = res_conv_norm_lrelu(16, 1)

        self.grad_pred_1 = conv1x1(2, 2)
        self.grad_pred_2 = conv1x1(2, 2)
        self.grad_pred_3 = conv1x1(2, 2)

        self.guide2base1 = guide_base(guide_channel=16 * 16, x_channel=channel_0 * 4)
        self.guide2base2 = guide_base(guide_channel=16 * 16, x_channel=channel_0 * 2)
        self.guide2base3 = guide_base(guide_channel=16 * 16, x_channel=channel_0 * 1)

    def forward(self, x, G16, grad_map):
        # x.shape[N,128,256,256],conv_output.shape[n,16,256,256]

        x = torch.cat([x, self.grad_pred_1(grad_map)], dim=1)
        x = self.conv_input(x)
        x = self.guide2base1(x, G16)


        x = torch.cat([x, self.grad_pred_2(grad_map)], dim=1)
        x = self.conv1(x)

        x = self.guide2base2(x, G16)

        x = torch.cat([x, self.grad_pred_3(grad_map)], dim=1)
        x = self.conv2(x)

        x = self.guide2base3(x, G16)

        x = self.conv3(x)
        return x


class Fusion_network(nn.Module):
    def __init__(self, mode="train"):
        super(Fusion_network, self).__init__()
        self.encoder = Encoder(mode)
        self.decoder = Decoder()
        self.grad_branch = Gradient_Branch()
    def forward(self, x):
        grad_map = gradient_canny(x)
        grad_map = torch.repeat_interleave(grad_map,2,dim=1)
        x, G16 = self.encoder(x)
        grad_map_re = self.grad_branch(x)
        x = self.decoder(x, G16, grad_map)
        return x, grad_map_re


# 输入bn*128*256*256
class Gradient_Branch(nn.Module):
    def __init__(self):
        super(Gradient_Branch, self).__init__()
        self.net = nn.Sequential(conv1x1(128,10),
                                 conv_norm_lrelu(10,1)
                                )

        # self.net = nn.Sequential(nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,groups=32),
        #                          nn.BatchNorm2d(32),
        #                          nn.LeakyReLU(0.2),
        #                          conv1x1(32,32))
    def forward(self, x):
        return self.net(x)

from net_utils import SSEBlock
class get_guidevalues(nn.Module):
    def __init__(self, channel_0=16):
        super(get_guidevalues, self).__init__()

        self.guidebrach = nn.Sequential(
            res_for_guider(channel_0, channel_0 * 2, normtype='bn',mode='down'),  # 16*256*256-32 x 128 x 128
            res_for_guider(channel_0 * 2, channel_0 * 4, normtype='bn',mode='down'),  # 64 x 64 x 64
            res_for_guider(channel_0 * 4, channel_0 * 8,normtype='bn', mode='down'),  # 128 x 32 x 32
            res_for_guider(channel_0 * 8, channel_0 * 16, normtype='bn',mode='down'),  # 256 x 16 x 16
            )
        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))  # 512x 1 x 1
        # self.maxpooling = nn.AdaptiveMaxPool2d((1, 1))
    def forward(self, x):
        # bs x  16 x 128 x 128
        x = self.guidebrach(x)  # bs x 256x 16 x 16
        x = self.globalpooling(x)
        return x


class guide_base(nn.Module):
    def __init__(self, guide_channel=16, x_channel=16):
        super(guide_base, self).__init__()
        self.x_channel = x_channel
        self.fc = nn.Sequential(
            nn.Linear(guide_channel, guide_channel//2),
            nn.BatchNorm1d(guide_channel//2),
            nn.LeakyReLU(0.2),
            nn.Linear(guide_channel//2,x_channel),
            nn.Sigmoid()
        )

    def forward(self, x_base, y_guide):
        # guide维度高，进行压缩
        # print(y_guide.shape)  # bs x 256 x 1 x 1
        y_guide = y_guide.view((y_guide.shape[0], y_guide.shape[1]))  # bs x 256
        y_guide = self.fc(y_guide)  # bs x x_channel
        y_guide = y_guide.view((y_guide.shape[0], y_guide.shape[1], 1, 1))  # bs x x_channel x 1 x 1
        # print(y_guide.shape)
        return x_base + x_base * y_guide





# 以下是gan的融合策略，效果不好暂不使用
# 把两组bn*128*256*256的数据合并成一组
# class fusion_layer(nn.Module):
#     def __init__(self):
#         super(fusion_layer, self).__init__()
#         self.conv1 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
#         self.conv2 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
#         self.activation = nn.GELU()
#         self.network = nn.Sequential(res_conv_norm_lrelu(256,256),
#                                      res_conv_norm_lrelu(256,128))
#
#     def forward(self, vi, ir):
#         vi = self.activation(self.conv1(vi))
#         ir = self.activation(self.conv2(ir))
#         input = torch.cat([vi, ir], dim=1)
#         output = self.network(input)
#         return output

#     输入bn*1*256*256




# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.network = nn.Sequential(
#             # 1*256*256->16*128*128
#             # conv_norm_lrelu(1, 16, mode='down'),
#             # # 16*128*128->32*64*64
#             # conv_norm_lrelu(16, 32, mode='down'),
#             # # 32*64*64->64*32*32
#             # conv_norm_lrelu(32, 64, mode='down'),
#             nn.Conv2d(1,16,3,2,1),
#             nn.Conv2d(16, 32, 3, 2, 1),
#             nn.Conv2d(32,64,3,2,1),
#             nn.Flatten(),
#             nn.Linear(64 * 32 * 32, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, 1),
#             nn.Tanh()
#             )
#
#     def forward(self, x):
#         return self.network(x)
