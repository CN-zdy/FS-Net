from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear = False, mode = None):
        super(up_conv, self).__init__()
        if mode == 'segmentation':
            if bilinear:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True))
            else:
                self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

        elif mode == 'fusion':
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(ch_in // 2, ch_in // 2, 2, stride=2)


    def forward(self, x):

        x = self.up(x)

        return x

class FS_Net(nn.Module):
    def __init__(self, n_channels=1, n_classes=2 ,img_ch=1, output_ch=1):
        super(FS_Net, self).__init__()

        filter_nums = [32, 64, 128, 256, 512]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#------------------------------------------        Fusion Network        -----------------------------------------------#
        self.f_conv_1 = conv_block(ch_in=n_classes*img_ch, ch_out=filter_nums[0])
        self.f_conv_2 = conv_block(ch_in=filter_nums[0], ch_out=filter_nums[1])
        self.f_conv_3 = conv_block(ch_in=filter_nums[1], ch_out=filter_nums[2])
        self.f_conv_4 = conv_block(ch_in=filter_nums[2], ch_out=filter_nums[3])
        self.f_conv_5 = conv_block(ch_in=filter_nums[3], ch_out=filter_nums[3])

        self.f_up_5 = up_conv(ch_in=filter_nums[4], ch_out=filter_nums[2], mode = 'fusion')
        self.f_up_conv_5 = conv_block(ch_in=filter_nums[4], ch_out=filter_nums[2])

        self.f_up_4 = up_conv(ch_in=filter_nums[3], ch_out=filter_nums[1], mode = 'fusion')
        self.f_up_conv_4 = conv_block(ch_in=filter_nums[3], ch_out=filter_nums[1])

        self.f_up_3 = up_conv(ch_in=filter_nums[2], ch_out=filter_nums[0], mode = 'fusion')
        self.f_up_conv_3 = conv_block(ch_in=filter_nums[2], ch_out=filter_nums[0])

        self.f_up_2 = up_conv(ch_in=filter_nums[1], ch_out=filter_nums[0], mode = 'fusion')
        self.f_up_conv_2 = conv_block(ch_in=filter_nums[1], ch_out=filter_nums[0])

        self.f_conv_11 = nn.Conv2d(filter_nums[0], n_channels, kernel_size=1, stride=1, padding=0, bias=True)

# --------------------------------------        Reconstruction Network        ------------------------------------------#

        self.r_conv_1 = conv_block(ch_in=n_channels, ch_out=filter_nums[0])
        self.r_conv_2 = conv_block(ch_in=filter_nums[0], ch_out=filter_nums[1])
        self.r_conv_3 = conv_block(ch_in=filter_nums[1], ch_out=filter_nums[2])
        self.r_conv_4 = conv_block(ch_in=filter_nums[2], ch_out=filter_nums[3])
        self.r_conv_5 = conv_block(ch_in=filter_nums[3], ch_out=filter_nums[3])

        self.r_up_5 = up_conv(ch_in=filter_nums[4], ch_out=filter_nums[2], mode = 'fusion')
        self.r_up_conv_5 = conv_block(ch_in=filter_nums[4], ch_out=filter_nums[2])

        self.r_up_4 = up_conv(ch_in=filter_nums[3], ch_out=filter_nums[1], mode = 'fusion')
        self.r_up_conv_4 = conv_block(ch_in=filter_nums[3], ch_out=filter_nums[1])

        self.r_up_3 = up_conv(ch_in=filter_nums[2], ch_out=filter_nums[0], mode = 'fusion')
        self.r_up_conv_3 = conv_block(ch_in=filter_nums[2], ch_out=filter_nums[0])

        self.r_up_2 = up_conv(ch_in=filter_nums[1], ch_out=filter_nums[0], mode = 'fusion')
        self.r_up_conv_2 = conv_block(ch_in=filter_nums[1], ch_out=filter_nums[0])

        self.r_conv_11 = nn.Conv2d(filter_nums[0], n_classes, kernel_size=1, stride=1, padding=0, bias=True)

# --------------------------------------        Segmentation Network        --------------------------------------------#

        self.s_conv_1 = conv_block(ch_in=img_ch, ch_out=filter_nums[0])
        self.s_conv_2 = conv_block(ch_in=filter_nums[0], ch_out=filter_nums[1])
        self.s_conv_3 = conv_block(ch_in=filter_nums[1], ch_out=filter_nums[2])
        self.s_conv_4 = conv_block(ch_in=filter_nums[2], ch_out=filter_nums[3])
        self.s_conv_5 = conv_block(ch_in=filter_nums[3], ch_out=filter_nums[4])

        self.s_conv_6 = conv_2(ch_in=img_ch, ch_out=filter_nums[0])
        self.s_conv_7 = conv_2(ch_in=filter_nums[0], ch_out=filter_nums[1])
        self.s_conv_8 = conv_2(ch_in=filter_nums[1], ch_out=filter_nums[2])
        self.s_conv_9 = conv_2(ch_in=filter_nums[2], ch_out=filter_nums[3])
        self.s_conv_10 = conv_2(ch_in=filter_nums[3], ch_out=filter_nums[4])

        self.s_conv_1x1_1 = nn.Conv2d(filter_nums[1], filter_nums[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.s_conv_1x1_2 = nn.Conv2d(filter_nums[2], filter_nums[1], kernel_size=1, stride=1, padding=0, bias=True)
        self.s_conv_1x1_3 = nn.Conv2d(filter_nums[3], filter_nums[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.s_conv_1x1_4 = nn.Conv2d(filter_nums[4], filter_nums[3], kernel_size=1, stride=1, padding=0, bias=True)
        self.s_conv_1x1_5 = nn.Conv2d(filter_nums[4], filter_nums[3], kernel_size=1, stride=1, padding=0, bias=True)

        self.s_up5 = up_conv(ch_in=filter_nums[4], ch_out=filter_nums[3], mode = 'segmentation')
        self.s_up_conv5 = conv_block(ch_in=filter_nums[4], ch_out=filter_nums[3])

        self.s_up4 = up_conv(ch_in=filter_nums[3], ch_out=filter_nums[2], mode = 'segmentation')
        self.s_up_conv4 = conv_block(ch_in=filter_nums[3], ch_out=filter_nums[2])

        self.s_up3 = up_conv(ch_in=filter_nums[2], ch_out=filter_nums[1], mode = 'segmentation')
        self.s_up_conv3 = conv_block(ch_in=filter_nums[2], ch_out=filter_nums[1])

        self.s_up2 = up_conv(ch_in=filter_nums[1], ch_out=filter_nums[0], mode = 'segmentation')
        self.s_up_conv2 = conv_block(ch_in=filter_nums[1], ch_out=filter_nums[0])

        self.s_conv_11 = nn.Conv2d(filter_nums[0], output_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        ##-----fusion-----##

        fx1 = self.f_conv_1(x)
        
        fx2 = self.Maxpool(fx1)
        fx2 = self.f_conv_2(fx2)
        
        fx3 = self.Maxpool(fx2)
        fx3 = self.f_conv_3(fx3)

        fx4 = self.Maxpool(fx3)
        fx4 = self.f_conv_4(fx4)

        fx5 = self.Maxpool(fx4)
        fx5 = self.f_conv_5(fx5)

        fd5 = self.f_up_5(fx5)
        fd5 = torch.cat((fx4, fd5), dim=1)
        fd5 = self.f_up_conv_5(fd5)

        fd4 = self.f_up_4(fd5)
        fd4 = torch.cat((fx3, fd4), dim=1)
        fd4 = self.f_up_conv_4(fd4)

        fd3 = self.f_up_3(fd4)
        fd3 = torch.cat((fx2, fd3), dim=1)
        fd3 = self.f_up_conv_3(fd3)

        fd2 = self.f_up_2(fd3)
        fd2 = torch.cat((fx1, fd2), dim=1)
        fd2 = self.f_up_conv_2(fd2)

        fd1 = self.f_conv_11(fd2)
        fr = F.sigmoid(fd1)

        ##-----Reconstruction-----##
        rx1 = self.r_conv_1(fr)

        rx2 = self.Maxpool(rx1)
        rx2 = self.r_conv_2(rx2)

        rx3 = self.Maxpool(rx2)
        rx3 = self.r_conv_3(rx3)

        rx4 = self.Maxpool(rx3)
        rx4 = self.r_conv_4(rx4)

        rx5 = self.Maxpool(rx4)
        rx5 = self.r_conv_5(rx5)

        rd5 = self.r_up_5(rx5)
        rd5 = torch.cat((rx4, rd5), dim=1)
        rd5 = self.r_up_conv_5(rd5)

        rd4 = self.r_up_4(rd5)
        rd4 = torch.cat((rx3, rd4), dim=1)
        rd4 = self.r_up_conv_4(rd4)

        rd3 = self.r_up_3(rd4)
        rd3 = torch.cat((rx2, rd3), dim=1)
        rd3 = self.r_up_conv_3(rd3)

        rd2 = self.r_up_2(rd3)
        rd2 = torch.cat((rx1, rd2), dim=1)
        rd2 = self.r_up_conv_2(rd2)

        rd1 = self.r_conv_11(rd2)

        ##-----segmentation-----##
        sx1 = self.s_conv_1(fr)
        sx6 = self.s_conv_6(fr)

        sx2 = self.Maxpool(sx1)
        sx2 = self.s_conv_2(sx2)

        sx3 = self.Maxpool(sx2)
        sx3 = self.s_conv_3(sx3)

        sx4 = self.Maxpool(sx3)
        sx4 = self.s_conv_4(sx4)

        sx5 = self.Maxpool(sx4)
        sx5 = self.s_conv_5(sx5)

        sx7 = self.Maxpool(sx6)
        sx7 = self.s_conv_7(sx7)

        sx8 = self.Maxpool(sx7)
        sx8 = self.s_conv_8(sx8)

        sx9 = self.Maxpool(sx8)
        sx9 = self.s_conv_9(sx9)

        sx10 = self.Maxpool(sx9)
        sx10 = self.s_conv_10(sx10)

        # Conv 1X1
        sx_5 = self.s_conv_1x1_5(sx5)
        sx_10 = self.s_conv_1x1_5(sx10)
        sx510 = torch.cat((sx_5, sx_10), dim=1)

        # decoding + concat path
        sd5 = self.s_up5(sx510)
        sx4 = torch.cat((sx4, sd5), dim=1)
        sx9 = torch.cat((sx9, sd5), dim=1)
        sx4 = self.s_conv_1x1_4(sx4)
        sx9 = self.s_conv_1x1_4(sx9)
        sx49 = torch.cat((sx4, sx9), dim=1)
        sd5 = self.s_up_conv5(sx49)

        sd4 = self.s_up4(sd5)
        sx3 = torch.cat((sx3, sd4), dim=1)
        sx8 = torch.cat((sx8, sd4), dim=1)
        sx3 = self.s_conv_1x1_3(sx3)
        sx8 = self.s_conv_1x1_3(sx8)
        sx38 = torch.cat((sx3, sx8), dim=1)
        sd4 = self.s_up_conv4(sx38)

        sd3 = self.s_up3(sd4)
        sx2 = torch.cat((sx2, sd3), dim=1)
        sx7 = torch.cat((sx7, sd3), dim=1)
        sx2 = self.s_conv_1x1_2(sx2)
        sx7 = self.s_conv_1x1_2(sx7)
        sx27 = torch.cat((sx2, sx7), dim=1)
        sd3 = self.s_up_conv3(sx27)

        sd2 = self.s_up2(sd3)
        sx1 = torch.cat((sx1, sd2), dim=1)
        sx6 = torch.cat((sx6, sd2), dim=1)
        sx1 = self.s_conv_1x1_1(sx1)
        sx6 = self.s_conv_1x1_1(sx6)
        sx16 = torch.cat((sx1, sx6), dim=1)
        sd2 = self.s_up_conv2(sx16)

        sd1 = self.s_conv_11(sd2)

        return sd1, fr, F.sigmoid(rd1)