import torch
import torch.nn as nn
import functools
from torch.nn import init
import torch.nn.functional as f

class DwConv(nn.Module):
    def __init__(self, input_nc, output_nc, innermost=False, outermost=False, norm_layer=nn.BatchNorm2d):
        super(DwConv, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        dwconv=nn.Conv2d(input_nc, output_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(output_nc)
        if outermost:
            dconv = [dwconv]
        elif innermost:
            dconv = [downrelu, dwconv]
        else:
            dconv = [downrelu, dwconv, downnorm]

        self.dconv = nn.Sequential(*dconv)

    def forward(self, x):
        return self.dconv(x)

class UpConv(nn.Module):
    def __init__(self, input_nc, output_nc, innermost=False, outermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UpConv, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        uprelu = nn.ReLU()
        upnorm = norm_layer(output_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(input_nc * 2, output_nc,
                            kernel_size=4, stride=2,
                            padding=1)
            uconv = [uprelu, upconv]
        elif innermost:
            upconv = nn.ConvTranspose2d(input_nc, output_nc,
                            kernel_size=4, stride=2,
                            padding=1)
            uconv = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(input_nc * 2, output_nc,
                            kernel_size=4, stride=2,
                            padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                uconv=up + [nn.Dropout(0.5)]
            else:
                uconv=up

        self.uconv=nn.Sequential(*uconv)

    def forward(self,x):
        return self.uconv(x)


class SpatialAttn(nn.Module):
    def __init__(self, input_channels=5, bias=True):
        super(SpatialAttn, self).__init__()
        self.conv1=nn.Conv2d(input_channels, 64, 4, 2, 1, bias=bias)
        self.conv2=nn.Conv2d(64, 1, 4, 2, 1, bias=bias)
        self.mpool=nn.AvgPool2d(4,2,1)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,inputs):
        # print inputs.shape
        y = self.conv1(inputs)
        y = self.conv2(y)
        y = self.mpool(y)
        # print y.shape
        n,c,h,w= y.shape
        y = self.softmax(y.view(n, -1)).view(n, 1, h, w)
        y = f.upsample(y,(inputs.shape[-2], inputs.shape[-1]))
        return y

class UNetMatsm(nn.Module):

    def __init__(self, input_ch, output_ch, norm_layer=nn.BatchNorm2d, use_dropout=False, num_lights=8):
        super().__init__()
                
        self.conv_down1 = DwConv(input_nc=input_ch, output_nc=64, innermost=False, outermost=True, norm_layer=nn.BatchNorm2d)
        self.conv_down2 = DwConv(input_nc=64, output_nc=128, norm_layer=nn.BatchNorm2d)
        self.conv_down3 = DwConv(input_nc=128, output_nc=256, norm_layer=nn.BatchNorm2d)
        self.conv_down4 = DwConv(input_nc=256, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down5 = DwConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down6 = DwConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down7 = DwConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)             
        
        self.sconv_up7 = UpConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)
        self.sconv_up6 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up5 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up4 = UpConv(input_nc=512, output_nc=256, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up3 = UpConv(input_nc=256, output_nc=128, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up2 = UpConv(input_nc=128, output_nc=64, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up1 = UpConv(input_nc=64, output_nc=output_ch, outermost=True, norm_layer=nn.BatchNorm2d)

        self.mconv_up7 = UpConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)
        self.mconv_up6 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up5 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up4 = UpConv(input_nc=512, output_nc=256, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up3 = UpConv(input_nc=256, output_nc=128, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up2 = UpConv(input_nc=128, output_nc=64, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up1 = UpConv(input_nc=64, output_nc=output_ch, outermost=True, norm_layer=nn.BatchNorm2d)

        self.htan=nn.Hardtanh(0,1)
        self.htan100=nn.Hardtanh(0,100)

        self.mat_confmap=nn.Sequential(nn.Conv2d(64, 1, kernel_size=3,stride=1, padding=1),
                                       nn.Sigmoid())
        self.mat_upsample2=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.mat_upsample4=nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.mat_final= nn.Conv2d(4, 3, kernel_size=4,stride=2, padding=1)

        
    def encoder(self, x):
        d1=self.conv_down1(x)
        d2=self.conv_down2(d1)
        d3=self.conv_down3(d2)
        d4=self.conv_down4(d3)
        d5=self.conv_down5(d4)
        d6=self.conv_down6(d5)
        d7=self.conv_down7(d6)
        return [d1,d2,d3,d4,d5,d6,d7] 

    def shd_decoder(self,encoded):
        x= encoded
        u7=self.sconv_up7(x[6])
        u7 = torch.cat([u7, x[5]], dim=1)
        u6=self.sconv_up6(u7)
        u6 = torch.cat([u6, x[4]], dim=1)
        u5=self.sconv_up5(u6)
        u5 = torch.cat([u5, x[3]], dim=1)
        u4=self.sconv_up4(u5)
        u4 = torch.cat([u4, x[2]], dim=1)
        u3=self.sconv_up3(u4)
        u3 = torch.cat([u3, x[1]], dim=1)
        u2=self.sconv_up2(u3) 
        u1 = torch.cat([u2, x[0]], dim=1)
        out=self.sconv_up1(u1)
        return out 

    def mat_decoder(self,encoded):
        x= encoded
        u7=self.mconv_up7(x[6])
        u7 = torch.cat([u7, x[5]], dim=1)
        u6=self.mconv_up6(u7)
        u6 = torch.cat([u6, x[4]], dim=1)
        u5=self.mconv_up5(u6)
        u5 = torch.cat([u5, x[3]], dim=1)
        u4=self.mconv_up4(u5)
        u4 = torch.cat([u4, x[2]], dim=1)
        u3=self.mconv_up3(u4)
        u3 = torch.cat([u3, x[1]], dim=1)
        u2=self.mconv_up2(u3) 
        u1 = torch.cat([u2, x[0]], dim=1)
        out1=self.mconv_up1(u1)
        conf=self.mat_confmap(u2)
        up4=self.mat_upsample4(conf)
        up2=self.mat_upsample2(out1)
        out=self.mat_final(torch.cat([up2, up4],dim=1))
        return out 
    
    def forward(self,x):
        e=self.encoder(x)
        shd=self.htan100(self.shd_decoder(e))
        mat=self.htan(self.mat_decoder(e))
    
        return {'shd':shd, 'mat':mat}

class UNetMatsm255(nn.Module):

    def __init__(self, input_ch, output_ch, norm_layer=nn.BatchNorm2d, use_dropout=False, num_lights=8):
        super().__init__()
                
        self.conv_down1 = DwConv(input_nc=input_ch, output_nc=64, innermost=False, outermost=True, norm_layer=nn.BatchNorm2d)
        self.conv_down2 = DwConv(input_nc=64, output_nc=128, norm_layer=nn.BatchNorm2d)
        self.conv_down3 = DwConv(input_nc=128, output_nc=256, norm_layer=nn.BatchNorm2d)
        self.conv_down4 = DwConv(input_nc=256, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down5 = DwConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down6 = DwConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down7 = DwConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)             
        
        self.sconv_up7 = UpConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)
        self.sconv_up6 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up5 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up4 = UpConv(input_nc=512, output_nc=256, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up3 = UpConv(input_nc=256, output_nc=128, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up2 = UpConv(input_nc=128, output_nc=64, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up1 = UpConv(input_nc=64, output_nc=output_ch, outermost=True, norm_layer=nn.BatchNorm2d)

        self.mconv_up7 = UpConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)
        self.mconv_up6 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up5 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up4 = UpConv(input_nc=512, output_nc=256, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up3 = UpConv(input_nc=256, output_nc=128, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up2 = UpConv(input_nc=128, output_nc=64, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up1 = UpConv(input_nc=64, output_nc=output_ch, outermost=True, norm_layer=nn.BatchNorm2d)

        self.htan=nn.Hardtanh(0,1)
        self.htan255=nn.Hardtanh(0,255)

        self.mat_confmap=nn.Sequential(nn.Conv2d(64, 1, kernel_size=3,stride=1, padding=1),
                                       nn.Sigmoid())
        self.mat_upsample2=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.mat_upsample4=nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.mat_final= nn.Conv2d(4, 3, kernel_size=4,stride=2, padding=1)
        
    def encoder(self, x):
        d1=self.conv_down1(x)
        d2=self.conv_down2(d1)
        d3=self.conv_down3(d2)
        d4=self.conv_down4(d3)
        d5=self.conv_down5(d4)
        d6=self.conv_down6(d5)
        d7=self.conv_down7(d6)
        return [d1,d2,d3,d4,d5,d6,d7] 

    def shd_decoder(self,encoded):
        x= encoded
        u7=self.sconv_up7(x[6])
        u7 = torch.cat([u7, x[5]], dim=1)
        u6=self.sconv_up6(u7)
        u6 = torch.cat([u6, x[4]], dim=1)
        u5=self.sconv_up5(u6)
        u5 = torch.cat([u5, x[3]], dim=1)
        u4=self.sconv_up4(u5)
        u4 = torch.cat([u4, x[2]], dim=1)
        u3=self.sconv_up3(u4)
        u3 = torch.cat([u3, x[1]], dim=1)
        u2=self.sconv_up2(u3) 
        u1 = torch.cat([u2, x[0]], dim=1)
        out=self.sconv_up1(u1)
        return out 


    def mat_decoder(self,encoded):
        x= encoded
        u7=self.mconv_up7(x[6])
        u7 = torch.cat([u7, x[5]], dim=1)
        u6=self.mconv_up6(u7)
        u6 = torch.cat([u6, x[4]], dim=1)
        u5=self.mconv_up5(u6)
        u5 = torch.cat([u5, x[3]], dim=1)
        u4=self.mconv_up4(u5)
        u4 = torch.cat([u4, x[2]], dim=1)
        u3=self.mconv_up3(u4)
        u3 = torch.cat([u3, x[1]], dim=1)
        u2=self.mconv_up2(u3) 
        u1 = torch.cat([u2, x[0]], dim=1)
        out1=self.mconv_up1(u1)
        conf=self.mat_confmap(u2)
        up4=self.mat_upsample4(conf)
        up2=self.mat_upsample2(out1)
        out=self.mat_final(torch.cat([up2, up4],dim=1))
        return out 
    
    def forward(self,x):
        e=self.encoder(x)
        shd=self.htan255(self.shd_decoder(e))
        mat=self.htan(self.mat_decoder(e))
    
        return {'shd':shd, 'mat':mat}



class UNetMatsm255nsf(nn.Module):
    # this Net removes the Skip connection between the First and the last layer
    def __init__(self, input_ch, output_ch, norm_layer=nn.BatchNorm2d, use_dropout=False, num_lights=8):
        super().__init__()
                
        self.conv_down1 = DwConv(input_nc=input_ch, output_nc=64, innermost=False, outermost=True, norm_layer=nn.BatchNorm2d)
        self.conv_down2 = DwConv(input_nc=64, output_nc=128, norm_layer=nn.BatchNorm2d)
        self.conv_down3 = DwConv(input_nc=128, output_nc=256, norm_layer=nn.BatchNorm2d)
        self.conv_down4 = DwConv(input_nc=256, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down5 = DwConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down6 = DwConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down7 = DwConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)             
        
        self.sconv_up7 = UpConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)
        self.sconv_up6 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up5 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up4 = UpConv(input_nc=512, output_nc=256, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up3 = UpConv(input_nc=256, output_nc=128, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up2 = UpConv(input_nc=128, output_nc=64, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up1 = UpConv(input_nc=32, output_nc=output_ch, outermost=True, norm_layer=nn.BatchNorm2d)

        self.mconv_up7 = UpConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)
        self.mconv_up6 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up5 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up4 = UpConv(input_nc=512, output_nc=256, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up3 = UpConv(input_nc=256, output_nc=128, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up2 = UpConv(input_nc=128, output_nc=64, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up1 = UpConv(input_nc=32, output_nc=output_ch, outermost=True, norm_layer=nn.BatchNorm2d)

        self.htan=nn.Hardtanh(0,1)
        self.htan255=nn.Hardtanh(0,255)

        self.mat_confmap=nn.Sequential(nn.Conv2d(64, 1, kernel_size=3,stride=1, padding=1),
                                       nn.Sigmoid())
        self.mat_upsample2=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.mat_upsample4=nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.mat_final= nn.Conv2d(4, 3, kernel_size=4,stride=2, padding=1)

        
    def encoder(self, x):
        d1=self.conv_down1(x)
        d2=self.conv_down2(d1)
        d3=self.conv_down3(d2)
        d4=self.conv_down4(d3)
        d5=self.conv_down5(d4)
        d6=self.conv_down6(d5)
        d7=self.conv_down7(d6)
        return [d1,d2,d3,d4,d5,d6,d7] 

    def shd_decoder(self,encoded):
        x= encoded
        u7=self.sconv_up7(x[6])
        u7 = torch.cat([u7, x[5]], dim=1)
        u6=self.sconv_up6(u7)
        u6 = torch.cat([u6, x[4]], dim=1)
        u5=self.sconv_up5(u6)
        u5 = torch.cat([u5, x[3]], dim=1)
        u4=self.sconv_up4(u5)
        u4 = torch.cat([u4, x[2]], dim=1)
        u3=self.sconv_up3(u4)
        u3 = torch.cat([u3, x[1]], dim=1)
        u2=self.sconv_up2(u3) 
        # u1 = torch.cat([u2, x[0]], dim=1)
        out=self.sconv_up1(u2)
        return out 


    def mat_decoder(self,encoded):
        x= encoded
        u7=self.mconv_up7(x[6])
        u7 = torch.cat([u7, x[5]], dim=1)
        u6=self.mconv_up6(u7)
        u6 = torch.cat([u6, x[4]], dim=1)
        u5=self.mconv_up5(u6)
        u5 = torch.cat([u5, x[3]], dim=1)
        u4=self.mconv_up4(u5)
        u4 = torch.cat([u4, x[2]], dim=1)
        u3=self.mconv_up3(u4)
        u3 = torch.cat([u3, x[1]], dim=1)
        u2=self.mconv_up2(u3) 
        # u1 = torch.cat([u2, x[0]], dim=1)
        out1=self.mconv_up1(u2)
        conf=self.mat_confmap(u2)
        up4=self.mat_upsample4(conf)
        up2=self.mat_upsample2(out1)
        out=self.mat_final(torch.cat([up2, up4],dim=1))
        return out 
    
    def forward(self,x):
        e=self.encoder(x)
        shd=self.htan255(self.shd_decoder(e))
        mat=self.htan(self.mat_decoder(e))
    
        return {'shd':shd, 'mat':mat}


class UNetMat(nn.Module):

    def __init__(self, input_ch, output_ch, norm_layer=nn.BatchNorm2d, use_dropout=False, num_lights=8):
        super().__init__()
                
        self.conv_down1 = DwConv(input_nc=input_ch, output_nc=64, innermost=False, outermost=True, norm_layer=nn.BatchNorm2d)
        self.conv_down2 = DwConv(input_nc=64, output_nc=128, norm_layer=nn.BatchNorm2d)
        self.conv_down3 = DwConv(input_nc=128, output_nc=256, norm_layer=nn.BatchNorm2d)
        self.conv_down4 = DwConv(input_nc=256, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down5 = DwConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down6 = DwConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d)
        self.conv_down7 = DwConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)             
        
        self.sconv_up7 = UpConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)
        self.sconv_up6 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up5 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up4 = UpConv(input_nc=512, output_nc=256, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up3 = UpConv(input_nc=256, output_nc=128, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up2 = UpConv(input_nc=128, output_nc=64, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.sconv_up1 = UpConv(input_nc=64, output_nc=output_ch, outermost=True, norm_layer=nn.BatchNorm2d)

        self.mconv_up7 = UpConv(input_nc=512, output_nc=512, innermost=True, norm_layer=nn.BatchNorm2d)
        self.mconv_up6 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up5 = UpConv(input_nc=512, output_nc=512, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up4 = UpConv(input_nc=512, output_nc=256, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up3 = UpConv(input_nc=256, output_nc=128, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up2 = UpConv(input_nc=128, output_nc=64, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        self.mconv_up1 = UpConv(input_nc=64, output_nc=output_ch, outermost=True, norm_layer=nn.BatchNorm2d)

        self.htan=nn.Hardtanh(0,1)
        self.htan100=nn.Hardtanh(0,100)
        
    def encoder(self, x):
        d1=self.conv_down1(x)
        d2=self.conv_down2(d1)
        d3=self.conv_down3(d2)
        d4=self.conv_down4(d3)
        d5=self.conv_down5(d4)
        d6=self.conv_down6(d5)
        d7=self.conv_down7(d6)
        return [d1,d2,d3,d4,d5,d6,d7] 

    def shd_decoder(self,encoded):
        x= encoded
        u7=self.sconv_up7(x[6])
        u7 = torch.cat([u7, x[5]], dim=1)
        u6=self.sconv_up6(u7)
        u6 = torch.cat([u6, x[4]], dim=1)
        u5=self.sconv_up5(u6)
        u5 = torch.cat([u5, x[3]], dim=1)
        u4=self.sconv_up4(u5)
        u4 = torch.cat([u4, x[2]], dim=1)
        u3=self.sconv_up3(u4)
        u3 = torch.cat([u3, x[1]], dim=1)
        u2=self.sconv_up2(u3) 
        u1 = torch.cat([u2, x[0]], dim=1)
        out=self.sconv_up1(u1)
        return out 

    def mat_decoder(self,encoded):
        x= encoded
        u7=self.mconv_up7(x[6])
        u7 = torch.cat([u7, x[5]], dim=1)
        u6=self.mconv_up6(u7)
        u6 = torch.cat([u6, x[4]], dim=1)
        u5=self.mconv_up5(u6)
        u5 = torch.cat([u5, x[3]], dim=1)
        u4=self.mconv_up4(u5)
        u4 = torch.cat([u4, x[2]], dim=1)
        u3=self.mconv_up3(u4)
        u3 = torch.cat([u3, x[1]], dim=1)
        u2=self.mconv_up2(u3) 
        u1 = torch.cat([u2, x[0]], dim=1)
        out=self.mconv_up1(u1)
        return out 
    

    def forward(self,x):
        e=self.encoder(x)
        shd=self.htan100(self.shd_decoder(e))
        mat=self.htan(self.mat_decoder(e))
    
        return {'shd':shd, 'mat':mat}


