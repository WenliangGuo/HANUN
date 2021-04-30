import torch
from torch import nn
from torch.nn import functional as F
from model import u2net



class Bottom_stage(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(Bottom_stage,self).__init__()

        self.rebnconvin = u2net.REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = u2net.REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = u2net.REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = u2net.REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv2d = u2net.REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = u2net.REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


class PSPBlock(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
   ##     print(str(b), str(c))
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class UPSPNET_8(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(UPSPNET_8,self).__init__()

        self.stage1 = PSPBlock(features=in_ch, out_features=64, sizes=(1, 2, 3, 6))
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = PSPBlock(features=64, out_features=128, sizes=(1, 2, 3, 6))
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = PSPBlock(features=128, out_features=256, sizes=(1, 2, 3, 6))
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = PSPBlock(features=256, out_features=512, sizes=(1, 2, 3, 6))
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = PSPBlock(features=512, out_features=1024, sizes=(1, 2, 3, 6))
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = PSPBlock(features=1024, out_features=2048, sizes=(1, 2, 3))
        self.pool67 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage7 = PSPBlock(features=2048, out_features=2048, sizes=(1, 2, 3))
        self.pool78 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage8 = Bottom_stage(2048, 1024, 2048)

        # decoder
        self.se1 = SE_Block(ch_in=4096);
        self.stage7d = PSPBlock(features=4096, out_features=2048)
        self.se2 = SE_Block(ch_in=4096);
        self.stage6d = PSPBlock(features=4096, out_features=1024)
        self.se3 = SE_Block(ch_in=2048);
        self.stage5d = PSPBlock(features=2048, out_features=512)
        self.se4 = SE_Block(ch_in=1024);
        self.stage4d = PSPBlock(features=1024, out_features=256)
        self.se5 = SE_Block(ch_in=512);
        self.stage3d = PSPBlock(features=512, out_features=128)
        self.se6 = SE_Block(ch_in=256);
        self.stage2d = PSPBlock(features=256, out_features=64)
        self.se7 = SE_Block(ch_in=128);
        self.stage1d = PSPBlock(features=128, out_features=64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(1024,out_ch,3,padding=1)
        self.side7 = nn.Conv2d(2048, out_ch, 3, padding=1)
        self.side8 = nn.Conv2d(2048, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(8,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx = self.pool67(hx6)

        hx7 = self.stage7(hx)
        hx = self.pool78(hx7)

        hx8 = self.stage8(hx)
        hx8up = _upsample_like(hx8, hx7)

        #-------------------- decoder --------------------

        se1 = self.se1(torch.cat((hx8up,hx7),1))
        hx7d = self.stage7d(se1)
        hx7dup = _upsample_like(hx7d,hx6)

        se2 = self.se2(torch.cat((hx7dup,hx6),1))
        hx6d = self.stage6d(se2)
        hx6dup = _upsample_like(hx6d,hx5)

        se3 = self.se3(torch.cat((hx6dup,hx5),1))
        hx5d = self.stage5d(se3)
        hx5dup = _upsample_like(hx5d,hx4)

        se4 = self.se4(torch.cat((hx5dup,hx4),1))
        hx4d = self.stage4d(se4)
        hx4dup = _upsample_like(hx4d,hx3)

        se5 = self.se5(torch.cat((hx4dup,hx3),1))
        hx3d = self.stage3d(se5)
        hx3dup = _upsample_like(hx3d, hx2)

        se6 = self.se6(torch.cat((hx3dup, hx2), 1))
        hx2d = self.stage2d(se6)
        hx2dup = _upsample_like(hx2d, hx1)

        se7 = self.se7(torch.cat((hx2dup, hx1), 1))
        hx1d = self.stage1d(se7)

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6d)
        d6 = _upsample_like(d6,d1)

        d7 = self.side7(hx7d)
        d7 = _upsample_like(d7, d1)

        d8 = self.side7(hx8)
        d8 = _upsample_like(d8, d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6,d7, d8),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6), F.sigmoid(d7), F.sigmoid(d8)


class UPSPNET_7(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(UPSPNET_7,self).__init__()

        self.stage1 = PSPBlock(features=in_ch, out_features=64, sizes=(1, 2, 3, 6))
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = PSPBlock(features=64, out_features=128, sizes=(1, 2, 3, 6))
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = PSPBlock(features=128, out_features=256, sizes=(1, 2, 3, 6))
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = PSPBlock(features=256, out_features=512, sizes=(1, 2, 3))
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = PSPBlock(features=512, out_features=1024, sizes=(1, 2, 3))
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = PSPBlock(features=1024, out_features=1024, sizes=(1, 2, 3))
        self.pool67 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage7 = Bottom_stage(1024, 512, 1024)

        # decoder
        self.se1 = SE_Block(ch_in=2048);
        self.stage6d = PSPBlock(features=2048, out_features=1024)
        self.se2 = SE_Block(ch_in=2048);
        self.stage5d = PSPBlock(features=2048, out_features=512)
        self.se3 = SE_Block(ch_in=1024);
        self.stage4d = PSPBlock(features=1024, out_features=256)
        self.se4 = SE_Block(ch_in=512);
        self.stage3d = PSPBlock(features=512, out_features=128)
        self.se5 = SE_Block(ch_in=256);
        self.stage2d = PSPBlock(features=256, out_features=64)
        self.se6 = SE_Block(ch_in=128);
        self.stage1d = PSPBlock(features=128, out_features=64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(1024,out_ch,3,padding=1)
        self.side7 = nn.Conv2d(1024, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(7,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx = self.pool67(hx6)

        hx7 = self.stage7(hx)
        hx7up = _upsample_like(hx7, hx6)

        #-------------------- decoder --------------------

        se1 = self.se1(torch.cat((hx7up,hx6),1))
        hx6d = self.stage6d(se1)
        hx6dup = _upsample_like(hx6d,hx5)

        se2 = self.se2(torch.cat((hx6dup,hx5),1))
        hx5d = self.stage5d(se2)
        hx5dup = _upsample_like(hx5d,hx4)

        se3 = self.se3(torch.cat((hx5dup,hx4),1))
        hx4d = self.stage4d(se3)
        hx4dup = _upsample_like(hx4d,hx3)

        se4 = self.se4(torch.cat((hx4dup,hx3),1))
        hx3d = self.stage3d(se4)
        hx3dup = _upsample_like(hx3d,hx2)

        se5 = self.se5(torch.cat((hx3dup,hx2),1))
        hx2d = self.stage2d(se5)
        hx2dup = _upsample_like(hx2d, hx1)

        se6 = self.se6(torch.cat((hx2dup, hx1), 1))
        hx1d = self.stage1d(se6)

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d7 = self.side7(hx7)
        d7 = _upsample_like(d7, d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6,d7),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6), F.sigmoid(d7)



class UPSPNET(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(UPSPNET,self).__init__()

        self.stage1 = PSPBlock(features=in_ch, out_features=64, sizes=(1, 2, 3, 6))
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = PSPBlock(features=64, out_features=128, sizes=(1, 2, 3, 6))
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = PSPBlock(features=128, out_features=256, sizes=(1, 2, 3, 6))
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = PSPBlock(features=256, out_features=512, sizes=(1, 2, 3))
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = PSPBlock(features=512, out_features=512, sizes=(1, 2, 3))
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = Bottom_stage(512, 256, 512)

        # decoder
        self.se1 = SE_Block(ch_in=1024);
        self.stage5d = PSPBlock(features=1024, out_features=512)
        self.se2 = SE_Block(ch_in=1024);
        self.stage4d = PSPBlock(features=1024, out_features=256)
        self.se3 = SE_Block(ch_in=512);
        self.stage3d = PSPBlock(features=512, out_features=128)
        self.se4 = SE_Block(ch_in=256);
        self.stage2d = PSPBlock(features=256, out_features=64)
        self.se5 = SE_Block(ch_in=128);
        self.stage1d = PSPBlock(features=128, out_features=64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------

        se1 = self.se1(torch.cat((hx6up,hx5),1))
        hx5d = self.stage5d(se1)
        hx5dup = _upsample_like(hx5d,hx4)

        se2 = self.se2(torch.cat((hx5dup,hx4),1))
        hx4d = self.stage4d(se2)
        hx4dup = _upsample_like(hx4d,hx3)

        se3 = self.se3(torch.cat((hx4dup,hx3),1))
        hx3d = self.stage3d(se3)
        hx3dup = _upsample_like(hx3d,hx2)

        se4 = self.se4(torch.cat((hx3dup,hx2),1))
        hx2d = self.stage2d(se4)
        hx2dup = _upsample_like(hx2d,hx1)

        se5 = self.se5(torch.cat((hx2dup,hx1),1))
        hx1d = self.stage1d(se5)


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

class UPSPNET_5(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(UPSPNET_5,self).__init__()

        self.stage1 = PSPBlock(features=in_ch, out_features=64, sizes=(1, 2, 3, 6))
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = PSPBlock(features=64, out_features=128, sizes=(1, 2, 3, 6))
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = PSPBlock(features=128, out_features=256, sizes=(1, 2, 3, 6))
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = PSPBlock(features=256, out_features=512, sizes=(1, 2, 3))
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = Bottom_stage(512, 256, 512)

        # decoder
        self.se1 = SE_Block(ch_in=1024);
        self.stage4d = PSPBlock(features=1024, out_features=256)
        self.se2 = SE_Block(ch_in=512);
        self.stage3d = PSPBlock(features=512, out_features=128)
        self.se3 = SE_Block(ch_in=256);
        self.stage2d = PSPBlock(features=256, out_features=64)
        self.se4 = SE_Block(ch_in=128);
        self.stage1d = PSPBlock(features=128, out_features=64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(5,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx5up = _upsample_like(hx5,hx4)

        #-------------------- decoder --------------------
        print(hx5up.size())
        print(hx4.size())

        se1 = self.se1(torch.cat((hx5up,hx4),1))
        hx4d = self.stage4d(se1)
        hx4dup = _upsample_like(hx4d,hx3)

        se2 = self.se2(torch.cat((hx4dup,hx3),1))
        hx3d = self.stage3d(se2)
        hx3dup = _upsample_like(hx3d,hx2)

        se3 = self.se3(torch.cat((hx3dup,hx2),1))
        hx2d = self.stage2d(se3)
        hx2dup = _upsample_like(hx2d,hx1)

        se4 = self.se4(torch.cat((hx2dup,hx1),1))
        hx1d = self.stage1d(se4)

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5)
        d5 = _upsample_like(d5,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)

class UPSPNET_4(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(UPSPNET_4,self).__init__()

        self.stage1 = PSPBlock(features=in_ch, out_features=64, sizes=(1, 2, 3, 6))
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = PSPBlock(features=64, out_features=128, sizes=(1, 2, 3, 6))
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = PSPBlock(features=128, out_features=256, sizes=(1, 2, 3, 6))
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = Bottom_stage(256, 128, 256)

        # decoder
        self.se1 = SE_Block(ch_in=512);
        self.stage3d = PSPBlock(features=512, out_features=128)
        self.se2 = SE_Block(ch_in=256);
        self.stage2d = PSPBlock(features=256, out_features=64)
        self.se3 = SE_Block(ch_in=128);
        self.stage1d = PSPBlock(features=128, out_features=64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(4,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx4up = _upsample_like(hx4,hx3)

        #-------------------- decoder --------------------
        print(hx4up.size())
        print(hx3.size())

        se1 = self.se1(torch.cat((hx4up,hx3),1))
        hx3d = self.stage3d(se1)
        hx3dup = _upsample_like(hx3d,hx2)

        se2 = self.se2(torch.cat((hx3dup,hx2),1))
        hx2d = self.stage2d(se2)
        hx2dup = _upsample_like(hx2d,hx1)

        se3 = self.se3(torch.cat((hx2dup,hx1),1))
        hx1d = self.stage1d(se3)

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4)
        d4 = _upsample_like(d4,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4)

class UPSPNET_RSU(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(UPSPNET_RSU,self).__init__()

        self.stage1 = PSPBlock(features=in_ch, out_features=64, sizes=(1, 2, 3, 6))
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = PSPBlock(features=64, out_features=128, sizes=(1, 2, 3, 6))
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
 
        self.stage3 = PSPBlock(features=128, out_features=256, sizes=(1, 2, 3, 6))
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = PSPBlock(features=256, out_features=512, sizes=(1, 2, 3))
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = PSPBlock(features=512, out_features=512, sizes=(1, 2, 3))
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = Bottom_stage(512, 256, 512)

        # decoder
        self.se1 = SE_Block(ch_in=1024)
        self.stage5d = u2net.RSU4F(1024,256,512)
        self.se2 = SE_Block(ch_in=1024)
        self.stage4d = u2net.RSU4(1024,128,256)
        self.se3 = SE_Block(ch_in=512)
        self.stage3d = u2net.RSU5(512,64,128)
        self.se4 = SE_Block(ch_in=256)
        self.stage2d = u2net.RSU6(256,32,64)
        self.se5 = SE_Block(ch_in=128)
        self.stage1d = u2net.RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------

        se1 = self.se1(torch.cat((hx6up,hx5),1))
        hx5d = self.stage5d(se1)
        hx5dup = _upsample_like(hx5d,hx4)

        se2 = self.se2(torch.cat((hx5dup,hx4),1))
        hx4d = self.stage4d(se2)
        hx4dup = _upsample_like(hx4d,hx3)

        se3 = self.se3(torch.cat((hx4dup,hx3),1))
        hx3d = self.stage3d(se3)
        hx3dup = _upsample_like(hx3d,hx2)

        se4 = self.se4(torch.cat((hx3dup,hx2),1))
        hx2d = self.stage2d(se4)
        hx2dup = _upsample_like(hx2d,hx1)

        se5 = self.se5(torch.cat((hx2dup,hx1),1))
        hx1d = self.stage1d(se5)


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
    
class UPSPNET_RSU_noSE(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(UPSPNET_RSU_noSE,self).__init__()

        self.stage1 = PSPBlock(features=in_ch, out_features=64, sizes=(1, 2, 3, 6))
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = PSPBlock(features=64, out_features=128, sizes=(1, 2, 3, 6))
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
 
        self.stage3 = PSPBlock(features=128, out_features=256, sizes=(1, 2, 3, 6))
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = PSPBlock(features=256, out_features=512, sizes=(1, 2, 3))
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = PSPBlock(features=512, out_features=512, sizes=(1, 2, 3))
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = Bottom_stage(512, 256, 512)

        # decoder
        self.stage5d = u2net.RSU4F(1024,256,512)       
        self.stage4d = u2net.RSU4(1024,128,256)
        self.stage3d = u2net.RSU5(512,64,128)
        self.stage2d = u2net.RSU6(256,32,64)
        self.stage1d = u2net.RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------

        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

