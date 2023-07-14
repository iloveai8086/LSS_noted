"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D  # 41
        self.C = C  # 64

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)  # （6，512，8，22）8 22 就是（128/16，352/16）得到的，下采样
        # Depth
        x = self.depthnet(x)  # （6，512，8，22）-> (6,105,8,22) 105 = 64 + 41,可以叫预测深度特征
        # 深度
        depth = self.get_depth_dist(
            x[:, :self.D])  # x[:, :self.D].shape = torch.Size([6, 41, 8, 22])，depth还是[6, 41, 8, 22]，softmax加权
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)  # 底下的维度就是torch.Size([6, 64, 8, 22])
        # 得到torch.Size([6, 64, 41, 8, 22]); torch.Size([6, 1, 41, 8, 22]) x torch.Size([6, 64, 1, 8, 22])
        # 元素乘，妙用squeeze改变shape
        # depth是[6, 41, 8, 22]，和视锥点云的shape是一致的，点云就还有一个3维度，代表xyz；
        # new_x：[6, 64, 41, 8, 22]  视锥、ego下的视锥都是41, 8, 22，这个64就是depth + 语义融一起的特征了（因为做了乘法嘛）
        return depth, new_x  # depth，6个相机的41个深度值加权了，概率分布or概率密度？

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])  # 这边应该是特征融合，上采样的，笔记里也有的
        return x  # （6，512，8，22）

    def forward(self, x):  # （6，3，128，352）
        depth, x = self.get_depth_feat(x)

        return x  # 返回的是new_x：[6, 64, 41, 8, 22]  视锥、ego下的视锥都是41, 8, 22，这个64就是depth + 语义融一起的特征了（因为做了乘法嘛）


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf  # {'dbound': [4.0, 45.0, 1.0], 'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5], 'zbound': [-10.0, 10.0, 20.0]}
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        # dx, bx, nx
        # tensor([ 0.5000,  0.5000, 20.0000])、
        # tensor([-49.7500, -49.7500,   0.0000])、
        # tensor([200, 200,   1])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()  # 视锥（41，8，22，3）,Lift splat的核心
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        # backbone，一直到深度的估计，lift操作，原本图像的feature map只有C的，现在想要D这个维度，就会形成C*D的维度，比如C=64，D是41，这样子就得到了2000多维度，很大的维度
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        '''
        在每个图像特征点位置（高H、宽W）扩展D个深度，最终输出是DxHxWx3，其中3表示特征点和深度坐标[h, w, d]，
        这里只是初步对1个相机视角的感知空间进行划分，形成视锥点云。
        后续会扩展到N个相机，并把视锥点云放到ego周围的空间，也会基于空间范围对点云进行筛选。
        :return:
        '''
        # 根据配置参数构建单个相机的视锥（Frustum）
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # final_dim：(128, 352)，这个到底是原始输入给网络的size还是final？
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 8 22    16倍下采样，图像进行16倍的压缩
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH,
                                                                                              fW)  # (41)->(41,1,1)->(41,8,22)
        D, _, _ = ds.shape  # ds内部数据是4~44，41个深度分布值，从4开始是因为近处的由于视野问题看不到;比如ds[0]就是8*22个4，以此类推
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH,
                                                                                      fW)  # (22)->(1,1,22)->(41,8,22)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH,
                                                                                      fW)  # (8)->(1,8,1)->(41,8,22)
        # expand会让你这个tensor重复很多遍,还需要注意的就是xs和ys这边是ogfH和ogfW的尺寸上的；
        # xs[0]:tensor([[  0.0000,  16.7143,  33.4286,  50.1429,  66.8571,  83.5714, 100.2857,
        #          117.0000, 133.7143, 150.4286, 167.1429, 183.8571, 200.5714, 217.2857,
        #          234.0000, 250.7143, 267.4286, 284.1429, 300.8571, 317.5714, 334.2857,
        #          351.0000]  22个数重复了八遍，然后22*8再重复41遍
        # ys[0]:tensor([[  0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
        #            0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
        #            0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
        #            0.0000]  22个数 按照0 18 36 ..... 128 每一个数内部先重复22遍，8个拼起来，最后最外层重复41遍
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # (41,8,22) (41,8,22) (41,8,22) 堆叠，得到的是(41,8,22,3)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        利用内外参，对N个相机Frustum进行坐标变换，输出视锥点云在自车周围物理空间的位置索引

        x:imgs,环视图片  (bs, N, 3, H, W)
        rots:由相机坐标系->车身坐标系的旋转矩阵，rots = (bs, N, 3, 3)
        trans:由相机坐标系->车身坐标系的平移矩阵，trans=(bs, N, 3)
        intrins:相机内参，intrinsic = (bs, N, 3, 3)
        post_rots:由图像增强引起的旋转矩阵，post_rots = (bs, N, 3, 3)
        post_trans:由图像增强引起的平移矩阵，post_trans = (bs, N, 3)

        对N个相机的frustum进行坐标变换，简单来说就是内参外参以及六个视角的变换，
        输出结果是BxNxDxHxWx3，其中3是ego坐标系下的空间位置[x,y,z]，
        B是batch_id，N是相机个数，D是深度分布数。
        这样就等同于把ego周围空间划分为DxHxW块。

        Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation 抵消数据增强及预处理对像素的变化
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1,
                                                3)  # frustum:[41, 8, 22, 3] 3这个shape就是torch.stack((xs, ys, ds), -1)堆叠出来的
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        # 3是什么：xs，ys，lambda
        # lambda * xs， lambda * ys，lambda 然后再最后一个维度stack
        # 图像坐标系 -> 归一化相机坐标系 -> 相机坐标系 -> 车身坐标系
        # 但是自认为由于转换过程是线性的，所以反归一化是在图像坐标系完成的，然后再利用
        # 求完逆的内参投影回相机坐标系；乘法就是成像过程像素坐标系的，除以Z，这边就乘Z回到归一化坐标系
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # （torch.Size([1, 6, 41, 8, 22, 3, 1])  3=[zc*xs，zc*ys，zc]
        # 这边的点是原来那个41*8*22*3的视锥点云转成归一化相机坐标系的点云
        combine = rots.matmul(torch.inverse(intrins))  # R x K  外参旋转矩阵乘内参，为什么要inverse，根公式到底是2D->3D还是反过来有关系
        # inverse是对内参求逆，成像过程反过来，求完逆的矩阵就是相机坐标系的矩阵，我们的成像过程除以的Z，就是D，也就是那哥们的lambda
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # 变成3*3的矩阵乘上3*1的向量，得到真实的点XYZ，torch.Size([1, 6, 41, 8, 22, 3, 1])->（1，6，41，8，22，3）
        # 这边的points就是刚把旋转矩阵做完，下面的就是加上平移矩阵
        points += trans.view(B, N, 1, 1, 1, 3)  # 已经转到了3D，再加上平移量， B N D H W 3(XYZ)
        # 视频里面讲这个是一个固定住的映射表，从feature map 到XYZ的

        # (bs, N, depth, H, W, 3)：其物理含义               points.shape = [1, 6, 41, 8, 22, 3]
        # 每个batch中的每个环视相机图像特征点，其在不同深度下位置对应
        # 在ego坐标系下的坐标
        # 需要明确的是原图的特征点，还是特征图的特征点；我觉得是原图的，其实也差不多的，特征图也就是16倍的下采样而已
        return points

    def get_cam_feats(self, x):
        """
        对环视图像进行特征提取，并构建图像特征点云
        Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)  # 合并batch和N维度
        x = self.camencode(x)  # 返回的是new_x：[6, 64, 41, 8, 22]  视锥、ego下的视锥都是41, 8, 22，这个64就是depth + 语义融一起的特征了（因为做了乘法嘛）
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)  # (1,6,64,41,8,22)拆开B N
        x = x.permute(0, 1, 3, 4, 5, 2)  # (1,6,41,8,22,64) channel特征 -> last,为了底下的voxel_pooling

        return x

    def voxel_pooling(self, geom_feats, x):  # (1,6,41,8,22,3) (1,6,41,8,22,64)
        # geom_feats；(B x N x D x H x W x 3)：在ego坐标系下的坐标点，视锥；映射表
        # x；(B x N x D x fH x fW x C)：图像点云特征，这个图像点云特征我看了代码，是没有什么深度的先验知识的，只是单纯在某一步，有64+41这个shape
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x # 将特征点云展平，一共有 B*N*D*H*W 个点
        x = x.reshape(Nprime, C)  # (43296,64)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()  # ego下的空间坐标转换到体素坐标（计算栅格坐标并取整）
        # bx:tensor([-49.7500, -49.7500,   0.0000], device='cuda:0')
        # dx:tensor([ 0.5000,  0.5000, 20.0000], device='cuda:0')
        # geom_feats是映射表，我们最终还是要生成一张图像（图像类似的tensor）的，不可能说一个矩阵，中点是负数。BEV空间是以自车为原点的；后相机都是负数；
        # 所以geom_feats我更感觉是坐标，平移一下
        # 平移完就是0~200的范围了，0~20
        # （50-（-50））
        #  * 感知范围
        #       * x轴方向的感知范围 -50m ~ 50m；y轴方向的感知范围 -50m ~ 50m；z轴方向的感知范围 -10m ~ 10m；
        #     * BEV单元格大小
        #       * x轴方向的单位长度 0.5m；y轴方向的单位长度 0.5m；z轴方向的单位长度 20m；
        #     * BEV的网格尺寸
        #       * 200 x 200 x 1；
        # 这边为什么要搞成(-49.75 - 0.5/2) = -50?
        # 先平移(geom_feats - (self.bx - self.dx / 2.)
        # 原始的geom_feats：([-50,50],[-50,50],(-10,10)) -> ([0,100],[0,100],(0,20))
        # 这边就是把一个超大立方体平移过来，然后除以每一个体素的大小(0.5,0.5,20)
        # 得到的范围就是（200，200，1）了
        geom_feats = geom_feats.view(Nprime, 3)  # # 将体素坐标同样展平，geom_feats: (B*N*D*H*W, 3) （43296，3）
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])  # 处理batch 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # 其实就是把batch拆开，（43296，4） 4 里面有一个是batch_idx

        # filter out points that are outside box  过滤掉在边界线之外的点 x:0~199  y: 0~199  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])  # 0<z<1  z只剩0了
        x = x[kept]
        geom_feats = geom_feats[kept]  # （42162，4）
        # 相同位置的特征的点，合在一起
        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面，（42162，）
        # geom_feats一共42162个点；
        # B先不管，就是哪个batch
        # self.nx[1] * self.nx[2] -》200*1
        # 第一行0-199x200 第二行199x1，后面Z被拍扁了都是0
        # geom_feats融合了六个相机的点坐标，没有相机N这个维度
        # z为0后，即使xy互换，互反，导致rank相等，是不会发生的，除非xyz完全相同，rank才会相同
        # X和Y乘的后面的self.nx[1] * self.nx[2] * B不同，其实我感觉就是访问C++数组一样，乘一行的个数加上列的偏移那种，就像算每一个坐标的绝对位置X*col + Y + Z
        # 不同相机的交界处，rank可能就会相等，4-5个点同rank，越近rank相同越多，远的少
        sorts = ranks.argsort()  # 按照rank排序，这样rank相近的点就在一起了
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 都重新排序

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)  # (42162,64) (42162,4) (42162,)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  #（1，64，1，200，200）这边N相机个数也没了，拍扁了？下架把说
        # 因为进入3D空间，就没有N这个概念了
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中
        # x和geom_feats一一对应的
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维，（1，64，200，200），Z只有一格，是把（1，64，1，200，200）第三维1干掉
        # 其实这个拍扁是不合理的，很多地方没特征，交叉处又是密集的
        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots,
                                 post_trans)  # 利用内外参，对N个相机Frustum进行坐标变换，输出视锥点云在自车周围物理空间的位置索引
        x = self.get_cam_feats(x)
        # gemo:(1,6,41,8,22,3) ego下的视锥点，x:(1,6,41,8,22,64) 图像特征，但是已经是按照点云的坐标系排布了
        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots,
                post_trans):  # （1，6，3，128，352）（1，6，3，3）（1，6，3）（1，6，3，3）（1，6，3）
        """
        x:imgs,环视图片  (bs, N, 3, H, W)
        rots:由相机坐标系->车身坐标系的旋转矩阵，rots = (bs, N, 3, 3)
        trans:由相机坐标系->车身坐标系的平移矩阵，trans=(bs, N, 3)
        intrins:相机内参，intrinsic = (bs, N, 3, 3)
        post_rots:由图像增强引起的旋转矩阵，post_rots = (bs, N, 3, 3)
        post_trans:由图像增强引起的平移矩阵，post_trans = (bs, N, 3)
        """
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)  # 得到的BEV特征，普通的CNN能搞定这种稀疏的（重叠区，特征多，边缘稀疏）重叠区域会有相当大的精度~
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
