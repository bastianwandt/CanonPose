import torch
import torch.nn as nn


class Lifter(nn.Module):
    def __init__(self):
        super(Lifter, self).__init__()

        self.upscale = nn.Linear(32+16, 1024)
        self.res_common = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        self.res_cam1 = res_block()
        self.res_cam2 = res_block()
        self.pose3d = nn.Linear(1024, 48)
        self.enc_rot = nn.Linear(1024, 3)

    def forward(self, p2d, conf):

        x = torch.cat((p2d, conf), axis=1)

        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        x_pose = self.pose3d(xp)

        # camera path
        xc = nn.LeakyReLU()(self.res_cam1(x))
        xc = nn.LeakyReLU()(self.res_cam2(xc))
        xc = self.enc_rot(xc)

        return x_pose, xc


class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.l1 = nn.Linear(1024, 1024)
        self.l2 = nn.Linear(1024, 1024)
        #self.bn1 = nn.BatchNorm1d(1024)
        #self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.LeakyReLU()(self.l2(x))
        x += inp

        return x

