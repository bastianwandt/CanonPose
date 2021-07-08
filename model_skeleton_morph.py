import torch.nn as nn


class Synthesizer(nn.Module):
    def __init__(self):
        super(Synthesizer, self).__init__()

        self.upscale = nn.Linear(32, 1024)
        self.res_common = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        self.res_cam1 = res_block()
        self.res_cam2 = res_block()
        self.pose_morph = nn.Linear(1024, 32)

    def forward(self, x):

        xu = self.upscale(x)

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(xu))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        x_pose = x + self.pose_morph(xp)

        return x_pose


class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.l1 = nn.Linear(1024, 1024)
        self.l2 = nn.Linear(1024, 1024)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.LeakyReLU()(self.l2(x))
        x += inp

        return x

