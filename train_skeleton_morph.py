## This is just an example file
# To train it on your own detections your need to have the ground truth 2d data.
# Unfortunately, we are not allowed to share this data due to licensing reasons.
##

import torch.optim
from torch.utils import data
from utils.data import H36MDataset
import torch.optim as optim
import model_skeleton_morph
from utils.print_losses import print_losses
from types import SimpleNamespace
import torch.nn as nn


config = SimpleNamespace()
config.learning_rate = 0.0001
config.BATCH_SIZE = 32
config.N_epochs = 100

data_folder = './data/'

config.datafile = data_folder + 'h36m_train_mpi_skeleton_pred.pickle'

BATCH_SIZE = 32

my_dataset = H36MDataset(config.datafile, subjects=[1])
train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

model = model_skeleton_morph.Synthesizer().cuda()

mse_loss = nn.MSELoss()

N_epochs = 100

params = list(model.parameters())  # + list(dec.parameters())

optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 95], gamma=0.1)

torch.autograd.set_detect_anomaly(True)

losses = SimpleNamespace()
losses_mean = SimpleNamespace()

cams = ['54138969', '55011271', '58860488', '60457274']
all_cams = ['cam0', 'cam1', 'cam2', 'cam3']

for epoch in range(N_epochs):

    for i, sample in enumerate(train_loader):

        poses_2d = {key: sample[key] for key in all_cams}
        poses_2dgt = sample['p2d_gt']

        inp_poses = torch.zeros((poses_2d['cam0'].shape[0] * 4, 32)).cuda()
        output_poses = torch.zeros((poses_2dgt['cam0'].shape[0] * 4, 32)).cuda()
        inp_confidences = torch.zeros((poses_2d['cam0'].shape[0] * 4, 16)).cuda()

        cnt = 0
        for b in range(poses_2d['cam0'].shape[0]):
            for c_idx, cam in enumerate(poses_2d):
                inp_poses[cnt] = poses_2d[cam][b]
                output_poses[cnt] = poses_2dgt[cam][b]
                inp_confidences[cnt] = sample['confidences'][cams[c_idx]][b]
                cnt += 1

        pred_poses = model(inp_poses)

        losses.loss = mse_loss(pred_poses, output_poses)

        optimizer.zero_grad()
        losses.loss.backward()

        optimizer.step()

        for key, value in losses.__dict__.items():
            if key not in losses_mean.__dict__.keys():
                losses_mean.__dict__[key] = []

            losses_mean.__dict__[key].append(value.item())

        if not i % 100:
            print_losses(epoch, i, len(my_dataset) / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not (i % 1000))

            losses_mean = SimpleNamespace()

    torch.save(model, 'models/model_skeleton_morph_S1_gh.pt')

print('done')
