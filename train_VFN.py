import os
import time

import numpy as np
import setproctitle
import torch

import torch.utils.data as data
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

import loss.loss as Loss
import utils.utils as utils
from args_VFN import args as args
from models import VFN as model
from models.dataset import TrainDataset
from utils.utils import check_dir

setproctitle.setproctitle("CLDyN")
device_id = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = device_id
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:" + device_id if USE_CUDA else "cpu")

model_name = "VFN"
now = int(time.time())
timeArr = time.localtime(now)
nowTime = time.strftime("%Y%m%d_%H-%M-%S", timeArr)
save_model_dir = args.train_save_model_dir + "/" + nowTime + "_" + model_name + "_model"
save_img_dir = args.train_save_img_dir + "/" + nowTime + "_" + model_name + "_img"
save_loss_dir = save_img_dir + "/loss_map"
save_model_info_dir = save_img_dir + "/" + "model_info"
save_feature_dir = save_img_dir + "/" + "Feature"

save_test_dir = save_img_dir + "/" + "test"
check_dir(save_test_dir)
check_dir(save_loss_dir)
check_dir(save_model_info_dir)
check_dir(save_model_dir)
check_dir(save_img_dir)
check_dir(save_feature_dir)

tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize([512, 512]),
    torchvision.transforms.ToTensor()
])

tf_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize([512, 512]),
    torchvision.transforms.ToTensor()
])

dataset_dir = '...'

vis_FMB_train_dir = os.path.join(dataset_dir, 'RoadScene', 'train', 'vis')
ir_FMB_train_dir = os.path.join(dataset_dir, 'RoadScene', 'train', 'ir')
vis_VT5000_train_dir = os.path.join(dataset_dir, 'MSRS', 'train', 'vis')
ir_VT5000_train_dir = os.path.join(dataset_dir, 'MSRS', 'train', 'ir')
vis_M3FD_train_dir = os.path.join(dataset_dir, 'M3FD', 'train', 'vis')
ir_M3FD_train_dir = os.path.join(dataset_dir, 'M3FD', 'train', 'ir')
vis_LLVIP_train_dir = os.path.join(dataset_dir, 'LLVIP', 'train', 'vis')
ir_LLVIP_train_dir = os.path.join(dataset_dir, 'LLVIP', 'train', 'ir')

vis_dir_list_train = [vis_FMB_train_dir, vis_M3FD_train_dir, vis_VT5000_train_dir, vis_LLVIP_train_dir]
ir_dir_list_train = [ir_FMB_train_dir, ir_M3FD_train_dir, ir_VT5000_train_dir, vis_LLVIP_train_dir]

dataset = TrainDataset(vis_dir_list_train, ir_dir_list_train, tf, crop_size=[256, 256], img_type='RGB')

data_iter = data.DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=args.batch_size,
    num_workers=args.batch_size
)

iter_num = int(dataset.__len__() / args.batch_size)
save_image_iter = int(iter_num / args.save_image_num)

with torch.no_grad():
    net = getattr(model, model_name)
    net = net(num_blocks=4)
    Lgrad = Loss.L_Grad().to(device)

net.train()
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=args.LR)

with open(save_model_info_dir + "/" + "model_info.txt", 'w') as f:
    f.writelines('--------------- start ---------------' + '\n')
    f.writelines('model_name' + '\n')
    f.writelines(model_name + '\n')
    f.writelines('train_dataset' + '\n')
    for train_dir in vis_dir_list_train:
        f.writelines(train_dir + '\n')
    f.writelines('-------------------------------------' + '\n')
    for i in net.children():
        f.writelines(str(i) + '\n')
    f.writelines('---------------- end ----------------' + '\n')
    f.close()

start_epoch = -1

args.resume = False
if args.resume:
    checkpoint = torch.load(args.path_checkpoint)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']


def train(epoch, net, data_iter):
    net.train()

    epoch_loss_l1 = []
    epoch_loss_grad = []

    for step, (vis, ir, vis_d2, ir_d2, _) in enumerate(data_iter):
        vis = vis.to(device).detach()
        ir = ir.to(device).detach()

        vis_ycbcr = utils.rgb2ycbcr(vis)
        vis_Y, vis_Cb, vis_Cr = torch.split(vis_ycbcr, 1, 1)

        ir = ir[:, :1, :, :]

        fusion_img = net.forward_ER(ir, vis_Y)

        loss_l1 = Loss.Loss_intensity(vis_Y, ir, fusion_img, model='max')
        loss_grad = Lgrad(vis_Y, ir, fusion_img)
        loss = 0.2 * loss_l1 + 1 * loss_grad

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss_l1.append(loss_l1.item())
        epoch_loss_grad.append(loss_grad.item())

        if step % save_image_iter == 0:
            epoch_step_name = str(epoch) + "epoch" + str(step) + "step"
            if epoch % 10 == 0:
                output_name = save_img_dir + "/" + epoch_step_name + ".jpg"
                out = torch.cat([vis_Y, ir, fusion_img], dim=2)
                save_image(out, output_name)


        if ((epoch + 1) == args.Epoch and (step + 1) % iter_num == 0) or (
                epoch % args.save_model_num == 0 and (step + 1) % iter_num == 0):
            ckpt = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(ckpt, os.path.join(save_model_dir, 'ckpt_%s.pth' % (str(epoch))))

    epoch_loss_l1_mean = np.mean(epoch_loss_l1)
    epoch_loss_grad_mean = np.mean(epoch_loss_grad)

    print()
    print(" -epoch " + str(epoch))
    print(" -loss_l1_mean " + str(epoch_loss_l1_mean))
    print(" -loss_grad_mean " + str(epoch_loss_grad_mean))

    with open(save_loss_dir + "/" + "loss.txt", 'a') as f:
        f.writelines(" -epoch " + str(epoch) + '\n')
        f.writelines(
            " -loss_l1_mean " + str(epoch_loss_l1_mean) + '\n')
        f.writelines(
            " -loss_grad_mean " + str(epoch_loss_grad_mean) + '\n')
        f.close()


if __name__ == "__main__":
    for epoch in tqdm(range(start_epoch + 1, args.Epoch)):
        train(epoch, net, data_iter)