import os
import time

import setproctitle
import torch
import torch.utils.data as data
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

import utils.utils as utils
from args_VFN import args as args
from models import VFN as model
from models.dataset import TestDataset

setproctitle.setproctitle("CLDyN")
device_id = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = device_id
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:" + device_id if USE_CUDA else "cpu")

model_name = 'VFN'

tf_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

with torch.no_grad():
    net = getattr(model, model_name)
    net = net(num_blocks=2)

net.eval()
net.to(device)

checkpoint = torch.load(args.test_checkpoint_path)
net.load_state_dict(checkpoint['net'])

datasets_name = ['M3FD_Detection', 'FMB', 'VT5000']

def test(net, test_data_iter, save_img_dir):
    for step, (vis, ir, file_dir) in tqdm(enumerate(test_data_iter)):
        vis = vis.to(device).detach()
        ir = ir.to(device).detach()

        vis_ycbcr = utils.rgb2ycbcr(vis)
        vis_Y, vis_Cb, vis_Cr = torch.split(vis_ycbcr, 1, 1)

        ir = ir[:, :1, :, :]

        with torch.no_grad():
            fusion_img = net.forward_ER(ir, vis_Y)

            plex = '.png'
            file_name = file_dir[0].split('/')[-1].split('.')[0]

            output_name = save_img_dir + "/" + file_name + plex
            out = utils.ycbcr2rgb(torch.cat([fusion_img, vis_Cb, vis_Cr], dim=1))
            save_image(out, output_name)


if __name__ == "__main__":
    is_d2 = False

    for dataset in datasets_name:
        now = int(time.time())
        timeArr = time.localtime(now)
        nowTime = time.strftime("%Y%m%d_%H-%M-%S", timeArr)
        save_img_dir = './results/' + nowTime + '_' + model_name + '_' + dataset
        utils.check_dir(save_img_dir)

        dataset_dir = os.path.join(args.test_dataset_path, dataset)

        if dataset == 'M3FD_Detection':
            vis_test_dir = os.path.join(dataset_dir, 'vi', 'test')
            ir_test_dir = os.path.join(dataset_dir, 'ir', 'test')
        elif dataset == 'FMB':
            vis_test_dir = os.path.join(dataset_dir, 'test', 'Visible')
            ir_test_dir = os.path.join(dataset_dir, 'test', 'Infrared')
        elif dataset == 'VT5000':
            vis_test_dir = os.path.join(dataset_dir, 'Test', 'RGB')
            ir_test_dir = os.path.join(dataset_dir, 'Test', 'T_GRAY')

        test_dataset = TestDataset(vis_test_dir, ir_test_dir, tf_test, is_d2, img_type='RGB')

        test_data_iter = data.DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=1
        )

        test(net, test_data_iter, save_img_dir)