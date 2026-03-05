import multiprocessing
import os

import setproctitle
import torch
import torch.utils.data as data
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

from args_RSC import args as args
from ctdnet.src.load_model import load_ctdnet_model
from ctdnet.src.loss import CTDNet_Loss
from models import RSC as fh_model
from models import base_fusion_network as bfn_model
from models.dataset import TestDataset
from utils import utils
from utils.utils import check_dir
from segformer.tools.load_dataiter import load_dataiter
from segformer.tools.load_model import load_segformer_model
from yolo.load_model import load_yolov5_model
from yolo.utils.loss import ComputeLoss

device_id = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = device_id
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:" + device_id if USE_CUDA else "cpu")

torch.cuda.set_device(int(device_id))

setproctitle.setproctitle("CLDyN")


def test(networks, save_inf, t_info, dataiters):
    tf_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    utils.clean_grad(networks["od_net"])
    utils.clean_grad(networks["seg_net"])
    utils.clean_grad(networks["sod_net"])

    task_name = None
    if t_info['is_TD']:
        task_name = 'od'
    elif t_info['is_SS']:
        task_name = 'seg'
    elif t_info['is_SOD']:
        task_name = 'sod'

    for step, x in tqdm(enumerate(dataiters[task_name])):
        with torch.no_grad():
            if t_info['is_TD'] or t_info['is_SOD']:
                vi = x[0].to(device).bfloat16().detach()
                ir = x[1].to(device).bfloat16().detach()
                f_dir = x[2]
            elif t_info['is_SS']:
                f_dir = x['img_metas'].data[0]
                ir = torch.stack([utils.read_image(dir['filename'], tf_test) for dir in f_dir], dim=0).to(
                    device).bfloat16().detach()
                vi = torch.stack(
                    [utils.read_image(dir['filename'].replace('Infrared', 'Visible'), tf_test) for dir in f_dir],
                    dim=0).to(device).bfloat16().detach()

            vi_ycbcr = utils.rgb2ycbcr(vi)
            vi_Y, vi_Cb, vi_Cr = torch.split(vi_ycbcr, 1, 1)
            ir = ir[:, :1, :, :]

            f_img_wofinetune = networks['bfn'].forward_ER(ir, vi_Y)
            f_img_wofinetune_c = utils.ycbcr2rgb(torch.cat([f_img_wofinetune, vi_Cb, vi_Cr], dim=1))

            basis_vector_bank = networks['bvb']

            if t_info['is_TD']:
                target_layer = 'model.model.23.cv3'
                input_tensor = {
                    'f_img': f_img_wofinetune_c
                }
            elif t_info['is_SS']:
                target_layer = 'module.backbone.norm4'
                input_tensor = {
                    'f_img': f_img_wofinetune_c,
                    'x': x
                }
            elif t_info['is_SOD']:
                target_layer = 'fuse23.bn_2'
                input_tensor = {
                    'f_img': f_img_wofinetune_c
                }
            task_f = utils.get_activation_test(networks[task_name + "_net"], target_layer, t_info,
                                               input_tensor).detach().bfloat16()


            for i in range(args.num_blocks):
                if i == 0:
                    ir_f_h = ir
                    vi_f_h = vi_Y

                ir_f = networks['bfn'].ir_e.forward_obo(ir_f_h, i)
                vi_f = networks['bfn'].vi_e.forward_obo(vi_f_h, i)

                if i != args.num_blocks - 1:
                    ir_f_h = networks['rsc_ir'].forward_obo(basis_vector_bank, task_f, ir_f, i)
                    vi_f_h = networks['rsc_vi'].forward_obo(basis_vector_bank, task_f, vi_f, i)

            f_f = networks['bfn'].fusion(ir_f, vi_f)

            for i in range(args.num_blocks):
                f_f = networks['bfn'].fb.forward_obo(f_f, i)

            f_img = f_f
            assert f_img.size(1) == 1

            f_img_c = utils.ycbcr2rgb(torch.cat([f_img, vi_Cb, vi_Cr], dim=1))

            plex = '.png'
            if t_info['is_TD'] or t_info['is_SOD']:
                file_name = f_dir[0].split('/')[-1].split('.')[0]
            elif t_info['is_SS']:
                file_name = f_dir[0]['filename'].split('/')[-1].split('.')[0]

            output_name = save_inf['save_finetune_fusion_dir'] + "/" + file_name + plex
            out = f_img_c
            save_image(out, output_name)

            output_name = save_inf['save_wofinetune_fusion_dir'] + "/" + file_name + plex
            out = f_img_wofinetune_c
            save_image(out, output_name)


def main():
    for ckpts_idx in ['***.pth']:
        for t_idx in [0, 1, 2]:
            TASKS = [
                {'is_TD': True, 'is_SS': False, 'is_SOD': False},
                {'is_TD': False, 'is_SS': True, 'is_SOD': False},
                {'is_TD': False, 'is_SS': False, 'is_SOD': True},
            ]
            t_info = TASKS[t_idx]
            if t_info['is_TD']:
                task_name = 'od'
            elif t_info['is_SS']:
                task_name = 'seg'
            elif t_info['is_SOD']:
                task_name = 'sod'

            bfn_model_name = "VFN"
            t_fh_model_name = "RSC"

            nowTime = utils.get_time()
            save_img_dir = './results/' + nowTime + '_' + bfn_model_name + '_' + t_fh_model_name + '_' + task_name
            save_finetune_fusion_dir = save_img_dir + "/finetune_fusion"
            save_wofinetune_fusion_dir = save_img_dir + "/wofinetune_fusion"
            check_dir(save_img_dir)
            check_dir(save_finetune_fusion_dir)
            check_dir(save_wofinetune_fusion_dir)

            od_dataset_dir = r'...'
            sod_dataset_dir = r'...'

            vis_od_test_dir = os.path.join(od_dataset_dir, 'vi', 'test')
            ir_od_test_dir = os.path.join(od_dataset_dir, 'ir', 'test')
            vis_sod_test_dir = os.path.join(sod_dataset_dir, 'Test', 'RGB')
            ir_sod_test_dir = os.path.join(sod_dataset_dir, 'Test', 'T_GRAY')

            tf_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])

            od_test_dataset = TestDataset(vis_od_test_dir, ir_od_test_dir, tf_test,
                                          img_type='RGB', is_d2=False)
            od_test_data_iter = data.DataLoader(
                dataset=od_test_dataset,
                shuffle=False,
                batch_size=1,
                num_workers=1,
                generator=torch.Generator(device=device)
            )
            sod_test_dataset = TestDataset(vis_sod_test_dir, ir_sod_test_dir, tf_test,
                                           img_type='RGB', is_d2=False)
            sod_test_data_iter = data.DataLoader(
                dataset=sod_test_dataset,
                shuffle=False,
                batch_size=1,
                num_workers=1,
                generator=torch.Generator(device=device)
            )

            ss_dataset, ss_data_iter = load_dataiter(args.segformer_test_config, device, 1)

            test_data_iters = {
                "od": od_test_data_iter,
                "sod": sod_test_data_iter,
                "seg": ss_data_iter,
            }

            with torch.no_grad():
                torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)

                bfn = getattr(bfn_model, bfn_model_name)
                bfn = bfn(num_blocks=args.num_blocks)
                bfn.load_state_dict(torch.load(args.bfn_ckpt_dir, map_location='cpu')['net'])

                t_fh = getattr(fh_model, t_fh_model_name)
                rsc_ir = t_fh(num_blocks=args.num_blocks - 1, embed_dim=args.embed_dim, inC=args.inC, filter_num=args.filter_num, vector_num=args.vector_num, set_num=args.set_num)
                rsc_vi = t_fh(num_blocks=args.num_blocks - 1, embed_dim=args.embed_dim, inC=args.inC, filter_num=args.filter_num, vector_num=args.vector_num, set_num=args.set_num)

                yolov5 = load_yolov5_model(args.yolov5_ckpt_dir, args.yolov5_data_yaml, device)

                seg = load_segformer_model(args.segformer_config, args.segformer_ckpt_dir, device)

                ctdnet = load_ctdnet_model(mode='train', ckpts_dir=args.ctdnet_ckpt_dir)

            utils.to_inference(bfn, device)
            utils.to_inference(yolov5, device)
            utils.to_inference(seg, device)
            utils.to_inference(ctdnet, device)

            checkpoint = torch.load(
                os.path.join(r'...', ckpts_idx),
                map_location='cpu')

            rsc_ir.load_state_dict(checkpoint['rsc_ir'], strict=True)
            rsc_vi.load_state_dict(checkpoint['rsc_vi'], strict=True)

            utils.to_inference(rsc_ir, device)
            utils.to_inference(rsc_vi, device)

            basis_vector_bank = checkpoint['bvb']
            basis_vector_bank.requires_grad = False
            basis_vector_bank = basis_vector_bank.to(device)
            basis_vector_bank = basis_vector_bank.view(args.set_num, args.vector_num, args.embed_dim)

            networks = {
                'od_net': yolov5,
                'seg_net': seg,
                'sod_net': ctdnet,
                'bfn': bfn,
                'rsc_ir': rsc_ir,
                'rsc_vi': rsc_vi,
                'bvb': basis_vector_bank
            }

            save_inf = {
                'save_finetune_fusion_dir': save_finetune_fusion_dir,
                'save_wofinetune_fusion_dir': save_wofinetune_fusion_dir,
            }

            test(networks, save_inf, t_info, test_data_iters)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()