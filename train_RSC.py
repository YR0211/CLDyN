import multiprocessing
import os

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.utils import save_image
from tqdm import tqdm

from args_RSC import args as args
from ctdnet.src.load_model import load_ctdnet_model
from ctdnet.src.loss import CTDNet_Loss
from models import RSC as fh_model
from models import VFN as vfn_model
from models.dataset import TrainDataset_od, TrainDataset_sod
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


def each_step(x, networks, criterions, task_name, order):
    with torch.no_grad():
        if task_name == 'od':
            vi = x[0].to(device).bfloat16().detach()
            ir = x[1].to(device).bfloat16().detach()
            labels = x[2].to(device).detach()
        elif task_name == 'seg':
            ir = (x['img'].data[0][:, 0:3, ...] / 255.).to(device).bfloat16().detach()
            vi = (x['img'].data[0][:, 3:, ...] / 255.).to(device).bfloat16().detach()
        elif task_name == 'sod':
            ir = x[0].to(device).bfloat16().detach()
            vi = x[1].to(device).bfloat16().detach()
            mask = x[2].to(device).bfloat16().detach()
            edge = x[3].to(device).bfloat16().detach()

        vi_ycbcr = utils.rgb2ycbcr(vi)
        vi_Y, vi_Cb, vi_Cr = torch.split(vi_ycbcr, 1, 1)
        ir = ir[:, :1, :, :]

        f_img_wofinetune = networks['vfn'].forward_ER(ir, vi_Y)
        f_img_wofinetune_c = utils.ycbcr2rgb(torch.cat([f_img_wofinetune, vi_Cb, vi_Cr], dim=1))

        basis_vector_bank = networks['bvb']

    if task_name == 'od':
        target_layer = 'model.model.23.cv3'
        input_tensor = {
            'f_img': f_img_wofinetune_c,
            'label': labels
        }
    elif task_name == 'seg':
        target_layer = 'module.backbone.norm4'
        input_tensor = {
            'f_img': f_img_wofinetune_c,
            'x': x
        }
    elif task_name == 'sod':
        target_layer = 'fuse23.bn_2'
        input_tensor = {
            'f_img': f_img_wofinetune_c,
            'mask': mask,
            'edge': edge,
        }
    task_f, prefine_loss = utils.get_activation(networks[task_name + "_net"], target_layer, task_name,
                                                  input_tensor, criterions[task_name])
    task_f = task_f.detach().bfloat16()
    prefine_loss = prefine_loss.detach()

    for i in range(args.num_blocks):
        if i == 0:
            ir_f_h = ir
            vi_f_h = vi_Y

        ir_f = networks['vfn'].ir_e.forward_obo(ir_f_h, i)
        vi_f = networks['vfn'].vi_e.forward_obo(vi_f_h, i)

        if i != args.num_blocks - 1:
            ir_f_h = networks['rsc_ir'].forward_obo(basis_vector_bank, task_f, ir_f, i)
            vi_f_h = networks['rsc_vi'].forward_obo(basis_vector_bank, task_f, vi_f, i)

    f_f = networks['vfn'].fusion(ir_f, vi_f)

    for i in range(args.num_blocks):
        f_f = networks['vfn'].fb.forward_obo(f_f, i)

    f_img = f_f
    assert f_img.size(1) == 1

    f_img_c = utils.ycbcr2rgb(torch.cat([f_img, vi_Cb, vi_Cr], dim=1))

    f_img_dict = {
        'f_img_c_finetune': f_img_c,
        'f_img_c_wofinetune': f_img_wofinetune_c,
    }

    if task_name == 'od':
        f_img_c = f_img_c.float()
        preds, train_out = networks[task_name + "_net"](f_img_c, augment=False, visualize=False)
        t_loss, loss_item = criterions[task_name](train_out, labels)
        loss = t_loss

        total_loss = {
            'lbox': loss_item[0].item(),
            'lobj': loss_item[1].item(),
            'lcls': loss_item[2].item(),
            't_loss': t_loss.item(),
            'pen_loss': 0
        }

    elif task_name == 'seg':
        mean = torch.from_numpy(np.array([123.675, 116.28, 103.53])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(device)
        std = torch.from_numpy(np.array([58.395, 57.12, 57.375])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(device)

        f_img_c1 = f_img_c * 255.
        f_img_c1 = (f_img_c1 - mean) / std

        x['img'].data[0] = f_img_c1

        outputs = networks[task_name + "_net"](return_loss=True, **x)
        t_loss = outputs.get('decode.loss_seg')
        acc = outputs.get('decode.acc_seg')
        loss = t_loss

        total_loss = {
            'acc': acc.item(),
            't_loss': t_loss.item(),
            'pen_loss': 0
        }

    elif task_name == 'sod':
        mean = torch.from_numpy(np.array([124.55, 118.90, 102.94])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(device)
        std = torch.from_numpy(np.array([56.77, 55.97, 57.50])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(device)

        f_img_c1 = f_img_c * 255.
        f_img_c1 = (f_img_c1 - mean) / std

        out1, out_edge, out2, out3, out4, out5 = networks[task_name + "_net"](f_img_c1)
        t_loss = criterions[task_name](out1, out_edge, out2, out3, out4, out5, mask, edge)
        loss = t_loss

        total_loss = {
            't_loss': t_loss.item(),
            'pen_loss': 0
        }

    pen_loss = F.relu(t_loss - prefine_loss)

    total_loss['pen_loss'] = pen_loss.item()

    loss = loss + 5 * pen_loss

    retain = (task_name != order[-1])
    loss.backward(retain_graph=retain)

    return total_loss, f_img_dict


def train_TFH(epoch, optimizers, criterions, networks, save_inf, dataiters, order, lam, best_losses):
    networks['bvb'].requires_grad_(True)
    utils.to_train(networks['rsc_ir'], device)
    utils.to_train(networks['rsc_vi'], device)

    utils.clean_grad(networks["od_net"])
    utils.clean_grad(networks["seg_net"])
    utils.clean_grad(networks["sod_net"])

    epoch_t_od_loss = []
    epoch_t_seg_loss = []
    epoch_t_sod_loss = []
    epoch_lbox = []
    epoch_lobj = []
    epoch_lcls = []
    epoch_acc = []
    epoch_pen_od_loss = []
    epoch_pen_seg_loss = []
    epoch_pen_sod_loss = []

    iters = {t: iter(dataiters[t]) for t in dataiters}
    optimizers['rsc_optimizer'].zero_grad(set_to_none=True)

    for step in tqdm(range(save_inf['steps_per_epoch'])):
        g_list = []
        all_params = sum([g['params'] for g in optimizers['rsc_optimizer'].param_groups], [])
        for task_name in order:
            x = next(iters[task_name])

            total_loss, f_img_dict = each_step(x, networks, criterions, task_name, order)
            g_list.append(utils.flatten_gradients(all_params))
            optimizers['rsc_optimizer'].zero_grad(set_to_none=True)

            if task_name == 'od':
                epoch_lbox.append(total_loss['lbox'])
                epoch_lobj.append(total_loss['lobj'])
                epoch_lcls.append(total_loss['lcls'])
                epoch_t_od_loss.append(total_loss['t_loss'])
                epoch_pen_od_loss.append(total_loss['pen_loss'])
            elif task_name == 'seg':
                epoch_acc.append(total_loss['acc'])
                epoch_t_seg_loss.append(total_loss['t_loss'])
                epoch_pen_seg_loss.append(total_loss['pen_loss'])
            elif task_name == 'sod':
                epoch_t_sod_loss.append(total_loss['t_loss'])
                epoch_pen_sod_loss.append(total_loss['pen_loss'])

            if step % save_inf['save_image_iter'] == 0:
                epoch_step_name = str(epoch) + "epoch" + str(step) + "step_" + task_name
                if epoch % 10 == 0:
                    output_name = save_inf['save_img_dir'] + "/" + epoch_step_name + ".jpg"
                    out = f_img_dict['f_img_c_finetune']
                    save_image(out, output_name)

                    output_name = save_inf['save_img_dir'] + "/" + epoch_step_name + "_wofinetune" + ".jpg"
                    out = f_img_dict['f_img_c_wofinetune']
                    save_image(out, output_name)

        G = torch.stack(g_list)
        alpha = utils.solve_cagrad(G, lam=lam)

        g_final = (alpha.unsqueeze(1) * G).sum(dim=0)
        utils.unflatten_to_params(g_final, all_params)

        torch.nn.utils.clip_grad_norm_(networks['rsc_ir'].parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(networks['rsc_vi'].parameters(), max_norm=1.0)
        optimizers['rsc_optimizer'].step()
        optimizers['rsc_optimizer'].zero_grad(set_to_none=True)

        if ((epoch + 1) == args.Epoch and (step + 1) % save_inf['steps_per_epoch'] == 0) or (
                epoch % args.save_model_num == 0 and (step + 1) % save_inf['steps_per_epoch'] == 0):
            ckpt = {
                "rsc_ir": networks['rsc_ir'].state_dict(),
                "rsc_vi": networks['rsc_vi'].state_dict(),
                'rsc_optimizer': optimizers['rsc_optimizer'].state_dict(),
                'bvb': networks['bvb'],
            }
            torch.save(ckpt, os.path.join(save_inf['save_model_dir'], 'ckpt_%s.pth' % (str(epoch))))


    epoch_t_od_loss_mean = np.mean(epoch_t_od_loss)
    epoch_t_seg_loss_mean = np.mean(epoch_t_seg_loss)
    epoch_t_sod_loss_mean = np.mean(epoch_t_sod_loss)
    epoch_pen_od_loss_mean = np.mean(epoch_pen_od_loss)
    epoch_pen_seg_loss_mean = np.mean(epoch_pen_seg_loss)
    epoch_pen_sod_loss_mean = np.mean(epoch_pen_sod_loss)
    epoch_lbox_mean = np.mean(epoch_lbox)
    epoch_lobj_mean = np.mean(epoch_lobj)
    epoch_lcls_mean = np.mean(epoch_lcls)
    epoch_acc_mean = np.mean(epoch_acc)

    cur_losses = {
        "od": epoch_t_od_loss_mean,
        "seg": epoch_t_seg_loss_mean,
        "sod": epoch_t_sod_loss_mean,
    }
    is_best_save = utils.is_best(cur_losses, best_losses)
    if is_best_save:
        best_losses.update(cur_losses)
        ckpt = {
            "rsc_ir": networks['rsc_ir'].state_dict(),
            "rsc_vi": networks['rsc_vi'].state_dict(),
            'rsc_optimizer': optimizers['rsc_optimizer'].state_dict(),
            'bvb': networks['bvb'],
            "cur_losses": cur_losses
        }
        torch.save(ckpt, os.path.join(save_inf['save_model_dir'], 'ckpt_best_%s.pth' % (str(epoch))))


    print("\n -epoch " + str(epoch))
    print(" -loss_t_od_loss " + str(epoch_t_od_loss_mean))
    print(" -loss_t_seg_loss " + str(epoch_t_seg_loss_mean))
    print(" -loss_t_sod_loss " + str(epoch_t_sod_loss_mean))
    print(" -loss_pen_od_loss " + str(epoch_pen_od_loss_mean))
    print(" -loss_pen_seg_loss " + str(epoch_pen_seg_loss_mean))
    print(" -loss_pen_sod_loss " + str(epoch_pen_sod_loss_mean))
    print(" -loss_lbox " + str(epoch_lbox_mean))
    print(" -loss_lobj " + str(epoch_lobj_mean))
    print(" -loss_lcls " + str(epoch_lcls_mean))
    print(" -loss_acc " + str(epoch_acc_mean))

    with open(save_inf['save_loss_dir'] + "/" + "loss.txt", 'a') as f:
        f.writelines(" -epoch " + str(epoch) + '\n')
        f.writelines(
            " -loss_t_od_loss " + str(epoch_t_od_loss_mean) + '\n')
        f.writelines(
            " -loss_t_seg_loss " + str(epoch_t_seg_loss_mean) + '\n')
        f.writelines(
            " -loss_t_sod_loss " + str(epoch_t_sod_loss_mean) + '\n')
        f.writelines(
            " -loss_pen_od_loss " + str(epoch_pen_od_loss_mean) + '\n')
        f.writelines(
            " -loss_pen_seg_loss " + str(epoch_pen_seg_loss_mean) + '\n')
        f.writelines(
            " -loss_pen_sod_loss " + str(epoch_pen_sod_loss_mean) + '\n')
        f.writelines(
            " -loss_lbox " + str(epoch_lbox_mean) + '\n')
        f.writelines(
            " -loss_lobj " + str(epoch_lobj_mean) + '\n')
        f.writelines(
            " -loss_lcls " + str(epoch_lcls_mean) + '\n')
        f.writelines(
            " -loss_acc " + str(epoch_acc_mean) + '\n')
        f.close()


def main():
    vfn_model_name = "VFN"
    t_fh_model_name = "RSC"

    nowTime = utils.get_time()
    save_model_dir = os.path.join(args.train_save_dir, "train_models",
                                  nowTime + "_" + t_fh_model_name + "_model")
    save_img_dir = os.path.join(args.train_save_dir, "images",
                                nowTime + "_" + t_fh_model_name + "_img")
    save_loss_dir = os.path.join(str(save_img_dir), "loss_map")
    save_model_info_dir = os.path.join(str(save_img_dir), "model_info")
    check_dir(save_loss_dir)
    check_dir(save_model_info_dir)
    check_dir(save_model_dir)
    check_dir(save_img_dir)

    od_vis_M3FD_train_dir = os.path.join(args.od_dataset_dir, 'vi', 'train')
    od_ir_M3FD_train_dir = os.path.join(args.od_dataset_dir, 'ir', 'train')
    od_label_M3FD_train_dir = os.path.join(args.od_dataset_dir, 'labels', 'train')

    od_vis_dir_list_train = [od_vis_M3FD_train_dir]
    od_ir_dir_list_train = [od_ir_M3FD_train_dir]
    od_label_dir_list_train = [od_label_M3FD_train_dir]
    od_dataset = TrainDataset_od(od_vis_dir_list_train, od_ir_dir_list_train, od_label_dir_list_train,
                                 img_size=[640, 640])
    od_data_iter = data.DataLoader(
        dataset=od_dataset,
        shuffle=True,
        batch_size=args.batch_size,  
        num_workers=args.batch_size,
        collate_fn=TrainDataset_od.collate_fn,
        generator=torch.Generator(device=device)
    )

    ss_dataset, ss_data_iter = load_dataiter(args.segformer_config, device, args.batch_size)

    sod_vis_dir_list_train = os.path.join(args.sod_dataset_dir, 'Train', 'RGB')
    sod_ir_dir_list_train = os.path.join(args.sod_dataset_dir, 'Train', 'T_GRAY')
    sod_label_train_dir = os.path.join(args.sod_dataset_dir, 'Train', 'GT')
    sod_edge_train_dir = os.path.join(args.sod_dataset_dir, 'Train', 'Edge')
    sod_dataset = TrainDataset_sod(sod_ir_dir_list_train, sod_vis_dir_list_train, sod_label_train_dir,
                                   sod_edge_train_dir)
    sod_data_iter = data.DataLoader(
        sod_dataset,
        collate_fn=sod_dataset.collate,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.batch_size,
        generator=torch.Generator(device=device)
    )

    dataiters = {
        'od': od_data_iter,
        'seg': ss_data_iter,
        'sod': sod_data_iter
    }
    steps_per_epoch = min(len(l) for l in dataiters.values())

    with torch.no_grad():
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)

        vfn = getattr(vfn_model, vfn_model_name)
        vfn = vfn(num_blocks=args.num_blocks)
        vfn.load_state_dict(torch.load(args.vfn_ckpt_dir, map_location='cpu')['net'])

        t_fh = getattr(fh_model, t_fh_model_name)
        rsc_ir = t_fh(num_blocks=args.num_blocks - 1, embed_dim=args.embed_dim, inC=args.inC, filter_num=args.filter_num, vector_num=args.vector_num, set_num=args.set_num)
        rsc_vi = t_fh(num_blocks=args.num_blocks - 1, embed_dim=args.embed_dim, inC=args.inC, filter_num=args.filter_num, vector_num=args.vector_num, set_num=args.set_num)

        yolov5 = load_yolov5_model(args.yolov5_ckpt_dir, args.yolov5_data_yaml, device)

        od_critertion = ComputeLoss(yolov5)

        seg = load_segformer_model(args.segformer_config, args.segformer_ckpt_dir, device)

        ctdnet = load_ctdnet_model(mode='train', ckpts_dir=args.ctdnet_ckpt_dir)

        sod_critertion = CTDNet_Loss()

    basis_vector_bank = utils.make_bank(args.embed_dim, args.vector_num * args.set_num, dtype=torch.bfloat16, device=device)

    utils.to_inference(vfn, device)
    utils.to_inference(yolov5, device)
    utils.to_inference(seg, device)
    utils.to_inference(ctdnet, device)

    utils.to_train(rsc_ir, device)
    utils.to_train(rsc_vi, device)

    rsc_optimizer = torch.optim.Adam(
        [{'params': rsc_ir.parameters()}, {'params': rsc_vi.parameters()}, {'params': basis_vector_bank}], lr=args.LR,
        weight_decay=1e-4)

    basis_vector_bank = basis_vector_bank.view(args.set_num, args.vector_num, args.embed_dim)

    networks = {
        'od_net': yolov5,
        'seg_net': seg,
        'sod_net': ctdnet,
        'vfn': vfn,
        'rsc_ir': rsc_ir,
        'rsc_vi': rsc_vi,
        'bvb': basis_vector_bank
    }

    optimizers = {
        'rsc_optimizer': rsc_optimizer,
    }
    criterions = {
        'od': od_critertion,
        'seg': None,
        'sod': sod_critertion,
    }

    save_image_iter = max(1, steps_per_epoch // args.save_image_num)

    save_inf = {
        'save_image_iter': save_image_iter,
        'save_model_dir': save_model_dir,
        'save_loss_dir': save_loss_dir,
        'save_img_dir': save_img_dir,
        "steps_per_epoch": steps_per_epoch
    }

    best_losses = {
        "od": float("inf"),
        "seg": float("inf"),
        "sod": float("inf"),
    }

    order = ['od', 'seg', 'sod']

    with open(save_model_info_dir + "/" + "model_info.txt", 'w') as f:
        f.writelines('--------------- start ---------------' + '\n')
        f.writelines('model_name' + '\n')
        f.writelines(t_fh_model_name)
        for i in rsc_ir.children():
            f.writelines(str(i) + '\n')
        f.writelines('---------------- end ----------------' + '\n')
        f.close()

    utils.save_args_to_txt(args, save_model_info_dir + "/" + "args_info.txt")

    start_epoch = -1
    for epoch in tqdm(range(start_epoch + 1, args.Epoch)):
        train_TFH(epoch, optimizers, criterions, networks,
                  save_inf, dataiters, order, args.lam, best_losses)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()