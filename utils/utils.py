import os
import time

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from PIL import Image

import args


def check_dir(base):
    if os.path.isdir(base):
        pass
    else:
        os.makedirs(base)


def rgb2ycbcr(img_rgb):
    b, _, _, _ = img_rgb.shape
    R = torch.unsqueeze(img_rgb[:, 0, :, :], dim=1)
    G = torch.unsqueeze(img_rgb[:, 1, :, :], dim=1)
    B = torch.unsqueeze(img_rgb[:, 2, :, :], dim=1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    img_ycbcr = torch.cat([Y, Cb, Cr], dim=1)
    return img_ycbcr


def ycbcr2rgb(img_ycbcr):
    b, _, _, _ = img_ycbcr.shape
    Y = torch.unsqueeze(img_ycbcr[:, 0, :, :], dim=1)
    Cb = torch.unsqueeze(img_ycbcr[:, 1, :, :], dim=1)
    Cr = torch.unsqueeze(img_ycbcr[:, 2, :, :], dim=1)
    R = Y + 1.402 * (Cr - 128 / 255.0)
    G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    B = Y + 1.772 * (Cb - 128 / 255.0)
    img_rgb = torch.cat([R, G, B], dim=1)
    return img_rgb


def to_inference(model, device):
    model.eval()
    model.to(device)
    for param in model.named_parameters():
        param[1].requires_grad = False


def to_train(model, device):
    model.train()
    model.to(device)
    for param in model.named_parameters():
        param[1].requires_grad = True


def clean_grad(model):
    for param in model.parameters():
        param.grad = None


def get_time():
    now = int(time.time())
    timeArr = time.localtime(now)
    nowTime = time.strftime("%Y%m%d_%H-%M-%S", timeArr)
    return nowTime


def get_activation(
        model: nn.Module,
        target_layer: str,
        task_name: str,
        input_tensor: torch.Tensor,
        criterions
):
    activation = None

    def forward_hook(out):
        nonlocal activation
        activation = out.detach()

    forward_hook_handle = None
    for name, module in model.named_modules():
        if name == target_layer:
            forward_hook_handle = module.register_forward_hook(forward_hook)
            break

    if forward_hook_handle is None:
        raise ValueError(f"Layer '{target_layer}' not found in the model.")

    if task_name == 'od':
        f_img_c = input_tensor['f_img'].float()
        preds, train_out = model(f_img_c, augment=False, visualize=False)

        prefine_loss, _ = criterions(train_out, input_tensor['label'])

    elif task_name == 'seg':
        mean = torch.from_numpy(np.array([123.675, 116.28, 103.53])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(input_tensor['f_img'].device)
        std = torch.from_numpy(np.array([58.395, 57.12, 57.375])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(input_tensor['f_img'].device)

        f_img_c1 = input_tensor['f_img'] * 255.
        f_img_c1 = (f_img_c1 - mean) / std

        input_tensor['x']['img'].data[0] = f_img_c1

        outputs = model(return_loss=True, **input_tensor['x'])

        prefine_loss = outputs.get('decode.loss_seg')

    elif task_name == 'sod':
        mean = torch.from_numpy(np.array([124.55, 118.90, 102.94])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(input_tensor['f_img'].device)
        std = torch.from_numpy(np.array([56.77, 55.97, 57.50])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(input_tensor['f_img'].device)

        f_img_c1 = input_tensor['f_img'] * 255.
        f_img_c1 = (f_img_c1 - mean) / std

        out1, out_edge, out2, out3, out4, out5 = model(f_img_c1)

        prefine_loss = criterions(out1, out_edge, out2, out3, out4, out5, input_tensor['mask'], input_tensor['edge'])

    forward_hook_handle.remove()

    return activation, prefine_loss


def get_activation_test(
        model: nn.Module,
        target_layer: str,
        t_info: dict,
        input_tensor: torch.Tensor,
):
    activation = None

    def forward_hook(out):
        nonlocal activation
        activation = out.detach()

    forward_hook_handle = None
    for name, module in model.named_modules():
        if name == target_layer:
            forward_hook_handle = module.register_forward_hook(forward_hook)
            break

    if forward_hook_handle is None:
        raise ValueError(f"Layer '{target_layer}' not found in the model.")

    if t_info['is_TD']:
        f_img_c = input_tensor['f_img'].float()
        new_h = (f_img_c.size(2) + 31) // 32 * 32
        new_w = (f_img_c.size(3) + 31) // 32 * 32
        f_img_c = F.interpolate(f_img_c, size=(new_h, new_w), mode='bilinear', align_corners=False)

        _ = model(f_img_c, augment=False, visualize=False)

    elif t_info['is_SS']:
        mean = torch.from_numpy(np.array([123.675, 116.28, 103.53])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(input_tensor['f_img'].device)
        std = torch.from_numpy(np.array([58.395, 57.12, 57.375])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(input_tensor['f_img'].device)

        f_img_c1 = input_tensor['f_img'] * 255.
        f_img_c1 = (f_img_c1 - mean) / std

        input_tensor['x']['img'].data[0] = f_img_c1

        _ = model(return_loss=True, **input_tensor['x'])

    elif t_info['is_SOD']:
        mean = torch.from_numpy(np.array([124.55, 118.90, 102.94])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(input_tensor['f_img'].device)
        std = torch.from_numpy(np.array([56.77, 55.97, 57.50])).bfloat16().unsqueeze(dim=0).unsqueeze(
            dim=-1).unsqueeze(dim=-1).to(input_tensor['f_img'].device)

        f_img_c1 = input_tensor['f_img'] * 255.
        f_img_c1 = (f_img_c1 - mean) / std

        f_img_c1 = torch.cat([f_img_c1, f_img_c1], dim=0)

        _ = model(f_img_c1)

    forward_hook_handle.remove()

    if t_info['is_SS'] or t_info['is_TD']:
        return activation
    elif t_info['is_SOD']:
        return activation[0:1, ...]


def read_image(path, transform, img_type='RGB'):
    img = Image.open(str(path)).convert(img_type)
    img = transform(img)
    return img


def get_same_padding(kernel_size, dilation=1, stride=1):
    effective_kernel = dilation * (kernel_size - 1) + 1
    padding = (effective_kernel - stride) // 2
    return int(padding)


def make_bank(embed_dim, vector_num, requires_grad=True, dtype=torch.bfloat16, device=None):

    if device is None:
        w = torch.empty(embed_dim, vector_num, dtype=torch.float32)
    else:
        w = torch.empty(embed_dim, vector_num, dtype=torch.float32, device=device)
    nn.init.orthogonal_(w)
    w = w.t().contiguous().clone()
    w = w / w.norm(dim=1, keepdim=True)
    return nn.Parameter(w.to(dtype), requires_grad=requires_grad)


def flatten_gradients(params):
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.detach().clone().view(-1))
    return torch.cat(grads)


def unflatten_to_params(vec, params):
    offset = 0
    for p in params:
        numel = p.numel()
        if p.grad is None:
            p.grad = vec.new_zeros(p.shape)
        p.grad.copy_(vec[offset: offset + numel].view_as(p))
        offset += numel


def project_simplex(v):
    if v.sum() == 1.0 and (v >= 0).all():
        return v
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(v.numel(), device=v.device) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = torch.clamp(v - theta, min=0)
    return w


def solve_cagrad(G, lam=0.5, eps=1e-8):
    T = G.size(0)
    GG = G @ G.t()
    pc = GG + eps * torch.eye(T, device=G.device)

    Q = 2 * pc

    g_min, _ = GG.min(dim=1, keepdim=True)
    c = -2 * lam * g_min.squeeze(1)

    alpha = torch.ones(T, device=G.device) / T
    lr, iters = 0.1, 250
    for _ in range(iters):
        grad = Q @ alpha + c
        alpha = project_simplex(alpha - lr * grad)
    return alpha


def is_best(cur_losses , best_losses, eps = 0.05):
    within_eps_all = (
            cur_losses["od"] <= best_losses["od"] * (1 + eps) and
            cur_losses["seg"] <= best_losses["seg"] * (1 + eps) and
            cur_losses["sod"] <= best_losses["sod"] * (1 + eps)
    )

    improvements = [
        cur_losses["od"] < best_losses["od"],
        cur_losses["seg"] < best_losses["seg"],
        cur_losses["sod"] < best_losses["sod"]
    ]
    num_improved = sum(improvements)

    should_save = (num_improved >= 2) and within_eps_all

    return should_save

def save_args_to_txt(args, filepath="args_record.txt"):
    with open(filepath, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")