import torch.nn as nn
import torch
import torch.nn.functional as F
import utils.utils as utils

class A2SI(nn.Module):
    def __init__(self, embed_dim, inC, filter_num, set_num, vector_num):
        super(A2SI, self).__init__()
        self.filter_num = filter_num
        self.set_num = set_num
        self.vector_num = vector_num
        self.embed_dim = embed_dim
        self.inC = inC

        self.orthogonal_convolutional_prototypes_embed_dim = 16
        self.orthogonal_convolutional_prototypes = utils.make_bank(self.orthogonal_convolutional_prototypes_embed_dim, set_num, requires_grad=False, dtype=torch.bfloat16)

        self.bv_f = BasisV_filter(embed_dim, inC, filter_num)

        self.p_task_f = nn.Sequential(
            nn.Conv1d(1, int(self.orthogonal_convolutional_prototypes_embed_dim / 4), kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv1d(int(self.orthogonal_convolutional_prototypes_embed_dim / 4), int(self.orthogonal_convolutional_prototypes_embed_dim / 2),
                      kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.p_img_f = nn.Sequential(
            nn.Conv2d(inC, int(self.orthogonal_convolutional_prototypes_embed_dim / 2),
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(self.orthogonal_convolutional_prototypes_embed_dim / 2)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.a_task_img = nn.Sequential(
            nn.Linear(int(self.orthogonal_convolutional_prototypes_embed_dim),
                      int((filter_num / 2) * self.orthogonal_convolutional_prototypes_embed_dim)),
            nn.LayerNorm(int((filter_num / 2) * self.orthogonal_convolutional_prototypes_embed_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int((filter_num / 2) * self.orthogonal_convolutional_prototypes_embed_dim),
                      int(filter_num * self.orthogonal_convolutional_prototypes_embed_dim)),
        )

        self.mlp_1x1 = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim / 4)),
            nn.LayerNorm(int(embed_dim / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embed_dim / 4), 1 * 1 * inC * inC),
        )

        self.mlp_3x3 = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim / 4)),
            nn.LayerNorm(int(embed_dim / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embed_dim / 4), 3 * 3 * inC * inC),
        )

        self.B_LR = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.LeakyReLU(0.2),
        )

        self.zc = self.zero_convolution(inC, inC)

    def zero_convolution(self, inC, outC, kernel_size=1, stride=1):
        zc = nn.Conv2d(inC, outC, kernel_size=kernel_size, stride=stride, bias=True)
        nn.init.zeros_(zc.weight)
        nn.init.zeros_(zc.bias)
        return zc

    def process_task_f(self, x, p_task_f):
        x = x.view(x.size(0), x.size(1), -1)
        b, c, e = x.size()
        out = p_task_f(x.view(b, 1, c * e))
        return out.mean(dim=2, keepdim=True)

    def forward(self, basis_vector_bank, task_f, img_f):
        B = img_f.size(0)
        task_f_1 = self.process_task_f(task_f, self.p_task_f).squeeze(dim=-1)
        img_f_1 = self.p_img_f(img_f).squeeze(dim=-1).squeeze(dim=-1)
        task_img_f_1 = self.a_task_img(torch.cat([task_f_1, img_f_1], dim=1)).view(B, self.orthogonal_convolutional_prototypes_embed_dim,
                                                                                   self.filter_num)

        set_type = torch.bmm(F.normalize(self.orthogonal_convolutional_prototypes.unsqueeze(dim=0).expand(B, -1, -1), p=2, dim=-1, eps=1e-8),
                             F.normalize(task_img_f_1, p=2, dim=1, eps=1e-8)).permute(0, 2, 1)

        set_type = F.gumbel_softmax(set_type, tau=0.5, hard=True, dim=-1)

        set_type_indices = set_type.argmax(dim=-1)

        basis_vector_bank = basis_vector_bank.unsqueeze(dim=0).expand(B, -1, -1, -1).view(B, self.set_num, self.vector_num * self.embed_dim)
        set_basis_vector_bank = torch.bmm(set_type, basis_vector_bank).view(B, self.filter_num, self.vector_num, self.embed_dim)

        basis_vector_filtered = self.bv_f(set_basis_vector_bank, task_f, img_f, set_type_indices)

        b_dps = []
        for b in range(B):
            dps = []
            for i in range(self.filter_num):
                indice = set_type_indices[b, i]
                if indice == 0:
                    kernel_size = 1

                    conv_weight = self.mlp_1x1(basis_vector_filtered[b:b + 1, i:i + 1, :])
                    conv_weight = conv_weight.view(self.inC, self.inC, kernel_size, kernel_size)

                    dp = F.conv2d(img_f[b:b + 1, ...], conv_weight, stride=1, padding=0, dilation=1,
                                  groups=1)

                elif indice == 1 or indice == 2 or indice == 3:
                    kernel_size = 3
                    dilation = int(1 + ((indice - 1) % 3))

                    conv_weight = self.mlp_3x3(basis_vector_filtered[b:b + 1, i:i + 1, :])
                    conv_weight = conv_weight.view(self.inC, self.inC, kernel_size, kernel_size)

                    dp = F.conv2d(img_f[b:b + 1, ...], conv_weight, stride=1,
                                  padding=utils.get_same_padding(kernel_size, dilation, 1), dilation=dilation,
                                  groups=1)

                dps.append(dp)
            dps = torch.cat(dps, dim=0)
            b_dps.append(dps)
        b_dps = torch.stack(b_dps, dim=0)

        B, f_n, c, h, w = b_dps.size()
        b_dps = b_dps.view(-1, c, h, w)
        b_dps = self.B_LR(b_dps)
        b_dps = b_dps.view(B, f_n, c, h, w)
        b_dps = b_dps.mean(dim=1)

        a_dp = self.zc(b_dps)

        img_h = a_dp + img_f

        return img_h


class BasisV_filter(nn.Module):
    def __init__(self, embed_dim, inC, filter_num):
        super(BasisV_filter, self).__init__()
        self.p_task_f = nn.Sequential(
            nn.Conv1d(1, int(embed_dim / 4), kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv1d(int(embed_dim / 4), int(embed_dim / 2), kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.p_img_f = nn.Sequential(
            nn.Conv2d(inC, int(embed_dim / 4), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(embed_dim / 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(int(embed_dim / 4), int(embed_dim / 2), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(embed_dim / 2)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim / 4)),
            nn.LayerNorm(int(embed_dim / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embed_dim / 4), embed_dim),
        )
        self.filter_num = filter_num

    def global_rms_norm(self, x: torch.Tensor, eps: float = 1e-8):
        rms = x.pow(2).mean(dim=[1, 2], keepdim=True).sqrt()
        return x / (rms + eps)

    def process_task_f(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = self.global_rms_norm(x)
        b, c, e = x.size()
        out = self.p_task_f(x.view(b, 1, c * e))
        return out.mean(dim=2, keepdim=True)

    def cosine_sim(self, a, b):
        return torch.matmul(F.normalize(a, dim=-1), F.normalize(b, dim=-1).T)

    def forward(self, basis_vector_bank, task_f, img_f, set_type_indices):
        B, F, V, D = basis_vector_bank.shape
        device = task_f.device
        dtype = basis_vector_bank.dtype

        task_emb = self.process_task_f(task_f).squeeze(dim=-1)
        img_emb = self.p_img_f(img_f).squeeze(dim=-1).squeeze(dim=-1)
        filter_f = self.mlp(torch.cat([task_emb, img_emb], dim=1))
        filter_f = filter_f.unsqueeze(1).expand(-1, F, -1)

        out = torch.empty(B, F, D, device=device, dtype=dtype)

        for b in range(B):
            types = torch.unique(set_type_indices[b])
            for t in types:
                mask = (set_type_indices[b] == t)
                idx_t = mask.nonzero(as_tuple=False).squeeze(-1)
                K_t = idx_t.numel()
                if K_t == 0:
                    continue

                Q = filter_f[b, idx_t]
                B_t = basis_vector_bank[b, idx_t[0]]

                S = self.cosine_sim(Q, B_t)
                sel = S.max(0).values.topk(min(K_t, V)).indices
                basis_sel = B_t[sel]
                out[b, idx_t] = basis_sel

        return out