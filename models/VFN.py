import kornia.filters as KF
import torch
import torch.nn as nn
import torch.nn.functional as F

class CB(nn.Module):
    def __init__(self, in_ch=8, growth_rate=8):
        super(CB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Sequential(
            nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2),
            nn.BatchNorm2d(growth_rate),
            nn.LeakyReLU(0.2)
        )
        in_ch_ += growth_rate
        self.Dcov2 = nn.Sequential(
            nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2),
            nn.BatchNorm2d(growth_rate),
            nn.LeakyReLU(0.2)
        )
        in_ch_ += growth_rate
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.conv(x2)
        out = x + F.leaky_relu(x3, 0.2)
        return out


class FRB1(nn.Module):
    def __init__(self, inc):
        super(FRB1, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.mean_pool = nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
        self.mlp_aa = nn.Sequential(
            nn.Linear(inc, inc * 2),
            nn.ReLU(),
            nn.Linear(inc * 2, inc)
        )
        self.mlp_ma = nn.Sequential(
            nn.Linear(inc, inc * 2),
            nn.ReLU(),
            nn.Linear(inc * 2, inc)
        )
        self.conv_mean_a = nn.Sequential(
            nn.Conv2d(inc, inc * 2, kernel_size=(3, 3), stride=1, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(inc * 2, inc, kernel_size=(3, 3), stride=1, padding=1, padding_mode="reflect")
        )
        self.conv_max_a = nn.Sequential(
            nn.Conv2d(inc, inc * 2, kernel_size=(3, 3), stride=1, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(inc * 2, inc, kernel_size=(3, 3), stride=1, padding=1, padding_mode="reflect")
        )

    def channel_attention(self, x):
        aa = self.gap(x).squeeze(dim=-1).squeeze(dim=-1)
        aa = self.mlp_aa(aa)
        ma = self.gmp(x).squeeze(dim=-1).squeeze(dim=-1)
        ma = self.mlp_ma(ma)
        ca = ma + aa
        return ca.unsqueeze(dim=-1).unsqueeze(dim=-1)

    def spatial_attention(self, x):
        mean_a = self.mean_pool(x)
        mean_a = self.conv_mean_a(mean_a)
        max_a = self.max_pool(x)
        max_a = self.conv_max_a(max_a)
        sa = mean_a + max_a
        return sa

    def get_graident(self, x):
        g = KF.spatial_gradient(x).abs().sum(dim=2)
        return g

    def forward(self, x, y):
        couple = torch.cat([x, y], dim=1)
        x_g = self.get_graident(x)
        y_g = self.get_graident(y)
        couple_g = torch.cat([x_g, y_g], dim=1)
        c_a = self.channel_attention(couple_g)
        s_a = self.spatial_attention(couple_g)
        a = c_a * s_a
        a = self.sigmoid(a)
        couple_a = couple * a
        return couple_a


class FEB(nn.Module):
    def __init__(self,
                 bfe_c=[1, 4, 8],
                 num_blocks=4,
                 ):
        super(FEB, self).__init__()

        self.num_blocks = num_blocks
        self.in_channels = bfe_c[0]
        self.out_channels = bfe_c[-1]

        layers = []
        in_channels = bfe_c[0]
        for i in range(len(bfe_c) - 1):
            out_channels = bfe_c[i + 1]
            layers.extend(self.make_block(in_channels, out_channels, kernel_size=3, stride=1))
            in_channels = out_channels
        self.bfe = nn.Sequential(*layers)

        for i in range(num_blocks):
            setattr(self, f"e{i}", CB(in_ch=int(bfe_c[-1]), growth_rate=int(bfe_c[-1])))

    def make_block(self, in_channels, out_channels, kernel_size, stride):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                      bias=True,
                      padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        ]

    def forward_obo(self, x, block_index):
        assert block_index <= self.num_blocks - 1 and block_index >= 0
        if block_index == 0:
            assert x.size()[1] == self.in_channels
            x = self.bfe(x)
        else:
            assert x.size()[1] == self.out_channels
        block = getattr(self, f"e{block_index}")
        x = block(x)
        return x

    def forward(self, x):
        x = self.bfe(x)
        for i in range(self.num_blocks):
            block = getattr(self, f"e{i}")
            x = block(x)
        return x


class FRB2(nn.Module):
    def __init__(self,
                 res_channels = [8, 4, 1],
                 num_blocks=4,
                 in_C = 16
                  ):
        super(FRB2, self).__init__()
        self.num_blocks = num_blocks
        assert int(in_C / 2) == res_channels[0]

        self.f_conv = nn.Sequential(
                nn.Conv2d(in_C, int(in_C / 2), kernel_size=3, stride=1, padding=1, bias=True,
                      padding_mode="reflect"),
                nn.BatchNorm2d(int(in_C / 2)),
                nn.LeakyReLU(0.2)
        )

        layers = []
        self.in_channels = res_channels[0]
        in_channels = res_channels[0]
        for i in range(len(res_channels) - 1):
            out_channels = res_channels[i + 1]
            layers.extend(self.make_block(in_channels, out_channels, kernel_size=3, stride=1))
            in_channels = out_channels
        self.res = nn.Sequential(*layers)

        for i in range(num_blocks):
            setattr(self, f"d{i}", CB(in_ch=int(in_C / 2), growth_rate=int(in_C / 2)))

    def make_block(self, in_channels, out_channels, kernel_size, stride):
        if out_channels == 1:
            return[
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=True,
                      padding_mode="reflect"),
                nn.Tanh()
            ]
        else:
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=True,
                      padding_mode="reflect"),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ]

    def forward_obo(self, x, block_index):
        assert block_index <= self.num_blocks - 1 and block_index >= 0
        if block_index == 0:
            x = self.f_conv(x)
        block = getattr(self, f"d{block_index}")
        x = block(x)
        if block_index == (self.num_blocks - 1):
            x = self.res(x)
            x = (x + 1) / 2
        return x

    def forward(self, x):
        x = self.f_conv(x)
        for i in range(self.num_blocks):
            block = getattr(self, f"d{i}")
            x = block(x)
        x = self.res(x)
        x = (x + 1) / 2
        return x

class VFN(nn.Module):
    def __init__(self, num_blocks=4, c_list=None):
        super(VFN, self).__init__()
        if c_list is None:
            c_list = [1, 4, 8]
        self.vi_feb = FEB(num_blocks=num_blocks, bfe_c=c_list)
        self.ir_feb = FEB(num_blocks=num_blocks, bfe_c=c_list)
        self.frb1 = FRB1(inc=int(c_list[-1] * 2))
        self.frb2 = FRB2(num_blocks=num_blocks, in_C=int(c_list[-1] * 2), res_channels = c_list[::-1])

    def fusion(self, ir_f, vi_f):
        f = self.frb1(ir_f, vi_f)
        return f

    def forward_ER(self, ir, vi):
        assert ir.size() == vi.size()
        ir_f = self.ir_feb(ir)
        vi_f = self.vi_feb(vi)
        fusion = self.frb1(ir_f, vi_f)
        f = self.frb2(fusion)
        return f