import math
import os

import torch
import torch.nn as nn

import torch.nn.functional as F

from layers.ChebyKANLayer import ChebyKANLinear
from layers.Embed import PatchEmbed


'''
HD-KAN: 
Hierarchical Dual-Dimension Kolmogorov-Arnold Networks with Bidirectional Time-Frequency Fusion 
for Long-term Time Series Forecasting
'''


class DualKAN(nn.Module):
    """
    Dual-Dimension Kolmogorov-Arnold Network.

    Applies KAN to both feature dimension and spatial dimension,
    then fuses them via multiplicative interaction with residual connection.

    Args:
        d_model: feature dimension size
        channel: spatial/channel dimension size
        degree1: polynomial degree for feature dimension
        degree2: polynomial degree for spatial dimension
    """
    def __init__(self, d_model, channel, degree1, degree2, dropout=0.1):
        super().__init__()
        self.fc1 = ChebyKANLinear(
                            d_model,
                            d_model,
                            degree1,)
        self.fc2 = ChebyKANLinear(
                            channel,
                            channel,
                            degree2,)
        self.l1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, N = x.shape

        x1 = self.fc1(x.reshape(B*C,N)).reshape(B,C,-1)

        x2 = self.fc2(x.permute(0,2,1).reshape(B*N,C)).reshape(B,N,-1).permute(0,2,1)

        x = self.l1(self.dropout(x1*x2) + x)

        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.patch_len = configs.patch_len
        self.patch_num = self.seq_len // self.patch_len
        self.dim = self.d_model // self.patch_num

        # d_model
        self.degree1 = configs.degree1
        # channel
        self.degree2 = configs.degree2
        # patch_num
        self.degree4 = int(math.sqrt(self.patch_num)) + 1
        # dim
        self.degree3 = self.degree1 - self.degree4 + 2
        if self.degree3 < 1:
            self.degree3 = 1

        self.use_revin = configs.use_revin

        self.DualKAN = nn.ModuleList()
        # time
        self.DualKAN.append(DualKAN(self.dim,self.patch_num,self.degree3,self.degree4,self.dropout))
        self.DualKAN.append(DualKAN(self.d_model,self.enc_in,self.degree1,self.degree2,self.dropout))
        # freq
        self.DualKAN.append(DualKAN(self.dim+2, self.patch_num, self.degree3,self.degree4,self.dropout))
        self.DualKAN.append(DualKAN(self.d_model+2,self.enc_in,self.degree1,self.degree2,self.dropout))

        self.embedding = PatchEmbed(configs,self.dim, num_p=self.patch_num)

        self.l1 = nn.Linear(self.patch_num*self.dim, self.d_model)
        self.l2 = nn.Linear(self.patch_num*(self.dim+2), self.d_model+2)
        self.proj = nn.Linear(self.d_model*2, self.pred_len,bias=True)


    def forward(self, x, x_mark):

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # patch
        x = self.embedding(x, None)
        b,c,n,d = x.shape
        x = x.reshape(-1,n,d)

        # freq 1
        xf = torch.fft.rfft(x, dim=-1)  # [B, S, C], complex
        xf = torch.cat([xf.real, xf.imag], dim=-1)

        xff = self.DualKAN[2](xf)

        L1 = xff.shape[-1] // 2
        xf_ = torch.complex(xff[:,:,:L1], xff[:,:,L1:])
        xf_res = torch.fft.irfft(xf_, dim=-1)

        # time 1
        xt = self.DualKAN[0](x)


        # time 2
        xt2 = xt + xf_res
        xt2 = xt2.reshape(b, c, -1)
        xt2 = self.l1(xt2)

        xt2 = self.DualKAN[1](xt2)


        # freq 2
        xt_res = torch.fft.rfft(xt, dim=-1)
        xt_res = torch.cat([xt_res.real, xt_res.imag], dim=-1)
        xf2 = xff + xt_res

        L2 = xf2.shape[-1] // 2
        xf_rr = xf2[:,:,:L2].reshape(b, c, -1)
        xf_ii = xf2[:,:,L2:].reshape(b, c, -1)

        xf2 = torch.cat([xf_rr, xf_ii], dim=-1)
        xf2_ = self.l2(xf2)

        xf2 = self.DualKAN[3](xf2_)

        L3 = xf2.shape[-1] // 2

        xf2 = torch.complex(xf2[:,:,:L3], xf2[:,:,L3:])
        xf2 = torch.fft.irfft(xf2, dim=-1)


        # projection
        yt = torch.cat([xt2, xf2], dim=-1)
        y = self.proj(yt)
        y = y[:,:self.enc_in,:].permute(0, 2, 1)

        # denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y
