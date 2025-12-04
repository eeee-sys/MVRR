
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import Parameter

# Permute是用来把三维张量（B，L，D）中最后两个维度对调变成（B，D，L）
class Permute(nn.Module):

    def forward(self, x):
        return x.transpose(-1, -2)

# LearnableEmbedding是可学习的嵌入向量，可以对应论文grounder中的e_p，m_v，m_r
class LearnableEmbedding(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.weights = Parameter(1, 1, dims)

    def forward(self, x):
        return x + self.weights.expand_as(x)


# ConvPyramid就是论文的Grounder中的Temporal Pyramid
class ConvPyramid(nn.Module):

    def __init__(self, dims, strides, act_cls=nn.ReLU):
        super().__init__()

        self.blocks = nn.ModuleList()
        for s in strides:
            p = int(math.log2(s))
            if p == 0:
                layers = act_cls()
            else:
                conv_cls = nn.Conv1d if p > 0 else nn.ConvTranspose1d
                layers = nn.Sequential()
                for _ in range(abs(p)):
                    module = [Permute(), conv_cls(dims, dims, 2, stride=2), Permute(), nn.LayerNorm(dims), act_cls()]
                    layers.extend(module)
            self.blocks.append(layers)

        self.strides = strides

    def forward(self, x, mask, return_mask=False):
        pymid, pymid_msk = [], []

        for s, blk in zip(self.strides, self.blocks):
            if x.size(1) < s:
                continue

            pymid.append(blk(x))

            if return_mask:
                if s > 1:
                    msk = F.max_pool1d(mask.float(), s, stride=s).long()
                elif s < 1:
                    msk = mask.repeat_interleave(int(1 / s), dim=1)
                else:
                    msk = mask
                pymid_msk.append(msk)

        return (pymid, pymid_msk) if return_mask else pymid

"""
Scale模块是配合着ConvPyramid使用的，ConvPyramid得到不同时间
尺度的特征，比如T/2，T/4，T/8，Scale模块就是给不同的时间尺度加一个
可学习的缩放因子，说明哪个时间尺度更加重要
"""
class Scale(nn.Module):

    def __init__(self, strides):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(len(strides)))

    def forward(self, x, i):
        return x * self.scale[i]

# ConvHead就是一维卷积
class ConvHead(nn.Module):

    def __init__(self, dims, out_dims, kernal_size=3, act_cls=nn.ReLU):
        super().__init__()

        # yapf:disable
        self.module = nn.Sequential(
            Permute(),
            nn.Conv1d(dims, dims, kernal_size, padding=kernal_size // 2),
            act_cls(),
            nn.Conv1d(dims, out_dims, kernal_size, padding=kernal_size // 2),
            Permute())
        # yapf:enable

    def forward(self, x):
        return self.module(x)
