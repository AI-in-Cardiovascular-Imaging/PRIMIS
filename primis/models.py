import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2, comb, prod


class IndexAutoEncoder(nn.Module):
    """
    An image autoencoder with Sparse Ternary Codes.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.rate_actual = None
        self.rate_ideal = None
        self.bpp_actual = None
        self.bpp_ideal = None

    def encode(self, inp):
        code = self.encoder(inp)
        return code

    def decode(self, code, quantize=False):
        out = self.decoder(code, quantize=quantize)
        return out

    def set_k(self, k):
        self.encoder.set_k(k)
        self.decoder.set_k(k)

    def forward(self, inp, quantize=False):
        code = self.encode(inp)
        out = self.decode(code, quantize=quantize)

        self.rate_actual = self.encoder.rate_actual
        self.rate_ideal = self.encoder.rate_ideal
        self.bpp_actual = self.rate_actual / out[0, 0].shape.numel()
        self.bpp_ideal = self.rate_ideal / out[0, 0].shape.numel()

        return out, code


class DownConvBlockInternal(nn.Module):

    def __init__(self, c_i, c_o, r):
        super(DownConvBlockInternal, self).__init__()

        self.c_i = c_i
        self.c_o = c_o
        self.r = r

        self.res_conv = nn.Conv2d(self.c_i, self.c_o, 1, padding=0, stride=self.r)
        self.seq = nn.Sequential(
            nn.Conv2d(self.c_i, self.c_o, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.c_o, self.c_o, kernel_size=5, padding=2, stride=self.r, groups=1),
            nn.BatchNorm2d(self.c_o),
        )

    def forward(self, inp):
        res = self.res_conv(inp)
        inp = self.seq(inp)
        inp = inp + res
        inp = inp.relu_()
        inp = F.leaky_relu(inp)

        return inp


class DownConvBlockExternal(nn.Module):

    def __init__(self, c_i, c_o, r):
        super(DownConvBlockExternal, self).__init__()

        self.c_i = c_i
        self.c_o = c_o
        self.r = r

        self.res_conv = nn.Conv2d(self.c_i, self.c_o, kernel_size=1, padding=0, stride=self.r)
        self.seq = nn.Sequential(
            nn.Conv2d(self.c_i, self.c_i, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.c_i, self.c_i, kernel_size=5, padding=2, stride=1),
            nn.Conv2d(self.c_i, self.c_o, kernel_size=1, padding=0, stride=self.r),
            nn.BatchNorm2d(self.c_o)
        )

    def forward(self, inp):
        res = self.res_conv(inp)
        inp = self.seq(inp)
        inp = inp + res
        inp = F.leaky_relu(inp)

        return inp


class UpConvBlockInternal(nn.Module):

    def __init__(self, c_i, c_o, r):
        super(UpConvBlockInternal, self).__init__()

        self.c_i = c_i
        self.c_o = c_o
        self.r = r

        self.up = nn.Upsample(scale_factor=self.r, mode='bilinear', align_corners=False)
        self.res_conv = nn.Conv2d(self.c_i, self.c_o, 1)
        self.seq = nn.Sequential(
            nn.Conv2d(self.c_i, self.c_o, 5, padding=2, stride=1),
            # nn.GroupNorm(num_groups=1, num_channels=self.c_o),  # Does it help?
            nn.LeakyReLU(),
            nn.Conv2d(self.c_o, self.c_o, 3, padding=1, stride=1),
        )

    def forward(self, inp):
        inp = self.up(inp)
        res = self.res_conv(inp)
        inp = self.seq(inp)
        inp = inp + res

        return inp


class UpConvBlockExternal(nn.Module):

    def __init__(self, c_i, c_o, r):
        super(UpConvBlockExternal, self).__init__()

        self.c_i = c_i
        self.c_o = c_o
        self.r = r

        self.up = nn.Upsample(scale_factor=self.r, mode='bilinear', align_corners=False)
        self.res_conv = nn.Conv2d(self.c_i, self.c_o, 1)
        self.seq = nn.Sequential(
            nn.Conv2d(self.c_i, self.c_i, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.c_i, self.c_o, 1, padding=0, stride=1),
        )

    def forward(self, inp):
        inp = self.up(inp)
        res = self.res_conv(inp)
        inp = self.seq(inp)
        inp = inp + res

        return inp


class ConvIndexer(nn.Module):

    def __init__(self, c, f, k, m, dropout):
        super(ConvIndexer, self).__init__()
        self.c = c
        self.f = f
        self.k = k
        self.m = m
        self.groups = self.c
        self.stride = self.f // 4

        self.dropout = nn.Dropout2d(dropout)

        self.proj = nn.Conv2d(self.c * self.m, self.c * self.m, kernel_size=self.f, stride=self.stride,
                              groups=self.groups)

    def set_k(self, k):
        self.k = k

    def forward(self, inp):
        inp = self.proj(inp)
        inp = self.dropout(inp)

        _b, _c, _w, _h = inp.shape

        inp = inp.view(_b, self.c, self.m, _w, _h)
        val_pos, ind_pos = torch.topk(inp, self.k, dim=1)
        val_neg, ind_neg = torch.topk(-inp, self.k, dim=1)

        code = {'pos': {'ind': ind_pos, 'val': val_pos},
                'neg': {'ind': ind_neg, 'val': -val_neg}}

        return code


class ConvDeIndexer(nn.Module):

    def __init__(self, indexer):
        super(ConvDeIndexer, self).__init__()

        self.c = indexer.c
        self.f = indexer.f
        self.k = indexer.k
        self.m = indexer.m
        self.groups = indexer.groups
        self.stride = indexer.stride

        self.weight = indexer.proj.weight
        self.beta_pos = nn.Parameter(torch.ones(self.m * self.k)).requires_grad_()
        self.beta_neg = nn.Parameter(torch.ones(self.m * self.k)).requires_grad_()

    def set_k(self, k):
        self.k = k

    def forward(self, code, quantize=False):
        device = code['pos']['ind'].device
        _b, _k, _m, _w, _h = code['pos']['ind'].shape

        ind_pos = code['pos']['ind'] + \
                  self.c * torch.arange(_m).reshape(1, 1, _m, 1, 1).repeat(_b, _k, 1, _w, _h).to(device)

        ind_neg = code['neg']['ind'] + \
                  self.c * torch.arange(_m).reshape(1, 1, _m, 1, 1).repeat(_b, _k, 1, _w, _h).to(device)

        ind_pos = ind_pos.reshape(_b, _k * _m, _w, _h)
        ind_neg = ind_neg.reshape(_b, _k * _m, _w, _h)
        val_pos = code['pos']['val'].reshape(_b, _k * _m, _w, _h)
        val_neg = code['neg']['val'].reshape(_b, _k * _m, _w, _h)

        out = torch.zeros(_b, self.c * self.m, _w, _h, device=device)

        if not quantize:
            out.scatter_(1, ind_pos, val_pos)
            out.scatter_(1, ind_neg, val_neg)
            if self.training:
                with torch.no_grad():
                    self.beta_pos.data = val_pos.mean(dim=(0, 2, 3))
                    self.beta_neg.data = val_neg.mean(dim=(0, 2, 3))
                    # TODO: better a running mean here?
        else:
            out.scatter_(1, ind_pos, self.beta_pos.view(1, self.c * self.m, 1, 1).expand(_b, self.c * self.m, _w, _h))
            out.scatter_(1, ind_neg, self.beta_neg.view(1, self.c * self.m, 1, 1).expand(_b, self.c * self.m, _w, _h))

        out = F.conv_transpose2d(out, weight=self.weight, stride=self.stride, groups=self.groups)

        return out


class BankEncoder(nn.Module):

    def __init__(self, init_dict):
        super(BankEncoder, self).__init__()

        self.rate_actual = None
        self.rate_ideal = None
        self.bpp_actual = None
        self.bpp_ideal = None

        self.r_list = init_dict["sampling_ratio_list"]
        self.c_list = init_dict["channels_list"]
        self.c_o = init_dict["num_proj_channels"]
        self.f = init_dict["proj_filter_size"]
        self.k = init_dict["k"]
        self.m = init_dict["num_code_maps"]
        self.dropout = init_dict["dropout"]
        self.num_blocks = len(self.r_list)

        assert len(self.c_list) == 1 + len(self.r_list)

        self.down_blocks_int = nn.ModuleList()
        self.down_blocks_ext = nn.ModuleList()
        for i_b in range(self.num_blocks):
            self.down_blocks_int.append(
                DownConvBlockInternal(self.c_list[i_b], self.c_list[i_b + 1], self.r_list[i_b]))
            self.down_blocks_ext.append(
                DownConvBlockExternal(self.c_list[i_b], self.c_list[-1], prod(self.r_list[i_b:])))

        self.conv_channel = nn.Conv2d(self.c_list[-1] * (self.num_blocks + 1), self.c_o * self.m, kernel_size=1)
        self.conv_indexer = ConvIndexer(self.c_o, self.f, self.k, self.m, self.dropout)

        # TODO: I thought it would be good to lighten the number of projection weights. So added 1-1 convs.
        #  Check if it helps. Otherwise, get rid of it.

    def set_k(self, k):
        self.k = k
        self.conv_indexer.set_k(k)

    def forward(self, inp):

        b, c, h, w = inp.shape
        for i_b in range(self.num_blocks):
            if i_b == 0:
                res = self.down_blocks_ext[0](inp)
            else:
                res = torch.cat((res, self.down_blocks_ext[i_b](inp)), 1)
            inp = self.down_blocks_int[i_b](inp)

        res = torch.cat((res, inp), 1)
        res = self.conv_channel(res)
        code = self.conv_indexer(res)

        # # TODO: This is ugly.
        self.rate_actual = 2 * self.m * code['pos']['ind'][0, 0].shape.numel() * log2(self.c_o)
        self.rate_ideal = self.m * code['pos']['ind'][0, 0, 0].shape.numel() * log2(comb(self.c_o, 2 * self.k))
        self.bpp_actual = self.rate_actual / h * w
        self.bpp_ideal = self.rate_ideal / h * w

        return code


class BankDecoder(nn.Module):

    def __init__(self, encoder):
        super(BankDecoder, self).__init__()

        self.r_list = encoder.r_list[::-1]
        self.c_list = encoder.c_list[::-1]
        self.c_o = encoder.c_o
        self.m = encoder.m
        self.f = encoder.f
        self.num_blocks = len(self.r_list)

        self.conv_deindexer = ConvDeIndexer(encoder.conv_indexer)
        self.conv_channel = nn.Conv2d(self.c_o * self.m, self.c_list[0] * (self.num_blocks + 1), kernel_size=1)

        self.k = self.conv_deindexer.k

        self.up_blocks_int = nn.ModuleList()
        self.up_blocks_ext = nn.ModuleList()
        for i_b in range(self.num_blocks):
            self.up_blocks_int.append(
                UpConvBlockInternal(self.c_list[i_b], self.c_list[i_b + 1], self.r_list[i_b]))
            self.up_blocks_ext.append(
                UpConvBlockExternal(self.c_list[0], self.c_list[i_b + 1], prod(self.r_list[0:i_b + 1])))

    def set_k(self, k):
        self.k = k
        self.conv_deindexer.set_k(k)

    def forward(self, code, quantize=False):

        out_init = self.conv_deindexer(code, quantize=quantize)
        out_init = self.conv_channel(out_init)

        out = out_init[:, 0:self.c_list[0], :, :]
        for i_b in range(self.num_blocks):
            out = self.up_blocks_int[i_b](out)
            res = out_init[:, self.c_list[0] * (i_b + 1): self.c_list[0] * (i_b + 2), :, :]
            res = self.up_blocks_ext[i_b](res)
            out = out + res

        return out
