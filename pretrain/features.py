import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import math
from matplotlib import pyplot as plt
import kornia
from kornia.filters.median import MedianBlur
from kornia.filters.gaussian import GaussianBlur2d
import torch


def channel_avg_pool(x, groups=9):
    B, C, H, W = x.shape
    return x.view(B, groups, -1, H, W).mean(dim=2)


class CannyTarget(torch.nn.Module):
    def __init__(
            self,
            low_threshold=0.1,
            high_threshold=0.2,
            kernel_size=(5, 5),
            sigma=(1.0, 1.0),
    ):
        super().__init__()

        self.canny = kornia.filters.Canny(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            kernel_size=kernel_size,
            sigma=sigma,
            hysteresis=True,
        )

    def forward(self, x):
        """
        x: [B, 3, H, W], assumed to be in [0, 1]
        return: [B, 1, H, W] Canny edge map
        """

        # RGB -> luminance
        if x.shape[1] == 3:
            x = kornia.color.rgb_to_grayscale(x)

        # magnitude: continuous gradient response
        # edges: thresholded Canny edge map
        magnitude, edges = self.canny(x)

        return edges


## SARATR-X Multi-scale grad feature
class GF(nn.Module):
    def __init__(self, kensize=7):
        super(GF, self).__init__()
        self.k = kensize

        def creat_gauss_kernel(r=1):
            M_13 = np.concatenate([np.ones([r + 1, 2 * r + 1]), np.zeros([r, 2 * r + 1])], axis=0)
            M_23 = np.concatenate([np.zeros([r, 2 * r + 1]), np.ones([r + 1, 2 * r + 1])], axis=0)

            M_11 = np.concatenate([np.ones([2 * r + 1, r + 1]), np.zeros([2 * r + 1, r])], axis=1)
            M_21 = np.concatenate([np.zeros([2 * r + 1, r]), np.ones([2 * r + 1, r + 1])], axis=1)
            return torch.from_numpy((M_13)).float(), torch.from_numpy((M_23)).float(), torch.from_numpy(
                (M_11)).float(), torch.from_numpy((M_21)).float()

        M13, M23, M11, M21 = creat_gauss_kernel(self.k)
        weight_x1 = M11.view(1, 1, self.k * 2 + 1, self.k * 2 + 1)
        weight_x2 = M21.view(1, 1, self.k * 2 + 1, self.k * 2 + 1)
        weight_y1 = M13.view(1, 1, self.k * 2 + 1, self.k * 2 + 1)
        weight_y2 = M23.view(1, 1, self.k * 2 + 1, self.k * 2 + 1)
        self.register_buffer("weight_x1", weight_x1)
        self.register_buffer("weight_x2", weight_x2)
        self.register_buffer("weight_y1", weight_y1)
        self.register_buffer("weight_y2", weight_y2)

    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(self.k, self.k, self.k, self.k), mode="reflect") + 1e-2
        gx_1 = F.conv2d(x, self.weight_x1, bias=None, stride=1, padding=0, groups=1)
        gx_2 = F.conv2d(x, self.weight_x2, bias=None, stride=1, padding=0, groups=1)
        gy_1 = F.conv2d(x, self.weight_y1, bias=None, stride=1, padding=0, groups=1)
        gy_2 = F.conv2d(x, self.weight_y2, bias=None, stride=1, padding=0, groups=1)
        gx_rgb = torch.log((gx_1) / (gx_2))
        gy_rgb = torch.log((gy_1) / (gy_2))
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        return norm_rgb


class MultiscaleGF(nn.Module):
    def __init__(self):
        super(MultiscaleGF, self).__init__()
        self.filter1 = GF(kensize=9)
        self.filter2 = GF(kensize=13)
        self.filter3 = GF(kensize=17)

    def forward(self, x):
        y1 = self.filter1(x)
        y2 = self.filter2(x)
        y3 = self.filter3(x)
        y = torch.cat([y1, y2, y3], dim=1)
        return y


class HOGLayerC(nn.Module):
    """Generate hog feature for each batch images. This module is used in
    Maskfeat to generate hog feature. This code is borrowed from.
    <https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/operators.py>
    Args:
        nbins (int): Number of bin. Defaults to 9.
        pool (float): Number of cell. Defaults to 8.
        gaussian_window (int): Size of gaussian kernel. Defaults to 16.
    """

    def __init__(self,
                 nbins: int = 9,
                 pool: int = 8,
                 gaussian_window: int = 16,
                 norm_out: bool = False,
                 in_channels: int = 1) -> None:
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        self.in_channels = in_channels
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = self.get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer('gkern', gkern)
        self.norm_out = norm_out

    def get_gkern(self, kernlen: int, std: int) -> torch.Tensor:
        """Returns a 2D Gaussian kernel array."""

        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n ** 2)
            return w

        gkern1d = _gaussian_fn(kernlen, std)
        gkern2d = gkern1d[:, None] * gkern1d[None, :]
        return gkern2d / gkern2d.sum()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate hog feature for each batch images.
        Args:
            x (torch.Tensor): Input images of shape (N, 3, H, W).
        Returns:
            torch.Tensor: Hog features.
        """
        # input is RGB image with shape [B 3 H W]
        hw = x.shape[-2], x.shape[-1]
        x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=self.in_channels)
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=self.in_channels)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w),
                          dtype=x.dtype,
                          device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, 'h {} gw {}'.format(
                    h, self.gaussian_window)
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        # out = out.sum(dim=2, keepdim=True)  # Sum across bins to get a single channel

        if self.norm_out:
            out = F.normalize(out, p=2, dim=2)
        if out.shape[1] == 1:  # single channel for SAR images
            out = out.squeeze(1)

        out = nn.functional.interpolate(out, hw, mode='bilinear')

        return out


class DualFeat(nn.Module):
    def __init__(self, filter='pixel', reduction='none', input_size=224, patch_size=4):
        super(DualFeat, self).__init__()
        self.filter_type = filter
        self.input_size = input_size
        if filter == 'mgf':
            self.filter = MultiscaleGF()
        elif filter == 'hog':
            self.filter = HOGLayerC()
        elif filter == 'canny':
            self.filter = CannyTarget()
        elif filter == 'pixel':
            self.filter = nn.Identity()
        self.filter.cuda()

        if reduction == 'median':
            self.median = MedianBlur(kernel_size=(5, 5))
        elif reduction in ['gaussian', 'dual']:
            self.gaussian = GaussianBlur2d(kernel_size=(7, 7), sigma=(1.5, 1.5))

        assert reduction in ['none', 'channel', 'spatial', 'median', 'dual', 'gaussian']

        self.reduction = reduction
        self.patch_size = patch_size

    @torch.no_grad()
    def forward(self, x):

        if self.filter_type == 'pixel':
            x_rgb = x['normalized'][:, :3, :, :].cuda(non_blocking=True)
            x_sar = x['normalized'][:, 3:, :, :].cuda(non_blocking=True)
            if self.reduction == 'none':
                return x_rgb, x_sar
            if self.reduction == 'channel' or self.reduction == 'dual':
                x_rgb = 0.299 * x_rgb[:, 0:1, :, :] + 0.587 * x_rgb[:, 1:2, :, :] + 0.114 * x_rgb[:, 2:3, :, :]
                x_rgb = x_rgb
                x_sar = x_sar.mean(dim=1, keepdim=True)
            if self.reduction == 'median':
                x_rgb = self.median(x_rgb)
                x_sar = self.median(x_sar)
            elif self.reduction == 'gaussian' or self.reduction == 'dual':
                x_rgb = self.gaussian(x_rgb)
                x_sar = self.gaussian(x_sar)
            return x_rgb, x_sar

        x_rgb = x['original'][:, :3, :, :].cuda(non_blocking=True)
        x_sar = x['original'][:, 3:, :, :].cuda(non_blocking=True)

        x_rgb = 0.299 * x_rgb[:, 0:1, :, :] + 0.587 * x_rgb[:, 1:2, :, :] + 0.114 * x_rgb[:, 2:3, :, :]
        x_sar = x_sar.mean(dim=1, keepdim=True)

        ft_rgb = self.filter(x_rgb)
        ft_sar = self.filter(x_sar)
        if self.reduction == 'channel' or self.reduction == 'dual':
            ft_rgb = ft_rgb.mean(dim=1, keepdim=True)
            ft_sar = ft_sar.mean(dim=1, keepdim=True)

        return ft_rgb, ft_sar

