import torch.nn as nn
import torch
from dinov3_lvd.dinov3 import DINOV3


class dino_distill(nn.Module):
    def __init__(self, model_name="./dinov3_lvd", **kwargs):
        super().__init__()
        self.scaling_layer = ScalingLayerForDINOv3()
        self.teacher_model = DINOV3()
        self.LN = nn.LayerNorm(1024, elementwise_affine=False)

    @torch.no_grad()
    def get_target(self, x):
        norm_imgs = self.scaling_layer(x)
        target = self.teacher_model(norm_imgs)
        target = self.LN(target)
        patch_tokens = target[:, 5:, :]
        return patch_tokens

    def forward(self, x, **kwargs):
        """
        x: shape [B, 3, H, W] in [0, 1]
        """
        target = self.get_target(x, **kwargs)
        return target


class ScalingLayerForDINOv3(nn.Module):
    def __init__(self):
        super(ScalingLayerForDINOv3, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, inp):
        out = (inp - self.shift) / self.scale
        return out
