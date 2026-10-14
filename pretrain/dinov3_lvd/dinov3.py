import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Callable, Optional
from functools import partial


class DINOv3ViTEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1024))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1024))
        self.register_tokens = nn.Parameter(torch.empty(1, 4, 1024))
        self.patch_embeddings = nn.Conv2d(
            3, 1024, kernel_size=16, stride=16)

    def forward(self, pixel_values, bool_masked_pos):
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.weight.dtype

        # (batch_size, num_channels, height, width) -> (batch_size, num_patches, hidden_size)
        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        if bool_masked_pos is not None:
            mask_token = self.mask_token.to(patch_embeddings.dtype)
            patch_embeddings = torch.where(bool_masked_pos.unsqueeze(-1), mask_token, patch_embeddings)

        # Add CLS and register tokens
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)

        return embeddings


def get_patches_center_coordinates(num_patches_h, num_patches_w, dtype, device):
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    # (height, width, 2) -> (height * width, 2)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    # Shift range [0, 1] to [-1, +1]
    coords = 2.0 * coords - 1.0
    return coords


def augment_patches_center_coordinates(coords, shift=None, jitter=None, rescale=None, ):
    # Shift coords by adding a uniform value in [-shift, shift]
    if shift is not None:
        shift_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
        shift_hw = shift_hw.uniform_(-shift, shift)
        coords = coords + shift_hw

    # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
    if jitter is not None:
        jitter_range = np.log(jitter)
        jitter_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
        jitter_hw = jitter_hw.uniform_(-jitter_range, jitter_range).exp()
        coords = coords * jitter_hw

    # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
    if rescale is not None:
        rescale_range = np.log(rescale)
        rescale_hw = torch.empty(1, device=coords.device, dtype=coords.dtype)
        rescale_hw = rescale_hw.uniform_(-rescale_range, rescale_range).exp()
        coords = coords * rescale_hw

    return coords


class DINOv3ViTRopePositionEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, ):
        super().__init__()
        self.base = 100.0
        self.head_dim = 1024 // 16
        self.num_patches_h = 224 // 16
        self.num_patches_w = 224 // 16

        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)  # (head_dim / 4,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape
        num_patches_h = height // 16
        num_patches_w = width // 16

        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            patch_coords = get_patches_center_coordinates(
                num_patches_h, num_patches_w, dtype=torch.float32, device=device
            )
            if self.training:
                patch_coords = augment_patches_center_coordinates(
                    patch_coords,
                    shift=None,
                    jitter=None,
                    rescale=2.0,
                )

            # (height * width, 2, head_dim / 4) -> (height * width, head_dim / 2) -> (height * width, head_dim)
            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
        q, k, cos, sin, **kwargs
):
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches  # cls token + register tokens

    q_prefix_tokens, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix_tokens, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    # apply rope only to patch tokens
    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    k = torch.cat((k_prefix_tokens, k_patches), dim=-2)

    return q, k


def eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=None,
        is_causal=None,
        **kwargs,
):
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


class DINOv3ViTAttention(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = False

        self.scaling = self.head_dim ** -0.5

        self.dropout = 0.0
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_embeddings=None,
            **kwargs,
    ):
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # attention_interface = eager_attention_forward
        attention_interface = sdpa_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GradientCheckpointingLayer(nn.Module):
    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            do_warn = False
            layer_name = self.__class__.__name__
            message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

            if "use_cache" in kwargs and kwargs["use_cache"]:
                kwargs["use_cache"] = False
                message += " `use_cache=False`,"
                do_warn = True

            if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                kwargs["past_key_value"] = None
                message += " `past_key_value=None`,"
                do_warn = True

            if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                kwargs["past_key_values"] = None
                message += " `past_key_values=None`,"
                do_warn = True

            if "layer_past" in kwargs and kwargs["layer_past"] is not None:
                kwargs["layer_past"] = None
                message += " `layer_past=None`,"
                do_warn = True

            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)


class Dinov2LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = nn.Parameter(1.0 * torch.ones(1024))

    def forward(self, hidden_state):
        return hidden_state * self.lambda1


class DINOv3ViTLayerScale(Dinov2LayerScale):
    pass


def drop_path(input, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class Dinov2DropPath(nn.Module):
    def __init__(self, drop_prob=None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class DINOv3ViTDropPath(Dinov2DropPath):
    pass


class GELUActivation(nn.Module):
    def __init__(self, use_gelu_python=False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)


class ArceeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 1024
        self.intermediate_size = 4096
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = GELUActivation()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DINOv3ViTMLP(ArceeMLP):
    pass


class DINOv3ViTLayer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, ):
        super().__init__()

        self.norm1 = nn.LayerNorm(1024, eps=1e-05)
        self.attention = DINOv3ViTAttention()
        self.layer_scale1 = DINOv3ViTLayerScale()
        self.drop_path = nn.Identity()

        self.norm2 = nn.LayerNorm(1024, eps=1e-05)

        self.mlp = DINOv3ViTMLP()
        self.layer_scale2 = DINOv3ViTLayerScale()

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_embeddings=None,
    ):
        # Attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


class DINOV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = DINOv3ViTEmbeddings()
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding()
        self.layer = nn.ModuleList([DINOv3ViTLayer() for _ in range(24)])
        self.norm = nn.LayerNorm(1024, eps=1e-05)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()

    def init_weights(self):
        from safetensors.torch import load_file, save_file
        state_dict = load_file('./dinov3_lvd/model.safetensors')
        self.load_state_dict(state_dict, strict=True)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        print('dinov3_lvd initilized')

    @torch.no_grad()
    def forward(
            self,
            pixel_values,
            bool_masked_pos=None,
            head_mask=None,
            **kwargs,
    ):

        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        position_embeddings = self.rope_embeddings(pixel_values)

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = None
            hidden_states = layer_module(
                hidden_states,
                attention_mask=layer_head_mask,
                position_embeddings=position_embeddings,
            )

        sequence_output = self.norm(hidden_states)

        return sequence_output


if __name__ == "__main__":
    model = DINOV3()

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    state_dict = load_file('./model.safetensors')

    print()