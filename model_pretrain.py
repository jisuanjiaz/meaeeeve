import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, LIFNode
from spikingjelly.clock_driven import functional
# from neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Block
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import math
import numpy as np
import einops
from knn_cuda import KNN
from functional import fps, get_sinusoid_encoding_table


class Projection(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=256):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Prediction(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=256):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SpikingTokenizer(nn.Module):
    def __init__(
            self, img_size=128, patch_size=4, in_channels=2, timestep=16, embed_dims=256, pool_state=[True, True, True, True],
            backend="cupy", lif_tau=1.5):
        super().__init__()
        self.image_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pool_state = pool_state
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.t_num_patches = timestep // 2

        self.proj_conv = nn.Conv3d(in_channels, embed_dims // 8, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm3d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1, ceil_mode=False)

        self.proj1_conv = nn.Conv3d(embed_dims // 8, embed_dims // 4, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm3d(embed_dims // 4)
        self.proj1_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1, ceil_mode=False)

        self.proj2_conv = nn.Conv3d(embed_dims // 4, embed_dims // 2, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm3d(embed_dims // 2)
        self.proj2_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), dilation=1, ceil_mode=False)

        self.proj3_conv = nn.Conv3d(embed_dims // 2, embed_dims, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm3d(embed_dims)
        self.proj3_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv3d(embed_dims, embed_dims, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.proj_conv(x)
        if self.pool_state[0]:
            x = self.maxpool(x)
        x = self.proj_bn(x).transpose(0, 2).contiguous()
        x = self.proj_lif(x).transpose(0, 2)

        x = self.proj1_conv(x)
        if self.pool_state[1]:
            x = self.maxpool1(x)
        x = self.proj1_bn(x).transpose(0, 2).contiguous()
        x = self.proj1_lif(x).transpose(0, 2)

        x = self.proj2_conv(x)
        if self.pool_state[2]:
            x = self.maxpool2(x)
        x = self.proj2_bn(x).transpose(0, 2).contiguous()
        x = self.proj2_lif(x).transpose(0, 2)

        x = self.proj3_conv(x)
        if self.pool_state[3]:
            x = self.maxpool3(x)
        x = self.proj3_bn(x).transpose(0, 2).contiguous()
        x = self.proj3_lif(x).transpose(0, 2)

        x = self.rpe_conv(x).flatten(3)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_channel, in_channels, point_dim, backend="cupy", lif_tau=1.5):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.in_channels = in_channels

        self.first_conv = nn.Sequential(
            nn.Conv1d(self.in_channels, point_dim, 1),
            nn.BatchNorm1d(point_dim),
            LIFNode(tau=lif_tau, detach_reset=True),
            nn.Conv1d(point_dim, point_dim*2, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(point_dim*4, point_dim*4, 1),
            nn.BatchNorm1d(point_dim*4),
            LIFNode(tau=lif_tau, detach_reset=True),
            nn.Conv1d(point_dim*4, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.in_channels).transpose(2, 1)
        feature = self.first_conv(point_groups)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        feature_global = feature_global.reshape(bs, g, self.encoder_channel)
        feature_global = feature_global.transpose(-1, -2)
        return feature_global


class Group(nn.Module):
    def __init__(self, num_group, group_size, in_channels):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.in_channels = in_channels
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)
        _, idx = self.knn(xyz, center)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, self.in_channels).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)

        self.attn_lif = LIFNode(tau=lif_tau, detach_reset=True)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn_q = nn.BatchNorm1d(dim)
        self.proj_bn_kv = nn.BatchNorm1d(dim)
        self.proj_lif_q = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)
        self.proj_lif_kv = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)

    def forward(self, query, kv):
        B, C_kv, T, N_kv = kv.shape
        _, C_q, _, N_q = query.shape

        query = self.proj_bn_q(query.flatten(2)).reshape(B, C_q, T, N_q).permute(2, 0, 1, 3).contiguous()
        query = self.proj_lif_q(query).permute(1, 2, 0, 3).flatten(2)

        x = self.proj_bn_kv(kv.flatten(2)).reshape(B, C_kv, T, N_kv).permute(2, 0, 1, 3).contiguous()
        x_for_qkv = self.proj_lif_kv(x).permute(1, 2, 0, 3).flatten(2)

        q_conv_out = self.q_conv(query)
        q_conv_out = self.q_bn(q_conv_out).reshape(B, C_q, T, N_q).permute(2, 0, 1, 3).contiguous()
        q_conv_out = self.q_lif(q_conv_out).permute(1, 0, 3, 2).flatten(1, 2)
        q = q_conv_out.reshape(B, T*N_q, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(B, C_kv, T, N_kv).permute(2, 0, 1, 3).contiguous()
        k_conv_out = self.k_lif(k_conv_out).permute(1, 0, 3, 2).flatten(1, 2)
        k = k_conv_out.reshape(B, T*N_kv, self.num_heads, C_kv // self.num_heads).permute(0, 2, 1, 3)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(B, C_kv, T, N_kv).permute(2, 0, 1, 3).contiguous()
        v_conv_out = self.v_lif(v_conv_out).permute(1, 0, 3, 2).flatten(1, 2)
        v = v_conv_out.reshape(B, T*N_kv, self.num_heads, C_kv // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale
        x = x.transpose(2, 3).reshape(B, C_q, T * N_q).contiguous()
        x = self.attn_lif(x)
        x = self.proj_conv(x).reshape(B, C_q, T, N_q)
        return x


class SpikingSelfAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)

        self.attn_lif = LIFNode(tau=lif_tau, detach_reset=True)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)

    def forward(self, x):
        B, C, T, N = x.shape

        x = self.proj_bn(x.flatten(2)).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()
        x_for_qkv = self.proj_lif(x).permute(1, 2, 0, 3).flatten(2)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()
        q_conv_out = self.q_lif(q_conv_out).permute(1, 0, 3, 2).flatten(1, 2)
        q = q_conv_out.reshape(B, T*N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()
        k_conv_out = self.k_lif(k_conv_out).permute(1, 0, 3, 2).flatten(1, 2)
        k = k_conv_out.reshape(B, T*N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()
        v_conv_out = self.v_lif(v_conv_out).permute(1, 0, 3, 2).flatten(1, 2)
        v = v_conv_out.reshape(B, T*N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale
        x = x.transpose(2, 3).reshape(B, C, T * N).contiguous()
        x = self.attn_lif(x)
        x = self.proj_conv(x).reshape(B, C, T, N)
        return x


class SpikingMLP(nn.Module):
    def __init__(
            self,
            in_features, hidden_features=None, out_features=None,
            backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B, C, T, N = x.shape

        x = self.fc2_bn(x.flatten(2)).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()
        x = self.fc2_lif(x).permute(1, 2, 0, 3).flatten(2)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).reshape(B, self.c_hidden, T, N).permute(2, 0, 1, 3).contiguous()
        x = self.fc1_lif(x).permute(1, 2, 0, 3).flatten(2)
        x = self.fc2_conv(x).reshape(B, C, T, N)

        return x


class SpikingTransformerBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4.,
            cross_attention=False, drop_path=0.0,
            backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_attn = CrossAttention(
                dim, num_heads=num_heads,
                backend=backend, lif_tau=lif_tau,
            )
        self.attn = SpikingSelfAttention(
            dim, num_heads=num_heads,
            backend=backend, lif_tau=lif_tau,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpikingMLP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            backend=backend, lif_tau=lif_tau,
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x, point=None):
        x = x + self.drop_path(self.attn(x))
        if self.cross_attention and point is not None:
            x = x + self.drop_path(self.cross_attn(query=x, kv=point))
        x = x + self.drop_path(self.mlp(x))
        return x


class MAE(nn.Module):
    def __init__(
            self,
            img_size=128, patch_size=16, image_channels=2, pool_state=[True, True, True, True],
            in_channels=2, mlp_ratios=4, image_timestep=4, n_win=16,
            num_group=64, group_size=32, point_dim=128,
            local_embed_dims=256, local_num_heads=4, local_image_depths=6, local_point_depths=6,
            global_embed_dims=384, global_num_heads=4, global_depths=6,
            decoder_embed_dims=256, decoder_depths=2, decoder_num_heads=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            image_mask_ratio=0.5, point_mask_ratio=0.5, merged_number_windows=1,
            num_classes=11, TET=True, drop_path=0.0, backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.local_embed_dims = local_embed_dims
        self.image_timestep = image_timestep
        self.n_win = n_win
        self.TET = TET
        self.num_group = num_group
        self.group_size = group_size
        self.image_mask_ratio = image_mask_ratio
        self.point_mask_ratio = point_mask_ratio
        self.merged_T = merged_number_windows

        self.group_divider_s = Group(num_group=self.num_group * 2, group_size=self.group_size // 2, in_channels=in_channels)
        self.group_divider_l = Group(num_group=self.num_group // 2, group_size=self.group_size * 2, in_channels=in_channels)
        self.group_divider_m = Group(num_group=self.num_group, group_size=self.group_size, in_channels=in_channels)
        self.point_encoder = Encoder(encoder_channel=local_embed_dims, in_channels=in_channels, point_dim=point_dim, backend=backend, lif_tau=lif_tau,)
        self.point_pos_embed = nn.Sequential(
            nn.Conv1d(in_channels, point_dim, 1),
            nn.BatchNorm1d(point_dim),
            LIFNode(tau=lif_tau, detach_reset=True),
            nn.Conv1d(point_dim, local_embed_dims, 1),
        )
        self.scale_s_pos_embed = nn.Parameter(torch.zeros(1, global_embed_dims, 1))
        self.scale_m_pos_embed = nn.Parameter(torch.zeros(1, global_embed_dims, 1))
        self.scale_l_pos_embed = nn.Parameter(torch.zeros(1, global_embed_dims, 1))

        self.patch_embed = SpikingTokenizer(
            img_size=img_size, patch_size=patch_size, in_channels=image_channels, embed_dims=local_embed_dims,
            timestep=image_timestep,
            pool_state=pool_state, backend=backend, lif_tau=lif_tau,
        )
        self.num_patches = self.patch_embed.num_patches
        self.t_num_patches = self.patch_embed.t_num_patches
        self.image_pos_embed = get_sinusoid_encoding_table(self.num_patches * self.t_num_patches, local_embed_dims)

        local_depths = max(local_point_depths, local_image_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path, global_depths + local_depths)]
        self.local_image_blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=local_embed_dims, num_heads=local_num_heads, mlp_ratio=mlp_ratios, drop_path=dpr[j],
                backend=backend, lif_tau=lif_tau,
            )
            for j in range(local_image_depths)
        ])
        self.local_point_blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=local_embed_dims, num_heads=local_num_heads, mlp_ratio=mlp_ratios, drop_path=dpr[j],
                backend=backend, lif_tau=lif_tau,
            )
            for j in range(local_point_depths)
        ])
        self.projection = Projection(in_dim=local_embed_dims)
        self.predictor = Prediction()
        self.global_blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=global_embed_dims, num_heads=global_num_heads, mlp_ratio=mlp_ratios, drop_path=dpr[j + local_depths],
                cross_attention=True,
                backend=backend, lif_tau=lif_tau,
            )
            for j in range(global_depths)
        ])
        self.global_bn = nn.BatchNorm1d(global_embed_dims)
        self.decoder_embed = nn.Linear(global_embed_dims, decoder_embed_dims, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dims))
        self.decoder_pos_embed = get_sinusoid_encoding_table(self.num_patches * self.n_win // 2, decoder_embed_dims)
        self.vit_blocks = nn.ModuleList([
            Block(decoder_embed_dims, decoder_num_heads, mlp_ratios, qkv_bias=True, norm_layer=norm_layer)
            for j in range(decoder_depths)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dims)
        self.decoder_pred = nn.Linear(decoder_embed_dims, 2 * patch_size ** 2 * image_channels, bias=True)

        self.norm_pix_loss = True
        self.mse_loss = nn.MSELoss()
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        trunc_normal_(self.scale_s_pos_embed, std=.02)
        trunc_normal_(self.scale_m_pos_embed, std=.02)
        trunc_normal_(self.scale_l_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def tube_masking(self, x, mask_ratio, density_rev):
        T, B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device) + density_rev.view(B, -1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(0).unsqueeze(-1).repeat(T, 1, 1, D))
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        B, C, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = 2
        h = w = H // p
        t = T // u
        x = imgs.reshape(B, C, t, u, h, p, w, p)
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(B, t * h * w, u * p ** 2 * C)
        return x

    def forward_rec_loss(self, pred, target, mask):
        target = self.patchify(target)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_teacher_encoder(self, imgs):
        x = self.patch_embed(imgs)
        B, D, T, N = x.shape
        pos_embed = self.image_pos_embed.transpose(1, 2).reshape(1, self.local_embed_dims, T, self.num_patches).type_as(x).to(x.device).clone().detach()
        x = x + pos_embed
        for blk in self.local_image_blocks:
            x = blk(x)

        for blk in self.global_blocks:
            x = blk(x)
        B, D, T, N = x.shape
        x = self.global_bn(x.flatten(2)).reshape(B, D, T, N)
        return x

    def forward_image_local_encoder(self, imgs, density_rev):
        x = self.patch_embed(imgs)
        B, D, T, N = x.shape
        x = x.view(B, D, self.merged_T, T // self.merged_T, N).mean(dim=3, keepdim=True)
        x = x.repeat(1, 1, 1, T // self.merged_T, 1).reshape(B, D, T, N)
        pos_embed = self.image_pos_embed.transpose(1, 2).reshape(1, self.local_embed_dims, T, self.num_patches).type_as(x).to(x.device).clone().detach()
        x = x + pos_embed
        x = rearrange(x, 'b d t n -> t b n d')
        x, mask, ids_restore = self.tube_masking(x, self.image_mask_ratio, density_rev)
        x = rearrange(x, 't b n d -> b d t n')
        mask = mask.unsqueeze(1).repeat(1, x.shape[2], 1)
        ids_restore = ids_restore.unsqueeze(1).repeat(1, x.shape[2], 1)

        for blk in self.local_image_blocks:
            x = blk(x)
        return x, mask, ids_restore

    def _mask_center_rand(self, center):
        B, G, _ = center.shape
        if self.point_mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        num_mask = int(self.point_mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([np.zeros(G - num_mask), np.ones(num_mask)])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        return overall_mask.to(center.device)

    def mask_point(self, point, center, B, C1, half_win):
        bool_masked_pos = self._mask_center_rand(center)
        point = point.transpose(1, 2)
        point = point[~bool_masked_pos].reshape(B, -1, self.local_embed_dims)
        point = point.transpose(1, 2)
        masked_center = center[~bool_masked_pos].reshape(B, -1, C1)
        pos_m = self.point_pos_embed(masked_center.transpose(-1, -2))
        point = point.reshape(half_win, B, self.local_embed_dims, -1).permute(1, 2, 0, 3)
        pos_m = pos_m.reshape(half_win, B, self.local_embed_dims, -1).permute(1, 2, 0, 3)
        functional.reset_net(self.point_encoder)
        functional.reset_net(self.point_pos_embed)
        return point, pos_m

    def forward_point_local_encoder(self, points):
        B, Win, N, C1 = points.shape
        half_win = Win // 2
        pts = points.reshape(B, half_win, 2 * N, C1)
        pts = pts.flatten(0, 1)
        # m
        neighborhood_m, center_m = self.group_divider_m(pts)
        point_m = self.point_encoder(neighborhood_m) + self.scale_m_pos_embed
        point_m, pos_m = self.mask_point(point_m, center_m, B, C1, half_win)
        # s
        neighborhood_s, center_s = self.group_divider_s(pts)
        point_s = self.point_encoder(neighborhood_s) + self.scale_s_pos_embed
        point_s, pos_s = self.mask_point(point_s, center_s, B, C1, half_win)
        # l
        neighborhood_l, center_l = self.group_divider_l(pts)
        point_l = self.point_encoder(neighborhood_l) + self.scale_l_pos_embed
        point_l, pos_l = self.mask_point(point_l, center_l, B, C1, half_win)
        point = torch.cat((point_s, point_m, point_l), dim=-1)
        pos = torch.cat((pos_s, pos_m, pos_l), dim=-1)

        for blk in self.local_point_blocks:
            point = blk(point + pos)
        return point, pos

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x_ = x + self.decoder_pos_embed.type_as(x).to(x.device).clone().detach()
        for blk in self.vit_blocks:
            x_ = blk(x_)

        x = self.decoder_norm(x_)
        x = self.decoder_pred(x)
        return x

    def NT_XentLoss(self, z1, z2, temperature):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        B, D = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        l_pos = torch.diag(similarity_matrix, B)  # B
        r_pos = torch.diag(similarity_matrix, -B)  # B
        positives = torch.cat([l_pos, r_pos]).view(2 * B, 1)
        diag = torch.eye(2 * B, dtype=torch.bool, device=device)
        diag[B:, :B] = diag[:B, B:] = diag[:B, :B]
        negatives = similarity_matrix[~diag].view(2 * B, -1)
        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature
        labels = torch.zeros(2 * B, device=device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * B)

    def forward(self, points, frames):
        frames = frames.transpose(1, 2)
        with torch.no_grad():
            T = frames.shape[2]
            density = nn.AvgPool3d((T, 16, 16), (T, 16, 16))(frames / 2 + 0.5)
            density = density.mean(1, True)
            min_val = density.amin(dim=(1, 2, 3, 4), keepdim=True)
            max_val = density.amax(dim=(1, 2, 3, 4), keepdim=True)
            density = (density - min_val) / (max_val - min_val + 1e-8)
            density_rev = 1 - density

        loss = 0
        tea_latent = self.forward_teacher_encoder(frames)
        functional.reset_net(self.patch_embed)
        functional.reset_net(self.local_image_blocks)
        functional.reset_net(self.global_blocks)
        images, mask, ids_restore = self.forward_image_local_encoder(frames, density_rev)

        point_features, pos_features = self.forward_point_local_encoder(points)
        proj1 = self.projection(images.mean(-1).mean(-1))
        proj2 = self.projection(point_features.mean(-1).mean(-1))
        pred1 = self.predictor(proj1)
        pred2 = self.predictor(proj2)
        contrast_loss = (F.mse_loss(pred2, proj1.detach()) + F.mse_loss(pred1, proj2.detach())) / 2
        loss = loss + contrast_loss

        for blk in self.global_blocks:
            images = blk(images, point_features + pos_features)

        B, D, T, N = images.shape
        images = self.global_bn(images.flatten(2)).reshape(B, D, T, N)
        align_loss = self.NT_XentLoss(tea_latent.mean(-1).mean(-1), images.mean(-1).mean(-1), self.temperature)
        loss = loss + align_loss

        images = rearrange(images, 'b d t n -> b (t n) d')
        mask = rearrange(mask, 'b t n -> b (t n)')
        ids_restore = rearrange(ids_restore, 'b t n -> b (t n)')
        x = self.forward_decoder(images, ids_restore)
        rec_loss = self.forward_rec_loss(x, frames, mask)
        loss = loss + rec_loss
        return loss


@register_model
def model_6_512(pretrained=False, **kwargs):
    model = MAE(
        img_size=224, patch_size=16, image_channels=2, pool_state=[True, True, True, True],
        in_channels=kwargs['in_channels'], mlp_ratios=4, image_timestep=kwargs['image_timestep'], n_win=kwargs['n_win'],
        num_group=kwargs['num_group'], group_size=kwargs['group_size'],
        local_embed_dims=512, local_num_heads=16, local_image_depths=1, local_point_depths=1,
        global_embed_dims=512, global_num_heads=16, global_depths=4,
        decoder_embed_dims=256, decoder_num_heads=8, decoder_depths=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        image_mask_ratio=kwargs['image_mask_ratio'], point_mask_ratio=kwargs['point_mask_ratio'],
        merged_number_windows=kwargs['merged_number_windows'],
        drop_path=kwargs['drop_path'], num_classes=kwargs['num_classes'], TET=kwargs['TET'],
        backend=kwargs['backend'], lif_tau=kwargs['lif_tau'],
    )
    return model
@register_model
def model_6_384(pretrained=False, **kwargs):
    model = MAE(
        img_size=224, patch_size=16, image_channels=2, pool_state=[True, True, True, True],
        in_channels=kwargs['in_channels'], mlp_ratios=4, image_timestep=kwargs['image_timestep'], n_win=kwargs['n_win'],
        num_group=kwargs['num_group'], group_size=kwargs['group_size'],
        local_embed_dims=384, local_num_heads=16, local_image_depths=1, local_point_depths=1,
        global_embed_dims=384, global_num_heads=16, global_depths=4,
        decoder_embed_dims=192, decoder_num_heads=8, decoder_depths=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        image_mask_ratio=kwargs['image_mask_ratio'], point_mask_ratio=kwargs['point_mask_ratio'],
        merged_number_windows=kwargs['merged_number_windows'],
        drop_path=kwargs['drop_path'], num_classes=kwargs['num_classes'], TET=kwargs['TET'],
        backend=kwargs['backend'], lif_tau=kwargs['lif_tau'],
    )
    return model
@register_model
def model_6_256(pretrained=False, **kwargs):
    model = MAE(
        img_size=224, patch_size=16, image_channels=2, pool_state=[True, True, True, True],
        in_channels=kwargs['in_channels'], mlp_ratios=4, image_timestep=kwargs['image_timestep'], n_win=kwargs['n_win'],
        num_group=kwargs['num_group'], group_size=kwargs['group_size'],
        local_embed_dims=256, local_num_heads=8, local_image_depths=1, local_point_depths=1,
        global_embed_dims=256, global_num_heads=8, global_depths=4,
        decoder_embed_dims=128, decoder_num_heads=4, decoder_depths=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        image_mask_ratio=kwargs['image_mask_ratio'], point_mask_ratio=kwargs['point_mask_ratio'],
        merged_number_windows=kwargs['merged_number_windows'],
        drop_path=kwargs['drop_path'], num_classes=kwargs['num_classes'], TET=kwargs['TET'],
        backend=kwargs['backend'], lif_tau=kwargs['lif_tau'],
    )
    return model
@register_model
def model_4_128(pretrained=False, **kwargs):
    model = MAE(
        img_size=224, patch_size=16, image_channels=2, pool_state=[True, True, True, True],
        in_channels=kwargs['in_channels'], mlp_ratios=4, image_timestep=kwargs['image_timestep'], n_win=kwargs['n_win'],
        num_group=kwargs['num_group'], group_size=kwargs['group_size'],
        local_embed_dims=128, local_num_heads=4, local_image_depths=1, local_point_depths=1,
        global_embed_dims=128, global_num_heads=4, global_depths=2,
        decoder_embed_dims=64, decoder_num_heads=4, decoder_depths=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        image_mask_ratio=kwargs['image_mask_ratio'], point_mask_ratio=kwargs['point_mask_ratio'],
        merged_number_windows=kwargs['merged_number_windows'],
        drop_path=kwargs['drop_path'], num_classes=kwargs['num_classes'], TET=kwargs['TET'],
        backend=kwargs['backend'], lif_tau=kwargs['lif_tau'],
    )
    return model

