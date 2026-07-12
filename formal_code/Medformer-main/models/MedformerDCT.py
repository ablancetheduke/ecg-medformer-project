"""
MedformerDCT — DCT频域分支 + Medformer时域 + Concat融合

横向对比实验第三变换: 离散余弦变换 (DCT Type-II)
DCT是实数变换, 能量集中在前几个系数, JPEG用的就是DCT。
对比逻辑:
  FFT (复数, 全局频谱) vs DCT (实数, 能量集中) vs Wavelet (时频局部化)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dct
import numpy as np

from layers.Embed import ListPatchEmbedding
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer


class DCTEncoder(nn.Module):
    """DCT变换替换FFT, 输出相同维度的频域特征。"""
    def __init__(self, seq_len, enc_in, d_model, dropout):
        super().__init__()
        self.flatten_dim = seq_len * enc_in
        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )

    def forward(self, x_enc):
        B, T, C = x_enc.shape
        device = x_enc.device
        # DCT along time dimension, per channel
        x_np = x_enc.cpu().numpy()
        dct_coeffs = dct(x_np, type=2, axis=1, norm='ortho')  # [B, T, C]
        dct_flat = dct_coeffs.reshape(B, -1)
        freq_feature = torch.from_numpy(dct_flat.astype(np.float32)).to(device)
        return self.encoder(freq_feature)


class TimeFrequencyModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.single_channel = configs.single_channel

        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")

        self.enc_embedding = ListPatchEmbedding(
            configs.enc_in, configs.d_model, configs.seq_len,
            patch_len_list, stride_list, configs.dropout,
            augmentations, configs.single_channel,
        )
        self.encoder = Encoder(
            [EncoderLayer(
                MedformerLayer(len(patch_len_list), configs.d_model,
                    configs.n_heads, configs.dropout,
                    configs.output_attention, configs.no_inter_attn),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation)
             for _ in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.time_feature_dim = (
            configs.d_model * len(patch_num_list)
            * (1 if not self.single_channel else configs.enc_in)
        )
        self.dct_encoder = DCTEncoder(
            configs.seq_len, configs.enc_in, configs.d_model, configs.dropout
        )
        self.projection = nn.Linear(
            self.time_feature_dim + configs.d_model,
            configs.num_class,
        )

    def extract_time_feature(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))
        output = self.act(enc_out)
        output = self.dropout(output)
        return output.reshape(output.shape[0], -1)

    def classification(self, x_enc, x_mark_enc):
        time_feature = self.extract_time_feature(x_enc)
        freq_feature = self.dct_encoder(x_enc)
        fused_feature = torch.cat([time_feature, freq_feature], dim=-1)
        return self.projection(fused_feature)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        raise NotImplementedError("MedformerDCT only supports classification.")


class Model(TimeFrequencyModel):
    pass
