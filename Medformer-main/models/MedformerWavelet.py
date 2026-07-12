"""
MedformerWavelet — Wavelet频域分支 + Medformer时域 + Concat融合

横向对比实验: 验证FFT不是唯一有效的频域变换。
Wavelet (小波变换) 有时频局部化能力, 在ECG信号处理中非常常见
(QRS检测、ST段分析常用小波)。
如果FFT > Wavelet: 说明全局频谱就够了, 不需要局部时频信息。
如果Wavelet > FFT: 说明时频局部化有优势, 但计算代价更大。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

from layers.Embed import ListPatchEmbedding
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer


class WaveletEncoder(nn.Module):
    """用小波变换替换FFT, 输出相同维度的频域特征。"""
    def __init__(self, seq_len, enc_in, d_model, dropout, wavelet='db4', level=3):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.seq_len = seq_len
        self.enc_in = enc_in

        # 预估小波系数的总长度 (近似, 每层分解后近似系数减半)
        # level=3: approx_len ≈ seq_len/8 + seq_len/8 + seq_len/4 + seq_len/2
        approx_len = (seq_len + 2**level - 1) // (2**level)
        total_coeffs = approx_len
        for l in range(level, 0, -1):
            total_coeffs += (seq_len + 2**l - 1) // (2**l)
        self.flatten_dim = total_coeffs * enc_in

        # 如果超出or不足, 后面会截断或padding处理
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

        # wavelet decomposition per channel, per sample
        coeffs_list = []
        for b in range(B):
            sample_coeffs = []
            for c in range(C):
                signal = x_enc[b, :, c].cpu().numpy()
                coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
                # flatten all coefficients
                flat = np.concatenate([np.atleast_1d(c) for c in coeffs])
                sample_coeffs.append(flat)
            # concatenate across channels
            all_coeffs = np.concatenate(sample_coeffs)
            coeffs_list.append(all_coeffs)

        # pad/truncate to fixed size
        max_len = self.flatten_dim
        padded = np.zeros((B, max_len), dtype=np.float32)
        for i, c in enumerate(coeffs_list):
            length = min(len(c), max_len)
            padded[i, :length] = c[:length]

        freq_feature = torch.from_numpy(padded).to(device)
        return self.encoder(freq_feature)


class TimeFrequencyModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
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
        self.wavelet_encoder = WaveletEncoder(
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
        freq_feature = self.wavelet_encoder(x_enc)
        fused_feature = torch.cat([time_feature, freq_feature], dim=-1)
        return self.projection(fused_feature)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        raise NotImplementedError("MedformerWavelet only supports classification.")


class Model(TimeFrequencyModel):
    pass
