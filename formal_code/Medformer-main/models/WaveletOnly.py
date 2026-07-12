"""
WaveletOnly — 纯小波变换频域 + MLP (消融: 无Medformer时域分支)
对比 FrequencyOnly(FFT), 验证纯小波单支的分类能力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

from layers.Embed import ListPatchEmbedding
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer


class WaveletOnlyEncoder(nn.Module):
    def __init__(self, seq_len, enc_in, d_model, dropout, wavelet='db4', level=3):
        super().__init__()
        approx_len = (seq_len + 2**level - 1) // (2**level)
        total_coeffs = approx_len
        for l in range(level, 0, -1):
            total_coeffs += (seq_len + 2**l - 1) // (2**l)
        self.flatten_dim = total_coeffs * enc_in
        self.wavelet = wavelet
        self.level = level

        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, 5)

    def forward(self, x_enc):
        B, T, C = x_enc.shape
        device = x_enc.device
        max_len = self.flatten_dim

        coeffs_list = []
        for b in range(B):
            sample_coeffs = []
            for c in range(C):
                signal = x_enc[b, :, c].cpu().numpy()
                coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
                flat = np.concatenate([np.atleast_1d(c) for c in coeffs])
                sample_coeffs.append(flat)
            all_coeffs = np.concatenate(sample_coeffs)
            coeffs_list.append(all_coeffs)

        padded = np.zeros((B, max_len), dtype=np.float32)
        for i, c in enumerate(coeffs_list):
            length = min(len(c), max_len)
            padded[i, :length] = c[:length]

        freq_feature = torch.from_numpy(padded).to(device)
        freq_feature = self.encoder(freq_feature)
        freq_feature = self.dropout(freq_feature)
        return self.projection(freq_feature)


class Model(WaveletOnlyEncoder):
    def __init__(self, configs):
        super().__init__(
            configs.seq_len, configs.enc_in, configs.d_model,
            configs.dropout
        )
        self.task_name = configs.task_name

    def classification(self, x_enc, x_mark_enc):
        return self.forward(x_enc)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if hasattr(self, 'task_name') and self.task_name == "classification":
            return super().forward(x_enc)
        return super().forward(x_enc)
