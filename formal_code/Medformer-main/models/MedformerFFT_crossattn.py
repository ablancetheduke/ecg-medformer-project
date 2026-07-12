"""
MedformerFFT with Cross-Attention Fusion (FIXED VERSION).

修复说明（2026-06-03）:

=== 旧版Bug ===
FrequencyEncoder 把整个FFT频谱压成1个向量 [B, d_model]。
Cross-Attention时: Query=1个token, Key=1个token, Value=1个token。
MultiheadAttention的8个头全部退化——softmax(1个值)=1, 权重恒为1,
注意力完全没学到东西。这就是AUROC只有0.84的原因。

=== 修复方案 ===
把FFT频谱按频段拆成多个bin(如8个频段), 每段独立编码为一个token。
现在: Query=1个时域token, Key/Value=8个频域token。
MultiheadAttention的8个头可以分别关注不同频段的频谱特征。
每个头可以学到不同的频段偏好(如头1关注低频, 头8关注高频)。

=== 频段划分 ===
FFT输出49个频率bin (seq_len=96 → 96//2+1=49)
12个导联 × 49 = 588个频率值
分8个频段: 每段约73个值, 大致对应:
  Band 0: 0-6Hz    (极低频, 基线漂移)
  Band 1: 6-12Hz   (心率频段)
  Band 2: 12-18Hz  (QRS谐波)
  Band 3: 18-24Hz  (T波相关)
  Band 4: 24-30Hz  (高频谐波)
  Band 5: 30-36Hz
  Band 6: 36-42Hz
  Band 7: 42-49Hz  (极高频)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import ListPatchEmbedding
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer


class MultiBandFrequencyEncoder(nn.Module):
    """将FFT频谱拆成多个频段, 每段编码为一个token, 输出多个频域token。"""
    def __init__(self, seq_len, enc_in, d_model, num_bands, dropout):
        super().__init__()
        self.num_bands = num_bands
        freq_bins = seq_len // 2 + 1  # 49 for seq_len=96
        # 每个频段的输入维度 (可能有1个bin的余数丢给最后一段)
        band_input_dim = (freq_bins * enc_in) // num_bands
        self.band_sizes = [band_input_dim] * num_bands
        # 最后一个band可能略大, 补齐余数
        remainder = (freq_bins * enc_in) % num_bands
        if remainder > 0:
            self.band_sizes[-1] += remainder

        # 每个频段共享同一个MLP编码器
        self.band_encoder = nn.Sequential(
            nn.Linear(max(self.band_sizes), d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )

    def forward(self, x_enc):
        B = x_enc.shape[0]
        freq_mag = torch.abs(torch.fft.rfft(x_enc, dim=1))  # [B, T, C, F]
        # 拉平通道和时间维, 只保留频率维度
        freq_flat = freq_mag.reshape(B, -1)  # [B, T*C*F] = [B, freq_bins * enc_in]

        # 按频段切分
        band_features = []
        start = 0
        for size in self.band_sizes:
            band = freq_flat[:, start:start + size]  # [B, band_size]
            # 补齐到max_size以适配共享MLP
            if band.shape[1] < max(self.band_sizes):
                pad = torch.zeros(B, max(self.band_sizes) - band.shape[1], device=band.device)
                band = torch.cat([band, pad], dim=-1)
            band_vec = self.band_encoder(band)  # [B, d_model]
            band_features.append(band_vec)
            start += size

        freq_tokens = torch.stack(band_features, dim=1)  # [B, num_bands, d_model]
        return freq_tokens


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

        # 频段数量 (可调整)
        num_bands = 8
        self.frequency_encoder = MultiBandFrequencyEncoder(
            configs.seq_len, configs.enc_in, configs.d_model,
            num_bands, configs.dropout
        )

        # 先把频域token投影到和时域特征同维度
        self.freq_proj = nn.Linear(configs.d_model, self.time_feature_dim)

        # Cross-attention: 时域(1个token)关注频域(num_bands个token)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.time_feature_dim,
            num_heads=configs.n_heads,
            dropout=configs.dropout,
            batch_first=True,
        )

        self.projection = nn.Linear(self.time_feature_dim, configs.num_class)

    def extract_time_feature(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))
        output = self.act(enc_out)
        output = self.dropout(output)
        return output.reshape(output.shape[0], -1)

    def classification(self, x_enc, x_mark_enc):
        time_feature = self.extract_time_feature(x_enc)   # [B, D_time]
        freq_tokens = self.frequency_encoder(x_enc)        # [B, num_bands, d_model]

        # 投影频域token到同时域维度
        freq_kv = self.freq_proj(freq_tokens)              # [B, num_bands, D_time]

        # Cross-attention: 时域Query关注频域Key/Value
        time_query = time_feature.unsqueeze(1)              # [B, 1, D_time]
        attended, attn_weights = self.cross_attn(time_query, freq_kv, freq_kv)
        # attn_weights: [B, 1, num_bands] — 可解释！每个样本对8个频段的关注度

        fused = attended.squeeze(1)  # [B, D_time]
        return self.projection(fused)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        raise NotImplementedError("MedformerFFT_crossattn only supports classification.")


class Model(TimeFrequencyModel):
    pass
