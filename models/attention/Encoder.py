import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, attention_dropout):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        # softmax互斥多分类问题
        self.softmax = nn.Softmax(dim=2)  # dimension=2

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播
        Args:
                q: Queries [B, L_q, D_q]
                k: Keys [B, L_k, D_k]
                v: Values [B, L_v, D_v]，一般来说就是k  （自己乘自己再乘自己）
                scale: 缩放因子，一个浮点标量
                attn_mask: Masking [B, L_q, L_k]

        Returns:
                上下文张量和attention张量
        """
        # bmm要求 第一个输入(b,h,w)和第二个输入(b,w,m)，且都为Tensor，不能是numpy
        # transpose转置
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        before_v = self.dropout(attention)
        # 和V做点积
        attention = torch.bmm(before_v, v)
        return attention, before_v

    def __call__(self, x):
        return self.forward(x)


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=256, num_heads=8, ffn_dim=1024, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = Attention(dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        attention, before_v = self.attention(inputs, inputs, inputs)

        return attention, before_v

    def __call__(self, x):
        return self.forward(x)


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        fe,
        before_pool=None,
        num_layers=6,
        model_dim=256,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.0,
    ):
        super(AttentionEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, fe, before_pool=None):
        attentions = []
        for encoder in self.encoder_layers:
            attention, before_v = encoder(fe, self_attention_mask)
            attentions.append(attention)

        return output, attentions

    def __call__(self, x):
        return self.forward(x)
