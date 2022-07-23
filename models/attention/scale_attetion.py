import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout):
        # TODO:继承哪个的？
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        # softmax互斥多分类问题
        self.softmax = nn.Softmax(dim=2)  # dimension=2

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播
        Args:
                q: Queries [B, L_q, D_q]
                k: Keys [B, L_k, D_k]
                v: Values [B, L_v, D_v]，一般来说就是k  TODO:why???
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
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention
