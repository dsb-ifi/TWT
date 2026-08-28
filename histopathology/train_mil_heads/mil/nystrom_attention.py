from math import ceil

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum, nn


def moore_penrose_iter_pinv(matrix, iterations=6):
    """Approximate a matrix pseudoinverse with the iterative Nyström update."""
    absolute = matrix.abs()
    column_norm = absolute.sum(dim=-1)
    row_norm = absolute.sum(dim=-2)
    inverse = rearrange(matrix, "... i j -> ... j i") / (torch.max(column_norm) * torch.max(row_norm))

    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    identity = rearrange(identity, "i j -> () i j")
    for _ in range(iterations):
        product = matrix @ inverse
        inverse = 0.25 * inverse @ (
            13 * identity - product @ (15 * identity - product @ (7 * identity - product))
        )
    return inverse


class NystromAttention(nn.Module):
    """Nyström self-attention implementation used by the released TransMIL head."""

    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.0,
    ):
        """Initialise projections, landmark settings, and the optional value residual convolution."""
        super().__init__()
        inner_dim = heads * dim_head
        self.eps = eps
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads,
                heads,
                (residual_conv_kernel, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False,
            )

    def forward(self, x, mask=None):
        """Apply landmark-based Nyström self-attention to a token sequence."""
        _, original_length, _ = x.shape
        heads = self.heads
        landmarks = self.num_landmarks

        remainder = original_length % landmarks
        if remainder:
            padding = landmarks - remainder
            x = F.pad(x, (0, 0, padding, 0), value=0)
            if mask is not None:
                mask = F.pad(mask, (padding, 0), value=False)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda tensor: rearrange(tensor, "b n (h d) -> b h n d", h=heads), (q, k, v))

        if mask is not None:
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda tensor: tensor * mask[..., None], (q, k, v))

        q = q * self.scale
        tokens_per_landmark = ceil(x.shape[1] / landmarks)
        reduction = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, reduction, "sum", l=tokens_per_landmark)
        k_landmarks = reduce(k, reduction, "sum", l=tokens_per_landmark)

        divisor = tokens_per_landmark
        if mask is not None:
            landmark_counts = reduce(mask, "... (n l) -> ... n", "sum", l=tokens_per_landmark)
            divisor = landmark_counts[..., None] + self.eps
            landmark_mask = landmark_counts > 0

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        equation = "... i d, ... j d -> ... i j"
        sim1 = einsum(equation, q, k_landmarks)
        sim2 = einsum(equation, q_landmarks, k_landmarks)
        sim3 = einsum(equation, q_landmarks, k)

        if mask is not None:
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * landmark_mask[..., None, :]), mask_value)
            sim2.masked_fill_(~(landmark_mask[..., None] * landmark_mask[..., None, :]), mask_value)
            sim3.masked_fill_(~(landmark_mask[..., None] * mask[..., None, :]), mask_value)

        attn1, attn2, attn3 = map(lambda tensor: tensor.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inverse = moore_penrose_iter_pinv(attn2, self.pinv_iterations)
        output = (attn1 @ attn2_inverse) @ (attn3 @ v)

        if self.residual:
            output = output + self.res_conv(v)

        output = rearrange(output, "b h n d -> b n (h d)", h=heads)
        output = self.to_out(output)
        return output[:, -original_length:]
