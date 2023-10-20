import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vit_pytorch.sinkhorn import SinkhornDistance
from vit_pytorch.swd import SWD5, SWD6, SWD7, SWD8, Haar_wavelet_basis
from vit_pytorch.ema import MultiHeadEMA



# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, dropout=0., max_iter=3, eps=1, attn='trans', layer_idx=None, zdim=2, hdim=2, ndim=2):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_iter = max_iter
        self.swd = SWD8()
        self.sink = SinkhornDistance(eps=eps, max_iter=max_iter)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.attn = attn
        self.col_descend = Haar_wavelet_basis(num_col=dim, num_basis=2 ** layer_idx)

        self.move = MultiHeadEMA(dim, ndim)
        self.mx_proj = nn.Linear(dim, zdim + hdim + 2 * dim)
        self.h_proj = nn.Linear(hdim, dim)
        self.v_proj = nn.Linear(dim, hdim)
        self.activation = F.silu

        # self.gamma = nn.Parameter(torch.Tensor(2, zdim))
        # self.beta = nn.Parameter(torch.Tensor(2, zdim))

        self.embed_dim = dim
        self.zdim = zdim
        self.hdim = hdim
        self.ndim = ndim

    def forward(self, x, training=True):

        residual = x
        v = self.activation(self.v_proj(x))

        # B x L x D
        mx = self.move(x, training=training)
        mx = self.dropout(mx)

        # B x L x (2*D+S+E)
        base = self.mx_proj(mx)
        u, zr, hx = torch.split(base, [self.embed_dim, self.zdim + self.hdim, self.embed_dim], dim=-1)
        # B x L x D
        u = torch.sigmoid(u)
        # B x L x (S+E)
        z, r = torch.split(F.silu(zr), [self.zdim, self.hdim], dim=-1)

        v = self.dropout(v)
        out, _ = self.swd(v, v, v, self.col_descend)

        h = self.activation(hx + self.h_proj(out * r))
        h = self.dropout(h)

        out = torch.addcmul(residual, u, h - residual)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.1, max_iter=1, eps=1, attn='trans', zdim=2, ndim=2, hdim=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx, _ in enumerate(range(depth)):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dropout=dropout, max_iter=max_iter, eps=eps, layer_idx=depth-idx-1,
                                       attn=attn, zdim=zdim, hdim=hdim, ndim=ndim)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, training=True):
        for idx, (attn, ff) in enumerate(self.layers):
            attn_x = attn(x, training=training)
            x = attn_x + x
            x = ff(x) + x
        return x


class ViT_ema(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., max_iter=1, eps=1, attn='trans', zdim=2, hdim=2, ndim=2):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, mlp_dim, dropout, max_iter=max_iter, eps=eps,
                                       attn=attn, zdim=zdim, hdim=hdim, ndim=ndim)

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, training=True):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        trans_x = self.transformer(x, training=training)
        x = trans_x

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x), None
