import torch
from torch import nn
from vit_pytorch.sinkhorn import SinkhornDistance
from vit_pytorch.swd import SWD5, SWD6, SWD7, SWD8
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

def Haar_wavelet_basis(num_col, num_basis):
    interval = max(1, num_col // num_basis)
    idx_basis = [value for idx in range(num_col // (interval * 2)) for value in
                 range((idx * 2 + 1) * interval, (idx * 2 + 2) * interval)]
    if num_basis > 1:
        idx_basis.extend(list(range(idx_basis[-1] + interval, num_col)))
    return idx_basis


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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., max_iter=3, eps=1, attn='trans', layer_idx=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.max_iter = max_iter
        self.swd = SWD8()
        self.sink = SinkhornDistance(eps=eps, max_iter=max_iter)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.attn = attn
        self.col_descend = Haar_wavelet_basis(num_col=dim_head, num_basis=2 ** layer_idx)

    def forward(self, x, training=True):
        # with torch.no_grad():
        #     self.to_qkv.weight.div_(torch.norm(self.to_qkv.weight, dim=1, keepdim=True))
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if self.attn == 'trans':
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        elif self.attn == 'sink':
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            dots_former_shape = dots.shape
            dots = dots.view(-1, dots_former_shape[2], dots_former_shape[3])
            attn = self.sink(dots)[0]
            attn = attn * attn.shape[-1]
            attn = attn.view(dots_former_shape)
            out = torch.matmul(attn, v)
        elif self.attn == 'swd':
            out, attn = self.swd(q, k, v, self.col_descend)
            # U, S, Vh = torch.linalg.svd(out)
            # print(S[0,0])

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1, max_iter=1, eps=1, attn='trans'):
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx, _ in enumerate(range(depth)):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, max_iter=max_iter, eps=eps,
                                       attn=attn, layer_idx=depth-idx-1)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, training=True):
        attn_weights = []
        for idx, (attn, ff) in enumerate(self.layers):
            
            # if idx == 0:
            #     # attn_x, attn_matrix = attn(x, layer_idx=idx)
            #     # x_sorted, x_indices = x.sort(dim=-2, descending=True)
            #     # x = attn_x + x_sorted
            #     attn_x, attn_matrix = attn(x, training=training)

            #     x_sorted, x_indices = x.sort(dim=-2, descending=True)
            #     col_descend = Haar_wavelet_basis(num_col=x.size(-1), num_basis=2 ** len(self.layers))
            #     col_descend = torch.tensor(col_descend, device=x.device, dtype=torch.long).flatten()
            #     x_sorted[..., col_descend] = x_sorted[..., col_descend].flip(-2)

            #     x = attn_x + x_sorted
            # else:
            #     attn_x, attn_matrix = attn(x)
            #     x = attn_x + x
            
            attn_x, attn_matrix = attn(x)
            x = attn_x + x
            x = ff(x) + x
            attn_weights.append(attn_matrix.detach().clone().cpu())
            
        return x, attn_weights


class ViT_Haar(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., max_iter=1, eps=1, attn='trans'):
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

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, max_iter=max_iter, eps=eps,
                                       attn=attn)

        self.pool = pool
        self.to_latent = nn.Identity()

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
        x = self.dropout(x)

        trans_x, attn_weights = self.transformer(x, training=training)
        x = trans_x

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), attn_weights
