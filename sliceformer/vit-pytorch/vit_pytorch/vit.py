import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.sinkhorn import SinkhornDistance
from vit_pytorch.swd import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

perm = torch.randperm(64)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., max_iter=3, eps=1, attn='trans', layer_idx=None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.max_iter = max_iter
        self.swd = SWD15()
        self.sink = SinkhornDistance(eps=eps, max_iter=max_iter)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        if attn == 'swd':
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.attn = attn
        
        if layer_idx % 2:
            self.col_descend = perm[:int(dim_head/2)]
        else:
            self.col_descend = perm[int(dim_head/2):]
            
    def forward(self, x, training=True):
        # with torch.no_grad():
        #     self.to_qkv.weight.div_(torch.norm(self.to_qkv.weight, dim=1, keepdim=True))
        
        # SWD15
        cls_indices = None
        
        if self.attn == 'trans':
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            # unsort(softmax(sorted(x)))
            # dots_sorted, dots_indices = dots.sort(dim=-1)
            # attn = self.attend(dots_sorted)
            # _, dots_rank = dots_indices.sort(dim=-1)
            # attn = attn.gather(dim=-1, index=dots_rank)
            
            # softmax(x)
            attn = self.attend(dots)
            
            # F.softmax(x)
            # attn = F.softmax(dots, dim=-1, dtype=torch.bfloat16).to(torch.float)
            
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            attn = out
            # U, S, Vh = torch.linalg.svd(out)
            # print(S[0,0])
            out = rearrange(out, 'b h n d -> b n (h d)')
        elif self.attn == 'sink':
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            dots_former_shape = dots.shape
            dots = dots.view(-1, dots_former_shape[2], dots_former_shape[3])
            attn = self.sink(dots)[0]
            attn = attn * attn.shape[-1]
            attn = attn.view(dots_former_shape)
            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
        elif self.attn == 'swd':
            # qkv = self.to_qkv(x)
            # v = rearrange(qkv, 'b n (h d) -> b h n d', h = self.heads)
            # out, attn = self.swd(v, v, v, training)
            # out = rearrange(out, 'b h n d -> b n (h d)')
            
            # SWD12
            # qkv = self.to_qkv(x).chunk(3, dim = -1)
            # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
            # out, attn = self.swd(q, k, v, training)
            # out = rearrange(out, 'b h n d -> b n (h d)')
            
            # SWD8
            # qkv = self.to_qkv(x).chunk(3, dim = -1)
            # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
            # out, attn = self.swd(q, k, v, training=training, col_descend=self.col_descend)
            # U, S, Vh = torch.linalg.svd(out)
            # print(S[0,0])
            # out = rearrange(out, 'b h n d -> b n (h d)')
            
            # SWD15 no heads
            q, k, v = self.to_qkv(x).chunk(3, dim = -1)
            # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
            out, attn, cls_indices = self.swd(q, k, v, training=training, col_descend=self.col_descend)
            
        return self.to_out(out), attn, cls_indices

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1, max_iter=1, eps=1, attn='trans'):
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx, _ in enumerate(range(depth)):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, max_iter=max_iter, eps=eps, attn=attn, layer_idx=idx)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, training=True):
        attn_weights = []
        for idx, (attn, ff) in enumerate(self.layers):
            # if idx == 0:
            #     attn_x, attn_matrix = attn(x, training=training)
            #     x_sorted, x_indices = x.sort(dim=-2, descending=True)
            #     x = attn_x + x_sorted
            # else:
            #     attn_x, attn_matrix = attn(x, training=training)
            #     x = attn_x + x
            attn_x, attn_matrix, cls_indices = attn(x, training=training)
            x = attn_x + x
            x = ff(x) + x
            attn_weights.append(attn_matrix.detach().clone().cpu())
            
        return x, attn_weights, cls_indices


class Transformer_only_Att(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., max_iter=1, eps=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, max_iter=max_iter, eps=eps)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, training=True):
        attn_weights = None
        # for attn, ff in self.layers:
        #     attn_x, attn_matrix = attn(x, training=training)
        #     x = attn_x + x
        #     if attn_weights is None and attn_matrix is not None:
        #         attn_weights = attn_matrix.detach().clone()

        return x, attn_weights

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3,
                 dim_head = 64, dropout = 0., emb_dropout = 0., max_iter=1, eps=1, attn='trans'):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, max_iter=max_iter, eps=eps, attn=attn)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, training=True):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # indices_rand = torch.randperm(x.size(dim=-2) - 1)
        # x_0 = x[:, 0, :].unsqueeze(-2)
        # x_b = x[:, 1:, :][:, indices_rand, :]
        # x = torch.cat([x_0, x_b], dim=-2)

        trans_x, attn_weights, cls_indices = self.transformer(x, training=training)
        x = trans_x

        if self.pool == 'mean':
            x = x.mean(dim = 1)
        elif cls_indices is not None:
            cls_indices = cls_indices.unsqueeze(1).repeat(1, 1, x.size(-1))
            x = x.gather(dim=-2, index=cls_indices).squeeze()
        else:
            x = x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), attn_weights


class ViT_only_Att(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3,
                 dim_head = 64, dropout = 0., emb_dropout = 0., max_iter=1, eps=1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_only_Att(dim, depth, heads, dim_head, mlp_dim, dropout, max_iter=max_iter, eps=eps)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, training=True):
        x = self.to_patch_embedding(img)
        # b, n, _ = x.shape
        #
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)
        # trans_x, attn_weights = self.transformer(x, training=training)
        # x = trans_x
        #
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #
        # x = self.to_latent(x)
        return self.mlp_head(x), attn_weights
