import torch
import torch.nn as nn
import time

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
# and from https://github.com/dfdazac/wassdistance/blob/master/layers.py

class SWD(nn.Module):
    def __init__(self, ):
        super(SWD, self).__init__()

    def forward(self, q, k, attn_mask):
        attn_mask_shape = attn_mask.shape
        q = q.contiguous().view(-1, q.size(-2), q.size(-1))
        k = k.contiguous().view(-1, k.size(-2), k.size(-1))
        attn_mask = attn_mask.contiguous().view(-1, attn_mask.size(-2), attn_mask.size(-1))

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        p = []
        for batch in range(q.size(0)):
            pd = torch.zeros(q.size(-2), k.size(-2)).to(q.device)
            for di in range(q.size(-1)):
                q_index = q_indices[batch, :, di]
                k_index = k_indices[batch, :, di]

                pi = torch.zeros(q.size(-2), k.size(-2)).to(q.device)
                pi[q_index, k_index] = 1

                qi = q[batch, :, di].view(-1, 1)
                ki = k[batch, :, di].view(1, -1)
                ci = torch.abs(qi-ki).pow(2)
                pd = pd + pi * torch.exp(-ci)
            pd = pd / q.size(-1)
            p.append(pd.unsqueeze(0))
        p = torch.cat(p, dim=0).masked_fill(attn_mask, 0).view(attn_mask_shape)
        return p

class SWD2(nn.Module):
    def __init__(self, ):
        super(SWD2, self).__init__()

    def forward(self, q, k, attn_mask):
        attn_mask_shape = attn_mask.shape
        q = q.contiguous().view(-1, q.size(-2), q.size(-1))
        k = k.contiguous().view(-1, k.size(-2), k.size(-1))
        attn_mask = attn_mask.contiguous().view(-1, attn_mask.size(-2), attn_mask.size(-1))

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        p = []
        for batch in range(q.size(0)):
            pd = torch.sparse_coo_tensor(size=(q.size(-2), k.size(-2))).to(q.device)
            for di in range(q.size(-1)):
                start_time = time.time()
                qi = q[batch, :, di].view(-1, 1)
                ki = k[batch, :, di].view(1, -1)
                ci = torch.abs(qi-ki).pow(2)
                ci = torch.exp(-ci)
                print('ci %.4f'%((time.time()-start_time)*100))

                start_time = time.time()
                q_index = q_indices[batch, :, di]
                k_index = k_indices[batch, :, di]
                coo_indices = torch.cat([q_index.unsqueeze(0), k_index.unsqueeze(0)], dim=0)
                pd = pd + torch.sparse_coo_tensor(coo_indices, ci[q_index, k_index])
                print('coo %.4f'%((time.time()-start_time)*100))

            start_time = time.time()
            pd = pd / q.size(-1)
            p.append(pd.to_dense().unsqueeze(0))
            print('to_dense %.4f'%((time.time()-start_time)*100))
        p = torch.cat(p, dim=0).masked_fill(attn_mask, 0).view(attn_mask_shape)
        return p

class SWD3(nn.Module):
    def __init__(self, ):
        super(SWD3, self).__init__()

    def forward(self, q, k, attn_mask):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        qd = q.unsqueeze(-1).repeat(1, 1, 1, 1, q_len)
        kd = k.unsqueeze(-1).repeat(1, 1, 1, 1, k_len).permute(0, 1, 4, 3, 2)
        c = torch.abs(qd-kd).pow(2)
        c = torch.exp(-c)

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        indices = q_indices * q_len + k_indices
        indices = indices.permute(0, 1, 3, 2).view(batch_size, n_heads, d_k, -1)
        c = c.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, n_heads, d_k, -1)
        p = torch.zeros(c.shape).to(c.device)
        p = p.scatter(-1, indices, c.gather(-1, indices)).sum(-2) / d_k
        p = p.view(batch_size, n_heads, q_len, k_len).masked_fill(attn_mask, 0)
        return p


class SWD4(nn.Module):
    def __init__(self):
        super(SWD4, self).__init__()

    def forward(self, q, k, attn_mask, stage='train'):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        k_indices_q = q_indices*k_len + k_indices
        # |k_indices_q| : (batch_size, n_heads, q_len*d_k)
        k_indices_q = k_indices_q.gather(-2, q_indices).view(batch_size, n_heads, -1)
        c = torch.exp(-torch.abs(q_sorted - k_sorted).pow(2))
        # |c_q| : (batch_size, n_heads, q_len*d_k)
        c_q = c.gather(-2, q_indices).view(batch_size, n_heads, -1) / d_k

        p = torch.zeros(batch_size, n_heads, q_len*k_len).to(q.device)
        p.scatter_add_(-1, k_indices_q, c_q)
        p = p.view(batch_size, n_heads, q_len, k_len).masked_fill(attn_mask, 0)

        # attn_mask_shape = attn_mask.shape
        # q = q.contiguous().view(-1, q.size(-2), q.size(-1))
        # k = k.contiguous().view(-1, k.size(-2), k.size(-1))
        # attn_mask = attn_mask.contiguous().view(-1, attn_mask.size(-2), attn_mask.size(-1))
        #
        # q_sorted, q_indices = q.sort(dim=-2)
        # k_sorted, k_indices = k.sort(dim=-2)
        # p2 = []
        # for batch in range(q.size(0)):
        #     pd = torch.zeros(q.size(-2), k.size(-2)).to(q.device)
        #     for di in range(q.size(-1)):
        #         q_index = q_indices[batch, :, di]
        #         k_index = k_indices[batch, :, di]
        #
        #         pi = torch.zeros(q.size(-2), k.size(-2)).to(q.device)
        #         pi[q_index, k_index] = 1
        #
        #         qi = q[batch, :, di].view(-1, 1)
        #         ki = k[batch, :, di].view(1, -1)
        #         ci = torch.abs(qi-ki).pow(2)
        #         pd = pd + pi * torch.exp(-ci)
        #     pd = pd / q.size(-1)
        #     p2.append(pd.unsqueeze(0))
        # p2= torch.cat(p2, dim=0).masked_fill(attn_mask, 0).view(attn_mask_shape)

        return p


class SWD5(nn.Module):
    def __init__(self):
        super(SWD5, self).__init__()

    def forward(self, q, k, v, stage='train'):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        k_indices_q = q_indices*k_len + k_indices
        # |k_indices_q| : (batch_size, n_heads, q_len*d_k)
        k_indices_q = k_indices_q.gather(-2, q_indices).view(batch_size, n_heads, -1)
        c = torch.exp(-torch.abs(q_sorted - k_sorted).pow(2))
        # |c_q| : (batch_size, n_heads, q_len*d_k)
        c_q = c.gather(-2, q_indices).view(batch_size, n_heads, -1) / d_k
        # if stage != 'train':
        #     c_q = (1 - c_q.detach() + c_q) / d_k

        p = torch.zeros(batch_size, n_heads, q_len*k_len, device=q.device)
        p.scatter_add_(-1, k_indices_q, c_q)
        p = p.view(batch_size, n_heads, q_len, k_len)
        # p = p + p.transpose(-2, -1)
        out = torch.matmul(p, v)
        return out, p


class SWD6(nn.Module):
    def __init__(self):
        super(SWD6, self).__init__()

    def forward(self, q, k, v, stage='train'):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        k_indices_q = q_indices*k_len + k_indices
        # |k_indices_q| : (batch_size, n_heads, q_len, d_k)
        k_indices_q = k_indices_q.gather(-2, q_indices)
        c = torch.exp(-torch.abs(q_sorted - k_sorted).pow(2))
        # |c_q| : (batch_size, n_heads, q_len, d_k)
        c_q = c.gather(-2, q_indices) / d_k

        out = []
        for d in range(d_k):
            p = torch.zeros(batch_size, n_heads, q_len*k_len, device=q.device)
            p.scatter_(-1, k_indices_q[:, :, :, d], c_q[:, :, :, d])
            p = p.view(batch_size, n_heads, q_len, k_len)
            p = torch.matmul(p, v[:, :, :, d].unsqueeze(-1))
            out.append(p)
        out = torch.cat(out, dim=-1)
        return out, p


class SWD7(nn.Module):
    def __init__(self):
        super(SWD7, self).__init__()
        self.N = 2
        self.weight = nn.Parameter(torch.randn(self.N))
        self.activation = nn.Softmax(dim=-1)

    def forward(self, q, k, v, stage='train'):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        # q_sorted, q_indices = q.sort(dim=-2)
        # # _, q_rank = q_indices.sort(dim=-2)
        # k_sorted, k_indices = k.sort(dim=-2)
        # v_sorted, v_indices = v.sort(dim=-2)
        # out = v_sorted

        # img_ps = 16 # 一共16*16=256个patch
        #
        # vs = []
        # vs.append(v[:,:,0,:].unsqueeze(-2))
        # for i in range(k_len//img_ps):
        #     # +1 因为CLS
        #     v_sorted, _ = v[:,:,1+i*img_ps:1+(i+1)*img_ps,:].sort(dim=-2)
        #     vs.append(v_sorted)
        # v = torch.cat(vs, dim=-2)
        #
        # new_v = torch.zeros_like(v)
        # new_v[:,:,0,:] = v[:,:,0,:]
        # for i in range(img_ps):
        #     # +1 因为CLS
        #     indices = [1+i+j*img_ps for j in range(k_len//img_ps)]
        #     v_sorted, _ = v[:,:,indices,:].sort(dim=-2)
        #     new_v[:,:,indices,:] = v_sorted
        # v = new_v
        # out = v

        # max pooling
        # new_v = v.clone()
        # values, _ = torch.max(v, dim=-2)
        # new_v[:,:,0,:] = values
        # out = new_v

        # min pooling
        # new_v = v.clone()
        # values, _ = torch.max(v, dim=-2)
        # new_v[:,:,0,:] = values
        # out = new_v

        # Max abs with sign
        # new_v = v.clone()
        # values, _ = torch.max(v, dim=-2)
        # abs_values, _ = torch.max(torch.abs(v), dim=-2)
        # values = torch.where(values==abs_values, abs_values, -abs_values)
        # new_v[:,:,0,:] = values
        # out = new_v

        # Max abs
        # new_v = v.clone()
        # abs_values, _ = torch.max(torch.abs(v), dim=-2)
        # new_v[:,:,0,:] = abs_values
        # out = new_v

        # max exchange
        # new_v = v.clone()
        # values, indices = torch.max(v, dim=-2)
        # v_cls = v[:, :, 0, :]
        # new_v[:, :, 0, :] = values
        # new_v.scatter_(-2, indices.unsqueeze(-2), v_cls.unsqueeze(-2))
        # out = new_v

        # sorting
        # v_sorted, v_indices = v.sort(dim=-2)
        # out = v_sorted
        v_sorted, v_indices = v.abs().sort(dim=-2, descending=True)
        out = v_sorted

        # v_sorted, v_indices = v.sort(dim=-2)
        #
        # def get_N_sort(v, indices, N):
        #     for n in range(N):
        #         v = v.gather(-2, indices)
        #     return v
        #
        # # out = (1/self.N) * v_sorted
        # # for n in range(2, self.N+1):
        # #     out = out + (1/self.N) * get_N_sort(v, v_indices, n)
        # weight = self.activation(self.weight)
        # out = weight[0] * v_sorted
        # for n in range(2, self.N+1):
        #     out = out + weight[n-1] * get_N_sort(v, v_indices, n)
        # print(weight[0])

        # q_sorted2 = q_sorted.flip(dims=[-2])
        # k_sorted2 = k_sorted.flip(dims=[-2])
        # v_sorted2 = v_sorted.flip(dims=[-2])
        # c = torch.exp(-torch.abs(q_sorted - k_sorted).pow(2))
        # out = c * v_sorted
        # out = out + torch.exp(-torch.abs(q_sorted2 - k_sorted2).pow(2)) * v_sorted2

        # |c_q| : (batch_size, n_heads, q_len, d_k)
        # c_q = c.gather(-2, q_rank)
        # k_indices_q = k_indices.gather(-2, q_rank)
        # v_q = v.gather(-2, k_indices_q)
        # out = c_q * v_q
        # print(c.max(), c.min())
        # print(c[0,0].max(), c[0,0].min())
        # print(c[0,0,:,0].max(), c[0,0,:,0].min())
        # print(c[0,0,:,0])
        #
        # print(out.max(), out.min())
        # print(out[0,0].max(), out[0,0].min())
        # print(out[0,0,:,0].max(), out[0,0,:,0].min())
        # print(out[0,0,:,0])
        # 1/0
        return out, None

class SWD8(nn.Module):
    def __init__(self):
        super(SWD8, self).__init__()

    def forward(self, q, k, v, col_descend):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        d_v = v.size(-1)

        v_sorted, v_indices = v.sort(dim=-2)
        out = v_sorted

        col_descend = torch.tensor(col_descend, device=out.device, dtype=torch.long).flatten()
        out[..., col_descend] = out[..., col_descend].flip(-2)
        return out, None

def Haar_wavelet_basis(num_col, num_basis):
    interval = max(1, num_col // num_basis)
    idx_basis = [value for idx in range(num_col // (interval * 2)) for value in
                 range((idx * 2 + 1) * interval, (idx * 2 + 2) * interval)]
    if num_basis > 1:
        idx_basis.extend(list(range(idx_basis[-1] + interval, num_col)))
    return idx_basis

