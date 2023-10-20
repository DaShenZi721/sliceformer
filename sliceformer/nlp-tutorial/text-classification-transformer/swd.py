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

    def forward(self, q, k, v, attn_mask, stage='train'):
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
        # c_q = (1 - c_q.detach() + c_q)/d_k
        p = torch.zeros(batch_size, n_heads, q_len*k_len, device=q.device)
        p.scatter_add_(-1, k_indices_q, c_q)
        attn_weights = p.view(batch_size, n_heads, q_len, k_len)
        attn_weights = attn_weights + attn_weights.transpose(-2, -1)
        p = attn_weights.masked_fill(attn_mask, 0)
        out = torch.matmul(p, v)

        return out, attn_weights


class SWD7(nn.Module):
    def __init__(self):
        super(SWD7, self).__init__()

    def forward(self, q, k, v, attn_mask, stage='train'):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        # |attn_mask| : (batch_size, n_heads, q_len, k_len)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        # q_sorted, q_indices = q.sort(dim=-2)
        # # _, q_rank = q_indices.sort(dim=-2)
        # k_sorted, k_indices = k.sort(dim=-2)
        # v_sorted, v_indices = v.sort(dim=-2)
        # out = v_sorted
        # out = v * torch.exp(-torch.abs(q_sorted - k_sorted).pow(2)).masked_fill(attn_mask[:,:,0,:].unsqueeze(-1).repeat(1, 1, 1, d_k), 0)


        # new_v = v.clone()
        # values, index = torch.max(v, dim=-2)
        # new_v[:,:,0,:] = values
        # out = new_v

        # max exchange
        new_v = v.clone()
        values, indices = torch.max(v, dim=-2)
        v_cls = v[:, :, 0, :]
        new_v[:, :, 0, :] = values
        new_v.scatter_(-2, indices.unsqueeze(-2), v_cls.unsqueeze(-2))
        out = new_v

        out = out.masked_fill(attn_mask[:,:,0,:].unsqueeze(-1).repeat(1, 1, 1, d_k), 0)
        # |c_q| : (batch_size, n_heads, q_len, d_k)
        # c_q = c.gather(-2, q_rank) / d_k
        # v_q = v.gather(-2, q_rank)
        # out = (c * v).gather(-2, q_rank).masked_fill(attn_mask[:,:,0,:].unsqueeze(-1).repeat(1, 1, 1, d_k), 0)
        return out, None



