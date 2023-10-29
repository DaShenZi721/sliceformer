import torch
import torch.nn as nn
import torch.nn.functional as F
import time



def SWD9(v):
    # |v| : (v_len, d_v)
    d_v = v.size(-1)

    v_sorted, v_indices = v.sort(dim=-2)
    v1_indices = torch.cat([v_indices[:, 1:], v_indices[:, 0].unsqueeze(-1)], dim=-1)
    _, v1_indices_T = v1_indices.sort(dim=-2)
    
    out = v.gather(dim=-2, index=v_indices)
    out = out.gather(dim=-2, index=v1_indices_T)
    
    return out

def SWD11(v):
    # |v| : (v_len, d_v)
    d_v = v.size(-1)
    v_len = v.size(-2)
    
    K = 3

    v_sorted, v_indices = v.sort(dim=-2)
    v_onehot = F.one_hot(v_indices).to(torch.float32)
    for k in range(1, K+1):
        vk_indices = torch.cat([v_indices[:, k:], v_indices[:, :k]], dim=-1)
        _, vk_indices_T = vk_indices.sort(dim=-2)
        vk_indices_T = vk_indices_T.unsqueeze(-1).repeat(1, 1, v_len)
        vk_onehot = v_onehot.gather(-3, vk_indices_T)
        if k == 1:
            P = vk_onehot
        else:
            P = P + vk_onehot
    
    P = P.permute(1, 0, 2)
    out = torch.matmul(P, v.unsqueeze(-1).permute(1, 0, 2)).squeeze(-1).permute(1, 0)

    return out


if __name__=='__main__':
    seq_len = 10
    d_v = 4
    
    permute_indices = torch.randperm(seq_len)
    v = torch.randn(seq_len, d_v)
    v_permute = v[permute_indices]
    
    out = SWD11(v)
    out_permute = SWD11(v_permute)
    
    print(out)
    print(out_permute)
    print((out[permute_indices]==out_permute).all())
    print((out[permute_indices]!=out_permute).nonzero())
    print((out[permute_indices]-out_permute))
    
    
    

