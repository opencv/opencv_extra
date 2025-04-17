import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import argparse
import os



# Reference implementation (used for correctness verfication) can be found here: 
# https://github.com/meta-llama/llama/blob/main/llama/model.py#L80
class Rotate(nn.Module):
    def __init__(self ):
        super().__init__()

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(
        self, x, freqs_cis: torch.Tensor,
    ):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis, x_)
        x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
        return x_out.type_as(x)

    def forward(self, x: torch.Tensor):
        freqs_cis = self.precompute_freqs_cis(
            dim=x.shape[-1], end=x.shape[1], theta=10000.0
        )
        print(freqs_cis)
        return self.apply_rotary_emb(x, freqs_cis)
    
class MHARot(nn.Module):
    def __init__(self, embed_dim, num_heads, rotary_dim=None, rotate=True, qkv_proj=None):
    
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rotary_dim = rotary_dim or self.head_dim  # default: full head_dim
        self.rotate = rotate

        if qkv_proj is not None:
            self.qkv_proj = qkv_proj
        else:
            self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        
        self.rot = Rotate()

    def forward(self, x):
        B, T, _ = x.shape  # (batch_size, seq_len, embed_dim)

        qkv = self.qkv_proj(x)  # (B, T, 3 * E)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)  # (3, B, T, num_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, T, num_heads, head_dim)
        # === APPLY ROTARY POSITIONAL EMBEDDINGS ===
        if self.rotate:
            q = self.rot(q)
            k = self.rot(k)

        q = q.permute(0, 2, 1, 3) # (B, num_heads, T, head_dim)
        k = k.permute(0, 2, 1, 3) # (B, num_heads, T, head_dim)
        v = v.permute(0, 2, 1, 3) # (B, num_heads, head_dim, T)

        # === ATTENTION ===
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.embed_dim)  # (B, T, E)
        return attn_output
        # return self.out_proj(attn_output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    # Generate random input data
    x = torch.randn(args.batch_size, args.seq_len, args.embed_dim)

    # Initialize the model
    model = MHARot(args.embed_dim, args.num_heads)
    y = model(x)

    qkv_weight = model.qkv_proj.weight.detach().cpu().numpy().T
    qkv_weight = np.ascontiguousarray(qkv_weight)  

    qkv_bias = model.qkv_proj.bias.detach().cpu().numpy().T
    qkv_bias = np.ascontiguousarray(qkv_bias) 

    x_numpy = x.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()

    import numpy as np

    np.save(os.path.join(args.data_dir, 'mha_rope.QKV.npy', qkv_weight))
    np.save(os.path.join(args.data_dir,'data/mha_rope.QKV_bias.npy', qkv_bias))
    np.save(os.path.join(args.data_dir,'data/mha_rope.input.npy', x_numpy))
    np.save(os.path.join(args.data_dir,'data/mha_rope.output.npy', y_numpy))

