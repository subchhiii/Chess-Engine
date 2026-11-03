"""Hybrid evaluator: CNN stem + squeeze-and-excitation residual blocks (the channel attention used in
EfficientNet's MBConv blocks) -> 64 square tokens -> transformer layers (self-attention over the 64 squares =
a graph-attention network on the fully connected graph of squares) -> hybrid pooling head
(mean-pool + max-pool + spatial summary).   Smoke test:  python hybrid_model.py"""
import math
import torch, torch.nn as nn
from attack_graph import relation_stack, V_PERM, H_PERM


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):                                   # x: (B, C, 8, 8)
        w = self.fc(x.mean(dim=(2, 3)))                     # (B, C) channel weights
        return x * w[:, :, None, None]


class SEResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                               nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        self.se = SE(c)

    def forward(self, x):
        return torch.relu(x + self.se(self.f(x)))


class HybridNet(nn.Module):
    def __init__(self, channels=64, blocks=2, layers=2, heads=4):
        super().__init__()
        c = channels
        self.stem = nn.Sequential(nn.Conv2d(18, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.blocks = nn.Sequential(*[SEResBlock(c) for _ in range(blocks)])
        self.squeeze = nn.Sequential(nn.Conv2d(c, 8, 1), nn.ReLU())          # spatial summary: 8 maps of 8x8
        self.pos = nn.Parameter(torch.randn(1, 64, c) * 0.02)               # learned square embeddings
        layer = nn.TransformerEncoderLayer(d_model=c, nhead=heads, dim_feedforward=4 * c, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(c),
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(2 * c + 8 * 64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):                                   # x: (B, 18, 8, 8) float
        h = self.blocks(self.stem(x))                       # (B, C, 8, 8)
        flat = self.squeeze(h).flatten(1)                   # (B, 512)
        t = h.flatten(2).transpose(1, 2) + self.pos         # (B, 64 squares, C)
        t = self.encoder(t)                                 # attention between all pairs of squares
        pooled = torch.cat([t.mean(dim=1), t.amax(dim=1), flat], dim=1)      # hybrid pooling
        return self.head(pooled)                            # (B, 1) in (-1, 1)


class RelAttnBlock(nn.Module):
    """Transformer block whose attention scores get an extra bias from the chess attack graph."""
    def __init__(self, c, heads):
        super().__init__()
        self.h, self.dk = heads, c // heads
        self.ln1, self.ln2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.qkv, self.proj = nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.ff = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))

    def forward(self, x, bias):                              # x: (B, 64, C); bias: (B, heads, 64, 64)
        B, L, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk) + bias
        out = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, L, C)
        x = x + self.proj(out)
        return x + self.ff(self.ln2(x))


class GraphHybridNet(nn.Module):
    """Hybrid CNN + SE blocks + graph-biased attention. The graph connects squares by chess relations:
    which piece attacks an enemy piece (threat), protects a friendly piece (defend), or attacks an empty square (quiet),
    in both directions. Each attention head learns how much weight to give each relation (Graphormer-style).
    Two extra input planes tell the CNN how many white / black pieces attack each square."""
    def __init__(self, channels=64, blocks=2, layers=2, heads=4):
        super().__init__()
        c = channels
        self.stem = nn.Sequential(nn.Conv2d(20, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.blocks = nn.Sequential(*[SEResBlock(c) for _ in range(blocks)])
        self.squeeze = nn.Sequential(nn.Conv2d(c, 8, 1), nn.ReLU())
        self.pos = nn.Parameter(torch.randn(1, 64, c) * 0.02)
        self.layers = nn.ModuleList([RelAttnBlock(c, heads) for _ in range(layers)])
        self.rel_w = nn.Parameter(torch.zeros(heads, 6))                    # per-head weight for each of the 6 relation types
        self.norm = nn.LayerNorm(c)
        self.head = nn.Sequential(nn.Linear(2 * c + 8 * 64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())
        self.register_buffer("shifts", torch.arange(7, -1, -1), persistent=False)

    def unpack(self, att):                                  # packed uint8 (B, 64, 8) -> float (B, 64, 64)
        bits = (att.long().unsqueeze(-1) >> self.shifts) & 1
        return bits.reshape(att.shape[0], 64, 64).float()

    def forward(self, x, att):                              # att: packed uint8 (B,64,8) or an unpacked float (B,64,64) matrix
        A = self.unpack(att) if att.dtype == torch.uint8 else att
        occ_w = x[:, :6].sum(dim=1).flatten(1)              # (B, 64) 1 where a white piece stands
        occ_b = x[:, 6:12].sum(dim=1).flatten(1)
        threat, defend, quiet = relation_stack(A, occ_w, occ_b)
        R = torch.stack([threat, defend, quiet], dim=1)
        R = torch.cat([R, R.transpose(-1, -2)], dim=1)      # (B, 6, 64, 64): the three relations and their reverses
        bias = torch.einsum("brij,hr->bhij", R, self.rel_w)  # (B, heads, 64, 64)
        att_w = (A * occ_w[:, :, None]).sum(dim=1)          # how many white pieces attack each square
        att_b = (A * occ_b[:, :, None]).sum(dim=1)
        x2 = torch.cat([x, (att_w / 4).view(-1, 1, 8, 8), (att_b / 4).view(-1, 1, 8, 8)], dim=1)   # (B, 20, 8, 8)
        h = self.blocks(self.stem(x2))
        flat = self.squeeze(h).flatten(1)
        t = h.flatten(2).transpose(1, 2) + self.pos
        for layer in self.layers:
            t = layer(t, bias)
        t = self.norm(t)
        pooled = torch.cat([t.mean(dim=1), t.amax(dim=1), flat], dim=1)
        return self.head(pooled)


def make_graph():
    return GraphHybridNet()


def flip_att(A, perm):
    """Relabel the squares of an unpacked attack matrix (B,64,64): use V_PERM for the colour flip, H_PERM for left-right."""
    p = torch.as_tensor(perm, device=A.device)
    return A[:, p][:, :, p]


def make_hybrid():
    return HybridNet()


if __name__ == "__main__":
    m = make_hybrid().eval()
    x = torch.rand(5, 18, 8, 8)
    with torch.no_grad():
        y = m(x)
    print("output shape:", tuple(y.shape), "| parameters:", sum(p.numel() for p in m.parameters()))
    m.train(); m(x).sum().backward(); print("backward pass OK")
    g = make_graph().eval()
    att = torch.randint(0, 256, (5, 64, 8), dtype=torch.uint8)
    with torch.no_grad():
        yg = g(x, att)
    print("graph model output shape:", tuple(yg.shape), "| parameters:", sum(p.numel() for p in g.parameters()))
    g.train(); g(x, att).sum().backward(); print("graph backward pass OK")
    A = g.unpack(att); print("flip_att keeps shape:", tuple(flip_att(A, V_PERM).shape), "| unpack values only 0/1:", bool(((A == 0) | (A == 1)).all()))
