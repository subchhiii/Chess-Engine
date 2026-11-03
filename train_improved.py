"""Improved CNN evaluator: residual blocks + BatchNorm, colour-flip augmentation,
AdamW + cosine LR schedule, keeps the best-validation checkpoint (saved as cnn_v2.pt)."""
import os, glob, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = r"C:\Work\Fen analysis"
QUICK_TEST = False            # True: 60k positions, 3 epochs.  False: full run
EPOCHS = 3 if QUICK_TEST else 15
CHANNELS, BLOCKS = 64, 2
torch.manual_seed(42); np.random.seed(42)

def load(name):
    z = np.load(glob.glob(os.path.join(DATA_DIR, f"*{name}_cnn.npz"))[0])
    return torch.from_numpy(z["X"]).float(), torch.from_numpy(z["y"]).float()
Xtr, ytr = load("train"); Xva, yva = load("validation"); Xte, yte = load("test")
if QUICK_TEST: Xtr, ytr = Xtr[:60000], ytr[:60000]

# ---- colour-flip augmentation: swap colours and mirror the board vertically.
# Targets are side-to-move relative, so the label stays the same. ----
def color_flip(x):
    pieces = torch.cat([x[:, 6:12], x[:, 0:6]], 1).flip(2)
    stm = 1 - x[:, 12:13]
    castle = torch.cat([x[:, 15:17], x[:, 13:15]], 1)
    ep = x[:, 17:18].flip(2)
    return torch.cat([pieces, stm, castle, ep], 1)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                               nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
    def forward(self, x): return torch.relu(x + self.f(x))

def make_model():
    return nn.Sequential(
        nn.Conv2d(18, CHANNELS, 3, padding=1, bias=False), nn.BatchNorm2d(CHANNELS), nn.ReLU(),
        *[ResBlock(CHANNELS) for _ in range(BLOCKS)],
        nn.Conv2d(CHANNELS, 8, 1), nn.ReLU(), nn.Flatten(),
        nn.Linear(8*64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    p = torch.cat([model(X[i:i+2048]).squeeze(1) for i in range(0, len(X), 2048)])
    err = p - y
    clear = y.abs() > 0.1
    return dict(MAE=round(err.abs().mean().item(), 4),
                RMSE=round(err.pow(2).mean().sqrt().item(), 4),
                R2=round(1 - (err**2).sum().item() / ((y - y.mean())**2).sum().item(), 4),
                corr=round(float(np.corrcoef(p.numpy(), y.numpy())[0, 1]), 4),
                sign_agree=round((torch.sign(p[clear]) == torch.sign(y[clear])).float().mean().item(), 4))

model = make_model()
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.MSELoss()
dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
best, best_path = 9.0, os.path.join(DATA_DIR, "cnn_v2.pt")

for ep in range(EPOCHS):
    model.train(); t0 = time.time()
    for xb, yb in dl:
        flip = torch.rand(len(xb)) < 0.5
        xb = torch.where(flip[:, None, None, None], color_flip(xb), xb)
        opt.zero_grad()
        loss_fn(model(xb).squeeze(1), yb).backward()
        opt.step()
    sched.step()
    r = evaluate(model, Xva, yva)
    mark = ""
    if r["MAE"] < best:
        best = r["MAE"]; torch.save(model.state_dict(), best_path); mark = "  <- saved"
    print(f"epoch {ep+1}/{EPOCHS} ({time.time()-t0:.0f}s) val: {r}{mark}", flush=True)

model.load_state_dict(torch.load(best_path))
print("TEST (best-validation checkpoint):", evaluate(model, Xte, yte))
print("previous CNN for comparison: MAE 0.1319, RMSE 0.1936, corr 0.8933")
