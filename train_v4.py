"""CNN v4: same architecture and augmentations as cnn_v3, but trained on your original training data PLUS the
extra Lichess positions from prepare_lichess.py. Validation and test sets are untouched, so results stay comparable.
Keeps the best-validation checkpoint (saved as cnn_v4.pt)."""
import os, sys, glob, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = r"C:\Work\Fen analysis"
QUICK_TEST = False            # True: 60k positions, 3 epochs.  False: full run
EPOCHS = 3 if QUICK_TEST else 40
OUT_NAME = "hyb_d"          # checkpoint name
SEED = 51                    # change the seed for every extra model you train
USE_LICHESS = False           # False = original data only
INIT_FROM = "hyb_a"              # e.g. "cnn_v4" = start from that checkpoint (fine-tuning) instead of random weights
LR = 1e-3                    # learning rate (use about 3e-4 when fine-tuning)
LICHESS_FILE = "lichess_extra_cnn.npz"   # which extra-data file to use (e.g. "lichess_extra_4m_cnn.npz")
MODEL = "hybrid"                 # "cnn" = the residual CNN used so far, "hybrid" = CNN + transformer hybrid (hybrid_model.py)
LICHESS_MAX = None            # use only this many Lichess positions (random subset), e.g. 246000 for a 50/50 mix with your original data
OVERWRITE = False             # False = refuse to overwrite an existing checkpoint with the same name
CHANNELS, BLOCKS = 64, 2
torch.manual_seed(SEED); np.random.seed(SEED)
best_path = os.path.join(DATA_DIR, OUT_NAME + ("_quick" if QUICK_TEST else "") + ".pt")     # quick tests never overwrite real checkpoints
if os.path.exists(best_path) and not OVERWRITE:
    sys.exit(f"{best_path} already exists. Change OUT_NAME (or set OVERWRITE = True) so a good model is not overwritten.")
dev = "cuda" if torch.cuda.is_available() else "cpu"      # uses the GPU automatically when PyTorch can see one
print("training on:", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU")

def load(name, as_float=True):
    z = np.load(glob.glob(os.path.join(DATA_DIR, f"*{name}_cnn.npz"))[0])
    X, y = torch.from_numpy(z["X"]), torch.from_numpy(z["y"]).float()
    return (X.float() if as_float else X), y

Xtr, ytr = load("train", as_float=False)        # stays uint8 and is converted batch by batch (saves memory)
Xva, yva = load("validation"); Xte, yte = load("test")
if USE_LICHESS:
    z = np.load(os.path.join(DATA_DIR, LICHESS_FILE))
    Xex, yex = torch.from_numpy(z["X"]), torch.from_numpy(z["y"]).float()
    if LICHESS_MAX and len(Xex) > LICHESS_MAX:
        sel = torch.randperm(len(Xex))[:LICHESS_MAX]
        Xex, yex = Xex[sel], yex[sel]
    if QUICK_TEST: Xtr, ytr, Xex, yex = Xtr[:30000], ytr[:30000], Xex[:30000], yex[:30000]
    Xtr, ytr = torch.cat([Xtr, Xex]), torch.cat([ytr, yex])
    print(f"training positions: {len(Xtr)} (original + {len(Xex)} from Lichess)")
    del Xex, yex                                # free the duplicate copy
elif QUICK_TEST:
    Xtr, ytr = Xtr[:60000], ytr[:60000]

# ---- colour-flip augmentation: swap colours and mirror the board vertically.
# Targets are side-to-move relative, so the label stays the same. ----
def color_flip(x):
    pieces = torch.cat([x[:, 6:12], x[:, 0:6]], 1).flip(2)
    stm = 1 - x[:, 12:13]
    castle = torch.cat([x[:, 15:17], x[:, 13:15]], 1)
    ep = x[:, 17:18].flip(2)
    return torch.cat([pieces, stm, castle, ep], 1)

# ---- left-right flip: valid only when nobody can castle (castling is not symmetric) ----
def h_flip(x):
    no_castle = x[:, 13:17].sum(dim=(1, 2, 3)) == 0
    return torch.where(no_castle[:, None, None, None], x.flip(3), x)

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
    p = torch.cat([model(X[i:i+2048].to(dev)).squeeze(1).cpu() for i in range(0, len(X), 2048)])
    err = p - y
    clear = y.abs() > 0.1
    return dict(MAE=round(err.abs().mean().item(), 4),
                RMSE=round(err.pow(2).mean().sqrt().item(), 4),
                R2=round(1 - (err**2).sum().item() / ((y - y.mean())**2).sum().item(), 4),
                corr=round(float(np.corrcoef(p.numpy(), y.numpy())[0, 1]), 4),
                sign_agree=round((torch.sign(p[clear]) == torch.sign(y[clear])).float().mean().item(), 4))

if MODEL == "hybrid":
    from hybrid_model import make_hybrid
    model = make_hybrid().to(dev)
else:
    model = make_model().to(dev)
if INIT_FROM:
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, INIT_FROM + ".pt"), map_location=dev))
    print("starting from", INIT_FROM)
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.MSELoss()
dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True, pin_memory=(dev == "cuda"))
best = 9.0

for ep in range(EPOCHS):
    model.train(); t0 = time.time(); run_loss, seen = 0.0, 0
    for xb, yb in dl:
        xb, yb = xb.to(dev).float(), yb.to(dev)
        flip = torch.rand(len(xb), device=dev) < 0.5
        xb = torch.where(flip[:, None, None, None], color_flip(xb), xb)
        hf = torch.rand(len(xb), device=dev) < 0.5
        xb = torch.where(hf[:, None, None, None], h_flip(xb), xb)
        opt.zero_grad()
        loss = loss_fn(model(xb).squeeze(1), yb)
        loss.backward()
        if MODEL == "hybrid": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.detach() * len(xb); seen += len(xb)
    sched.step()
    r = evaluate(model, Xva, yva)
    mark = ""
    if r["MAE"] < best:
        best = r["MAE"]; torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_path); mark = "  <- saved"
    train_rmse = (float(run_loss) / seen) ** 0.5            # compare with val RMSE: a growing gap means overfitting
    print(f"epoch {ep+1}/{EPOCHS} ({time.time()-t0:.0f}s) train RMSE {train_rmse:.4f} | val: {r}{mark}", flush=True)

model.load_state_dict(torch.load(best_path, map_location=dev))
print("TEST (best-validation checkpoint):", evaluate(model, Xte, yte))
print("for comparison, cnn_v3 (original data only): MAE 0.1117, RMSE 0.1713, corr 0.9159")
