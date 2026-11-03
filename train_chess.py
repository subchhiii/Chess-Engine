import os, glob, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = r"C:\Work\Fen analysis"     # raw string r"..." keeps the backslashes and space safe
QUICK_TEST = False                      # set to False for the full run
EPOCHS = 3 if QUICK_TEST else 10

def load(name):
    path = glob.glob(os.path.join(DATA_DIR, f"*{name}_cnn.npz"))[0]
    z = np.load(path)
    return torch.from_numpy(z["X"]).float(), torch.from_numpy(z["y"]).float()

Xtr, ytr = load("train"); Xva, yva = load("validation"); Xte, yte = load("test")
if QUICK_TEST:
    Xtr, ytr = Xtr[:60000], ytr[:60000]
print("train size:", len(Xtr))

dev = "cuda" if torch.cuda.is_available() else "cpu"

# Baseline 1: material count (no learning)
vals = torch.tensor([1,3,3,5,9,0, -1,-3,-3,-5,-9,0]).float()
def material(X):
    m = (X[:, :12].sum(dim=(2,3)) * vals).sum(1) / 40
    return m * torch.where(X[:, 12, 0, 0] == 1, 1.0, -1.0)

cnn = nn.Sequential(
    nn.Conv2d(18, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
    nn.Flatten(), nn.Linear(64*8*8, 128), nn.ReLU(),
    nn.Linear(128, 1), nn.Tanh(),
).to(dev)

mlp = nn.Sequential(nn.Flatten(), nn.Linear(18*64, 256), nn.ReLU(),
                    nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh()).to(dev)

@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    p = torch.cat([model(X[i:i+2048].to(dev)).squeeze(1).cpu() for i in range(0, len(X), 2048)])
    mae = (p-y).abs().mean().item(); rmse = ((p-y)**2).mean().sqrt().item()
    corr = np.corrcoef(p.numpy(), y.numpy())[0,1]
    return dict(MAE=round(mae,4), RMSE=round(rmse,4), corr=round(float(corr),4))

def train(model, epochs):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb).squeeze(1), yb).backward()
            opt.step()
        print(f"epoch {ep+1}/{epochs}  val:", evaluate(model, Xva, yva), flush=True)

print("baseline (predict median)  MAE ~0.326, RMSE ~0.426")
print("material corr:", round(float(np.corrcoef(material(Xva).numpy(), yva.numpy())[0,1]), 4))
print("--- MLP"); train(mlp, EPOCHS)
print("--- CNN"); train(cnn, EPOCHS)
if not QUICK_TEST:
    print("TEST mlp:", evaluate(mlp, Xte, yte))
    print("TEST cnn:", evaluate(cnn, Xte, yte))
    torch.save(cnn.state_dict(), os.path.join(DATA_DIR, "cnn.pt"))
    torch.save(mlp.state_dict(), os.path.join(DATA_DIR, "mlp.pt"))