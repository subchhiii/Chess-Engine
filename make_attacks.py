"""Pre-compute the attack matrices for every position in your data (about 10-20 minutes for the Lichess set).
Creates train_att.npy, validation_att.npy, test_att.npy and lichess_extra_att.npy next to your data.
Run once:  python make_attacks.py"""
import os, glob, numpy as np
from multiprocessing import Pool
from attack_graph import attacks_from_tensor

DATA_DIR = r"C:\Work\Fen analysis"
LICHESS_FILE = "lichess_extra_cnn.npz"
WORKERS = 6                                    # CPU processes to use

def work(chunk):
    return attacks_from_tensor(chunk)

def convert(npz_path, out_name):
    out = os.path.join(DATA_DIR, out_name)
    if os.path.exists(out):
        print("exists, skipping:", out); return
    X = np.load(npz_path)["X"]
    chunks = [X[i:i + 4000] for i in range(0, len(X), 4000)]
    with Pool(WORKERS) as pool:
        parts = []
        for k, part in enumerate(pool.imap(work, chunks)):
            parts.append(part)
            if (k + 1) % 50 == 0: print(f"  {out_name}: {(k + 1) * 4000:,} / {len(X):,}", flush=True)
    A = np.concatenate(parts)
    np.save(out, A)
    print(f"saved {out}: {A.shape}")

if __name__ == "__main__":
    for name in ("validation", "test", "train"):
        convert(glob.glob(os.path.join(DATA_DIR, f"*{name}_cnn.npz"))[0], f"{name}_att.npy")
    convert(os.path.join(DATA_DIR, LICHESS_FILE), "lichess_extra_att.npy")
