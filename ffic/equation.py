"""UNIQUE 3-D EQUATIONS for the photo: colour = f(x, y), learned per image instead of sines and cosines.

    python equation.py IMAGE.rgb W H OUTDIR

  siren-S / siren-L   a small sine-activation network (SIREN, as in COIN): 3 or 4 layers, 48 or 96 wide
  hills-2k / hills-6k  a sum of K stretched, rotated 3-D hills (2-D Gaussians with colour, as in GaussianImage)
Parameters are stored as 16-bit numbers (their cost is counted). For each equation: fit quality (PSNR), equation bytes, and
the leftover needed to make the photo exact (zero-order estimate). Writes OUTDIR/<name>.wave (float32 R,G,B planes, the
equation's picture) + .side (equation bytes), so `ffic wmodel=` can use it as a guesser with its bytes added.
"""
import json, math, os, sys, time
import numpy as np, torch

torch.manual_seed(0); torch.set_num_threads(4)


class Siren(torch.nn.Module):
    def __init__(s, width, depth, w0=30.0):
        super().__init__(); s.w0 = w0
        dims = [2] + [width] * depth; s.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        s.out = torch.nn.Linear(width, 3)
        with torch.no_grad():
            for i, l in enumerate(s.layers):
                bound = 1 / l.in_features if i == 0 else math.sqrt(6 / l.in_features) / w0; l.weight.uniform_(-bound, bound)
    def forward(s, p):
        h = p
        for i, l in enumerate(s.layers): h = torch.sin(s.w0 * l(h))
        return s.out(h)


class Hills(torch.nn.Module):
    def __init__(s, K, img_t, H, W):
        super().__init__()
        idx = torch.randint(0, H * W, (K,)); s.mu = torch.nn.Parameter(torch.stack([(idx % W).float() / W * 2 - 1, (idx // W).float() / H * 2 - 1], 1))
        s.logs = torch.nn.Parameter(torch.full((K, 2), math.log(2.5 / math.sqrt(K / (H * W)) / max(H, W))))
        s.rot = torch.nn.Parameter(torch.zeros(K)); s.col = torch.nn.Parameter(img_t[idx].clone() * 0.5)
        s.base = torch.nn.Parameter(img_t.mean(0).clone())
    def forward(s, p, chunk=4096):
        out = []
        c, sn = torch.cos(s.rot), torch.sin(s.rot); inv = torch.exp(-s.logs)
        for i in range(0, len(p), chunk):
            d = p[i:i + chunk, None, :] - s.mu[None]                     # (n, K, 2)
            u = (d[..., 0] * c + d[..., 1] * sn) * inv[:, 0]; v = (-d[..., 0] * sn + d[..., 1] * c) * inv[:, 1]
            w = torch.exp(-0.5 * (u * u + v * v)); out.append(s.base + w @ s.col)
        return torch.cat(out)


def ent(r):
    v, c = np.unique(r, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())


def fit(model, coords, target, steps, batch, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for t in range(steps):
        i = torch.randint(0, len(coords), (batch,)); loss = ((model(coords[i]) - target[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return torch.cat([model(coords[i:i + 65536]) for i in range(0, len(coords), 65536)]).numpy()


if __name__ == "__main__":
    f, W, H, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; os.makedirs(OUT, exist_ok=True)
    img = np.fromfile(f, np.uint8).reshape(H, W, 3).astype(np.float32)
    ys, xs = np.mgrid[0:H, 0:W]; coords = torch.tensor(np.stack([xs.ravel() / (W - 1) * 2 - 1, ys.ravel() / (H - 1) * 2 - 1], 1), dtype=torch.float32)
    target = torch.tensor(img.reshape(-1, 3) / 255.0)
    runs = [("siren-S", lambda: Siren(48, 3), 3000, 32768, 1e-3), ("siren-L", lambda: Siren(96, 4), 3000, 32768, 5e-4),
            ("hills-2k", lambda: Hills(2000, target, H, W), 1500, 8192, 5e-3), ("hills-6k", lambda: Hills(6000, target, H, W), 1500, 4096, 5e-3)]
    res = {}
    for name, mk, steps, batch, lr in runs:
        t0 = time.time(); m = mk(); npar = sum(p.numel() for p in m.parameters())
        for p in m.parameters(): p.data = p.data.half().float()                      # start from 16-bit-representable values
        pred = fit(m, coords, target, steps, batch, lr)
        with torch.no_grad():                                                        # the stored equation is 16-bit
            for p in m.parameters(): p.data = p.data.half().float()
            pred = torch.cat([m(coords[i:i + 65536]) for i in range(0, len(coords), 65536)]).numpy()
        pic = np.clip(pred * 255, 0, 255).reshape(H, W, 3)
        mse = float(((pic - img) ** 2).mean()); psnr = 10 * math.log10(255 ** 2 / mse)
        left = (img - np.rint(pic)).astype(int); left_bits = sum(ent(left[..., c].ravel()) for c in range(3)) * H * W
        side = npar * 2
        pic.transpose(2, 0, 1).astype(np.float32).tofile(f"{OUT}/{name}.wave"); open(f"{OUT}/{name}.wave.side", "w").write(str(side))
        res[name] = {"params": npar, "equation_bytes": side, "psnr": round(psnr, 2), "leftover_bytes": round(left_bits / 8), "total_bytes": round(side + left_bits / 8), "fit_s": round(time.time() - t0)}
        print(f"{name:9} {npar:7,} numbers = {side:8,} B equation | fit {psnr:5.2f} dB | leftover {left_bits / 8:9,.0f} B | equation + leftover {side + left_bits / 8:9,.0f} B  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(f"{OUT}/equation.json", "w"), indent=1)
