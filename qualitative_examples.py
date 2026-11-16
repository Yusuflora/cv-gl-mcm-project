"""
Qualitative figures: real OOD images where adding local features changes the decision.

  fig12: OOD images that MCM wrongly accepts as in-distribution but GL-MCM correctly rejects
  fig13: the opposite case (GL-MCM wrongly accepts, MCM rejects)

Setting: ID = COCO_single (60 classes), OOD = a random sample of one OOD folder (default SUN), CLIP ViT-B/16, tau = 1.
Decision rule = the usual FPR95 operating point: the threshold keeps 95% of the ID images (computed from results_raw/.../ID.npz).
The coloured overlay is the LOCAL score map (14x14 patches): for every patch, the maximum class probability.
Run (about 2 minutes on a Mac):  PYTORCH_ENABLE_MPS_FALLBACK=1 python qualitative_examples.py
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace
from PIL import Image
from torchvision import datasets, transforms

import clip
from utils.common import get_test_labels
from utils.train_eval_util import set_model_clip

p = argparse.ArgumentParser()
p.add_argument("--ood", default="SUN", choices=["SUN", "iNaturalist", "Texture", "IN22k"])
p.add_argument("--n", type=int, default=600, help="number of random OOD images to score")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--k", type=int, default=8, help="examples per figure")
a = p.parse_args()

DEVICE = "mps"
FOLDER = {"SUN": "SUN", "iNaturalist": "iNaturalist", "Texture": os.path.join("Texture", "images"), "IN22k": "ImageNet-22K"}[a.ood]
args = SimpleNamespace(CLIP_ckpt="ViT-B/16", model="CLIP", in_dataset="COCO_single")
net, preprocess = set_model_clip(args)
net.eval()
labels = list(get_test_labels(args))
with torch.no_grad():
    tf = clip.tokenize([f"a photo of a {c}" for c in labels]).to(DEVICE)
    text = net.encode_text(tf).float()
    text = text / text.norm(dim=-1, keepdim=True)

# thresholds that keep 95% of the ID images (same operating point as FPR95)
idr = np.load("results_raw/COCO_single/CLIP_ViT-B_16_final/ID.npz")
thr_m = np.percentile(idr["g_1"], 5)
thr_gl = np.percentile(idr["g_1"] + idr["l_1"], 5)

ds = datasets.ImageFolder(os.path.join("datasets", FOLDER))
idx = np.random.RandomState(a.seed).permutation(len(ds))[: a.n]
show = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
rows = []
with torch.no_grad():
    for i in idx:
        path = ds.samples[i][0]
        img = Image.open(path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(DEVICE)
        gf, lf = net.encode_image(x)
        gf = gf.float(); lf = lf.float()
        gf = gf / gf.norm(dim=-1, keepdim=True); lf = lf / lf.norm(dim=-1, keepdim=True)
        pg = F.softmax(gf @ text.T, dim=1)[0]
        pl = F.softmax(lf @ text.T, dim=-1)[0]                 # 14 x 14 x classes
        lmap = pl.max(dim=-1).values.cpu().numpy()             # 14 x 14 local score map
        g, l = pg.max().item(), float(lmap.max())
        # class with the highest local probability anywhere in the image
        flat = pl.reshape(-1, pl.shape[-1]); best = flat.max(dim=1).values.argmax().item()
        rows.append(dict(path=path, img=show(img), g=g, l=l, gl=g + l, lmap=lmap,
                         gcls=labels[pg.argmax().item()], lcls=labels[flat[best].argmax().item()]))
print(f"scored {len(rows)} {a.ood} images; ID thresholds: MCM {thr_m:.4f}, GL-MCM {thr_gl:.4f}")

fooled = [r for r in rows if r["g"] >= thr_m and r["gl"] < thr_gl]      # MCM accepts, GL-MCM rejects  (GL-MCM correct)
broken = [r for r in rows if r["g"] < thr_m and r["gl"] >= thr_gl]      # MCM rejects, GL-MCM accepts  (GL-MCM wrong)
fooled.sort(key=lambda r: (r["gl"] - thr_gl))                           # most clearly rejected by GL-MCM first
broken.sort(key=lambda r: -(r["gl"] - thr_gl))
print(f"MCM accepts & GL-MCM rejects: {len(fooled)} images; MCM rejects & GL-MCM accepts: {len(broken)} images")
os.makedirs("figures", exist_ok=True)


def draw(sel, title, fname):
    if not sel:
        print("no examples for", fname); return
    cols = 4; nrows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.6 * cols, 4.6 * nrows), squeeze=False)
    for ax in axes.ravel(): ax.axis("off")
    for ax, r in zip(axes.ravel(), sel):
        ax.imshow(r["img"])
        ax.imshow(r["lmap"], cmap="inferno", alpha=0.55, extent=(0, 224, 224, 0), interpolation="bilinear", vmin=float(r["lmap"].min()), vmax=float(r["lmap"].max()))
        ax.set_title(f"MCM {r['g']:.4f} ({'accept' if r['g'] >= thr_m else 'reject'}, thr {thr_m:.4f})\n"
                     f"GL-MCM {r['gl']:.4f} ({'accept' if r['gl'] >= thr_gl else 'reject'}, thr {thr_gl:.4f})\n"
                     f"global: {r['gcls']} | local: {r['lcls']}", fontsize=7.5)
    fig.suptitle(title, y=1.0, fontsize=11)
    fig.tight_layout(h_pad=3.0); fig.savefig(f"figures/{fname}", bbox_inches="tight", dpi=130); plt.close(fig)


draw(fooled[: a.k], f"OOD images ({a.ood}) wrongly ACCEPTED by MCM but correctly REJECTED by GL-MCM (95% of ID kept).\n"
     "Overlay = local score map (colour scale relative within each image; bright = best-matching patch)", f"fig12_examples_mcm_fooled_glmcm_correct_{a.ood}.png")
draw(broken[: a.k], f"OOD images ({a.ood}) correctly rejected by MCM but wrongly ACCEPTED by GL-MCM (95% of ID kept).\n"
     "Overlay = local score map (colour scale relative within each image; bright = best-matching patch)", f"fig13_examples_glmcm_fooled_mcm_correct_{a.ood}.png")
print("done")
