"""
Post-hoc OOD baselines from the article's ImageNet table: MSP, Energy, ODIN.
They use an ImageNet-1K supervised ViT-B/16 (torchvision weights) instead of CLIP.
One pass computes all three scores; raw scores are cached in results_raw/ImageNet/ViT-B_16_supervised/.

  MSP    = max softmax(logits)                       (Hendrycks & Gimpel)
  Energy = logsumexp(logits)  (T=1)                  (Liu et al.)
  ODIN   = max softmax(logits/T) on a perturbed input (Liang et al.), T=1000, eps=0.0014

ID = ImageNet-1K validation (datasets/ImageNet). OOD = iNaturalist, SUN, Places, Texture (same folders as the CLIP runs).
Higher score = more in-distribution, same convention as eval_all_scores.py.
ViM and KNN need ImageNet *training* features and are therefore not implemented.
"""
import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

from utils.detection_util import get_measures
from utils.file_ops import save_as_dataframe

DEVICE = "mps"
OOD = {'iNaturalist': 'iNaturalist', 'SUN': 'SUN', 'places365': 'Places', 'Texture': os.path.join('Texture', 'images')}
TAG = 'ViT-B_16_supervised'


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--root-dir', default='./datasets')
    p.add_argument('--name', default='final')
    p.add_argument('-b', '--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--num_id_sample', type=int, default=-1, help='random subset of the ID set (-1 = all 50000)')
    p.add_argument('--num_ood_sumple', type=int, default=-1, help='random subset of each OOD set (-1 = all)')
    p.add_argument('--odin_T', type=float, default=1000.0)
    p.add_argument('--odin_eps', type=float, default=0.0014)
    p.add_argument('--skip_odin', action='store_true', help='compute only MSP and Energy (much faster)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--arch', default='tv', choices=['tv', 'timm'],
                   help="tv = torchvision ViT-B/16 (ImageNet-1K only, strong regularisation); "
                        "timm = ViT-B/16 pre-trained on ImageNet-21K and fine-tuned on ImageNet-1K (closer to the model used by the cited baselines)")
    a = p.parse_args()
    a.tag = TAG if a.arch == 'tv' else 'ViT-B_16_in21k_ft_in1k'
    return a


def make_loader(root, tf, k, seed, bs, workers):
    ds = datasets.ImageFolder(root, transform=tf)
    if k > 0 and k < len(ds):
        idx = np.random.RandomState(seed).permutation(len(ds))[:k]
        ds = Subset(ds, idx.tolist())
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)


def odin_score(model, x, std, T, eps):
    x = x.clone().requires_grad_(True)
    logits = model(x) / T
    pred = logits.argmax(dim=1)
    loss = F.cross_entropy(logits, pred)
    loss.backward()
    grad = torch.sign(x.grad)
    grad = grad / std.view(1, 3, 1, 1)          # same normalisation as the original ODIN code
    x_pert = (x - eps * grad).detach()
    with torch.no_grad():
        out = model(x_pert) / T
        return F.softmax(out, dim=1).max(dim=1).values


def extract(model, loader, std, args):
    msp, energy, odin = [], [], []
    for images, *_ in tqdm(loader, total=len(loader)):
        images = images.to(DEVICE)
        with torch.no_grad():
            logits = model(images).float()
            msp.append(F.softmax(logits, dim=1).max(dim=1).values.cpu().numpy())
            energy.append(torch.logsumexp(logits, dim=1).cpu().numpy())
        if not args.skip_odin:
            odin.append(odin_score(model, images, std, args.odin_T, args.odin_eps).cpu().numpy())
        del images
        torch.mps.empty_cache()
    out = {'MSP': np.concatenate(msp), 'Energy': np.concatenate(energy)}
    if odin:
        out['ODIN'] = np.concatenate(odin)
    return out


def cached(path, fn):
    if os.path.exists(path):
        print(f'[cache] {path}')
        return dict(np.load(path))
    res = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **res)
    return res


def write_outputs(args, id_raw, ood_raw):
    done = list(ood_raw.keys())
    for s in id_raw:
        fprs, aurocs, auprs = [], [], []
        for ood in done:
            auroc, aupr, fpr = get_measures(id_raw[s], ood_raw[ood][s])
            fprs.append(fpr); aurocs.append(auroc); auprs.append(aupr)
        d = os.path.join('results', 'ImageNet', s, f'{args.tag}_ID_{args.name}')
        os.makedirs(d, exist_ok=True)
        save_as_dataframe(SimpleNamespace(log_directory=d, name=args.name, score=s), done, fprs, aurocs, auprs)
        print(f'--- {s}: FPR95 {100*np.mean(fprs):.2f}  AUROC {100*np.mean(aurocs):.2f}  ({len(done)} OOD sets)')


def main():
    args = parse()
    torch.manual_seed(args.seed)
    if args.arch == 'tv':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights).to(DEVICE).eval()
        tf = weights.transforms()
        std = torch.tensor(tf.std, device=DEVICE)
    else:
        import timm
        model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True).to(DEVICE).eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        tf = timm.data.create_transform(**cfg)
        std = torch.tensor(cfg['std'], device=DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)

    raw_dir = os.path.join('results_raw', 'ImageNet', f'{args.tag}_{args.name}')
    id_loader = make_loader(os.path.join(args.root_dir, 'ImageNet'), tf, args.num_id_sample, args.seed, args.batch_size, args.workers)
    id_raw = cached(os.path.join(raw_dir, 'ID.npz'), lambda: extract(model, id_loader, std, args))

    ood_raw = {}
    for ood, folder in OOD.items():
        def run(folder=folder, ood=ood):
            print(f'=== OOD set: {ood}')
            ld = make_loader(os.path.join(args.root_dir, folder), tf, args.num_ood_sumple, args.seed, args.batch_size, args.workers)
            return extract(model, ld, std, args)
        ood_raw[ood] = cached(os.path.join(raw_dir, f'{ood}.npz'), run)
        write_outputs(args, id_raw, ood_raw)
    print('done')


if __name__ == '__main__':
    main()
