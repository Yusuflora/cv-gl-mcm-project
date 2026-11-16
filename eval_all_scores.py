"""
One-pass evaluation of MCM, L-MCM and GL-MCM (and several temperatures).

Why: the original eval_ood_detection.py re-runs CLIP on every image once per score.
The three scores only differ in how the CLIP outputs are combined, so we run CLIP once,
store the raw global / local maximum-softmax values, and compute all scores from them.
Raw values are cached in results_raw/ so any lambda or tau analysis is free afterwards.

Outputs (same layout as the original code):
  results/<ID>/<SCORE>/CLIP_<ckpt>_T_<T>_ID_<name>/<name>.csv  and  <SCORE>_<ood>.png
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader

import clip
from utils.common import setup_seed, get_test_labels
from utils.detection_util import get_measures
from utils.file_ops import save_as_dataframe
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_model_clip, set_val_loader, set_ood_loader_ImageNet, get_subset_with_len

DEVICE = "mps"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dataset', default='COCO_single', choices=['COCO_single', 'COCO_multi', 'VOC_single', 'ImageNet'])
    p.add_argument('--root-dir', default='./datasets')
    p.add_argument('--name', default='all')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('-b', '--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--CLIP_ckpt', default='ViT-B/16', choices=['ViT-B/16', 'RN50', 'RN101'])
    p.add_argument('--Ts', default='1', help='comma separated temperatures, e.g. 1,0.01')
    p.add_argument('--lambda_local', type=float, default=None, help='default 1.0 (COCO/VOC), 0.5 (ImageNet)')
    p.add_argument('--out_list', default='', help='comma separated OOD sets')
    p.add_argument('--num_ood_sumple', type=int, default=-1, help='random subset size for each OOD set (-1 = all)')
    p.add_argument('--num_id_sample', type=int, default=-1, help='random subset size for the ID set (-1 = all)')
    p.add_argument('--burn', default='', help='OOD sets handled in an earlier invocation: only advance the random subset draws, so that later sets get the same random subsets as in a single run')
    p.add_argument('--model', default='CLIP')
    a = p.parse_args()
    a.Ts = [float(t) for t in a.Ts.split(',')]
    if a.lambda_local is None:
        a.lambda_local = 0.5 if a.in_dataset == 'ImageNet' else 1.0
    a.CLIP_ckpt_name = a.CLIP_ckpt.replace('/', '_')
    return a


def tkey(T):
    return f"{T:g}"


@torch.no_grad()
def extract(net, loader, text_feats, Ts):
    """returns {'g_<T>': array, 'l_<T>': array}: max softmax over classes (global) / over regions and classes (local)"""
    out = {}
    for T in Ts:
        out[f'g_{tkey(T)}'] = []
        out[f'l_{tkey(T)}'] = []
    for images, *_ in tqdm(loader, total=len(loader)):
        images = images.to(DEVICE)
        gf, lf = net.encode_image(images)
        gf = gf.float(); lf = lf.float()
        gf = gf / gf.norm(dim=-1, keepdim=True)
        lf = lf / lf.norm(dim=-1, keepdim=True)
        og = gf @ text_feats.T
        ol = lf @ text_feats.T
        for T in Ts:
            out[f'g_{tkey(T)}'].append(F.softmax(og / T, dim=1).max(dim=1).values.cpu().numpy())
            out[f'l_{tkey(T)}'].append(F.softmax(ol / T, dim=-1).flatten(1).max(dim=1).values.cpu().numpy())
        del images, gf, lf, og, ol
        torch.mps.empty_cache()
    return {k: np.concatenate(v) for k, v in out.items()}


def cached(path, fn):
    if os.path.exists(path):
        print(f'[cache] {path}')
        return dict(np.load(path))
    res = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **res)
    return res


def scores_from_raw(raw, T, lam):
    g, l = raw[f'g_{tkey(T)}'], raw[f'l_{tkey(T)}']
    return {'MCM': g, 'L-MCM': l, 'GL-MCM': g + lam * l}   # higher = more in-distribution


def main():
    args = parse()
    setup_seed(args.seed)
    net, preprocess = set_model_clip(args)
    net.eval()

    if args.in_dataset in ['COCO_single', 'COCO_multi']:
        out_datasets = ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_voc']
    elif args.in_dataset == 'VOC_single':
        out_datasets = ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_coco']
    else:
        out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']
    if args.out_list:
        out_datasets = args.out_list.split(',')

    test_labels = get_test_labels(args)
    text_inputs = clip.tokenize([f"a photo of a {c}" for c in test_labels]).to(DEVICE)
    with torch.no_grad():
        text_feats = net.encode_text(text_inputs).float()
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    raw_dir = os.path.join('results_raw', args.in_dataset, f'{args.model}_{args.CLIP_ckpt_name}_{args.name}')

    id_loader = set_val_loader(args, preprocess)
    ds = id_loader.dataset
    if args.num_id_sample > 0:
        ds = get_subset_with_len(ds, args.num_id_sample, shuffle=True)
    id_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    id_raw = cached(os.path.join(raw_dir, 'ID.npz'), lambda: extract(net, id_loader, text_feats, args.Ts))

    full = {'COCO_single': ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_voc'], 'COCO_multi': ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_voc'],
            'VOC_single': ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_coco'], 'ImageNet': ['iNaturalist', 'SUN', 'places365', 'Texture']}[args.in_dataset]

    # advance the numpy random state exactly as a single full run would have done (see --burn)
    for ood in (args.burn.split(',') if args.burn else []):
        set_ood_loader_ImageNet(args, ood, preprocess, root=args.root_dir)
        print(f'[burn] random subset draw advanced for {ood}')

    def all_cached():
        return {d: dict(np.load(os.path.join(raw_dir, f'{d}.npz'))) for d in full if os.path.exists(os.path.join(raw_dir, f'{d}.npz'))}

    for ood in out_datasets:
        def run(ood=ood):
            print(f'=== OOD set: {ood}')
            loader = set_ood_loader_ImageNet(args, ood, preprocess, root=args.root_dir)
            loader = DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            return extract(net, loader, text_feats, args.Ts)
        cached(os.path.join(raw_dir, f'{ood}.npz'), run)
        write_outputs(args, id_raw, all_cached())   # after every set, so partial results always exist
    print('done')


def write_outputs(args, id_raw, ood_raw):
    done = list(ood_raw.keys())
    for T in args.Ts:
        sid = scores_from_raw(id_raw, T, args.lambda_local)
        res = {s: ([], [], []) for s in sid}
        for ood in done:
            sout = scores_from_raw(ood_raw[ood], T, args.lambda_local)
            for s in sid:
                auroc, aupr, fpr = get_measures(sid[s], sout[s])
                res[s][0].append(fpr); res[s][1].append(auroc); res[s][2].append(aupr)
        for s in sid:
            d = os.path.join('results', args.in_dataset, s, f'{args.model}_{args.CLIP_ckpt_name}_T_{tkey(T)}_ID_{args.name}')
            os.makedirs(d, exist_ok=True)
            a = SimpleNamespace(log_directory=d, name=args.name, score=s)
            save_as_dataframe(a, done, *res[s])
            if tkey(T) == tkey(args.Ts[0]):
                for ood in done:
                    sout = scores_from_raw(ood_raw[ood], T, args.lambda_local)
                    plot_distribution(a, -sid[s], -sout[s], ood)   # same sign convention as the original plots
    for T in args.Ts:
        print(f'--- T={tkey(T)}, lambda={args.lambda_local}: results so far ({len(done)} OOD sets: {", ".join(done)})')


if __name__ == '__main__':
    main()
