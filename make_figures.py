"""
Figure generator for the GL-MCM midterm project.

Inputs : results/**/<run>.csv            (official metrics written by eval_all_scores.py / baselines_imagenet.py)
         results_raw/**/*.npz            (raw per-image scores; small files, no datasets needed)
Outputs: figures/*.png                   (see docs/DOSYA_VE_GORSEL_REHBERI.md for what each figure shows)
Needs  : numpy + matplotlib only.   Usage: python make_figures.py   (add --check to only verify the metric code)

All metrics are recomputed from the raw scores with the same definitions as utils/detection_util.py
(ID = positive class, higher score = more in-distribution, FPR95 = FPR at 95% TPR) and the script
asserts that they reproduce the CSV numbers before drawing anything.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from make_tables import ART, ART_IMAGENET

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
COL = {"MCM": "#2a78d6", "L-MCM": "#eb6834", "GL-MCM": "#1baf7a"}     # one fixed colour per method in every figure
OODCOL = ["#6250d6", "#e87ba4", "#e34948", "#008300", "#c98500"]       # OOD sets (line colours)
LAMBDA = {"COCO_single": 1.0, "VOC_single": 1.0, "ImageNet": 0.5}      # lambda used in the article
IDS = [("COCO_single", "COCO (ID)"), ("VOC_single", "VOC (ID)"), ("ImageNet", "ImageNet-1K (ID)")]
OOD = {"COCO_single": ["iNaturalist", "SUN", "Texture", "IN22k", "ood_voc"],
       "VOC_single": ["iNaturalist", "SUN", "Texture", "IN22k", "ood_coco"],
       "ImageNet": ["iNaturalist", "SUN", "places365", "Texture"]}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "figure.dpi": 130})


# ---------------------------------------------------------------- metrics (numpy re-implementation)
def _curve(pos, neg):
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(s, kind="mergesort")[::-1]; s, y = s[o], y[o]
    idx = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[idx]; fps = 1 + idx - tps
    return tps, fps, len(pos), len(neg)


def measures(pos, neg, recall_level=0.95):
    """returns (FPR95, AUROC, AUPR) as fractions; same definitions as utils/detection_util.get_measures"""
    tps, fps, P, N = _curve(pos, neg)
    tpr, fpr = tps / P, fps / N
    auroc = np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]) if hasattr(np, "trapezoid") else np.trapz(np.r_[0, tpr], np.r_[0, fpr])
    prec = tps / (tps + fps)
    aupr = np.sum(np.diff(np.r_[0, tpr]) * prec)
    last = tps.searchsorted(tps[-1]); sl = slice(last, None, -1)
    rec = np.r_[tpr[sl], 1]; f = np.r_[fps[sl], 0]
    cutoff = np.argmin(np.abs(rec - recall_level))
    return f[cutoff] / N, auroc, aupr


def roc_pr(pos, neg):
    tps, fps, P, N = _curve(pos, neg)
    tpr, fpr = np.r_[0, tps / P], np.r_[0, fps / N]
    prec = np.r_[1, tps / (tps + fps)]
    return fpr, tpr, np.r_[0, tps / P], prec


# ---------------------------------------------------------------- data access
def raw(id_name, name, tag="CLIP_ViT-B_16_final"):
    d = f"results_raw/{id_name}/{tag}"
    return np.load(f"{d}/{name}.npz")


def scores(r, T, lam):
    g, l = r[f"g_{T}"], r[f"l_{T}"]
    return {"MCM": g, "L-MCM": l, "GL-MCM": g + lam * l}


def csv_rows(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[list(r.values())[0]] = (float(r["FPR95"]), float(r["AUROC"]), float(r["AUPR"]))
    return rows


def clip_csv(id_name, score, T="1"):
    return csv_rows(f"results/{id_name}/{score}/CLIP_ViT-B_16_T_{T}_ID_final/final.csv")


def all_metrics(id_name, T="1", lam=None):
    lam = LAMBDA[id_name] if lam is None else lam
    idr = scores(raw(id_name, "ID"), T, lam)
    out = {s: {} for s in idr}
    for o in OOD[id_name]:
        orr = scores(raw(id_name, o), T, lam)
        for s in idr:
            out[s][o] = np.array(measures(idr[s], orr[s])) * 100
    for s in out:
        out[s]["AVG"] = np.mean([out[s][o] for o in OOD[id_name]], axis=0)
    return out


def check():
    """The recomputed metrics must reproduce the CSV files written by the official code."""
    worst = 0
    for idn, _ in IDS:
        for T in ["1", "0.01"]:
            m = all_metrics(idn, T)
            for s in m:
                c = clip_csv(idn, s, T)
                for k in c:
                    worst = max(worst, np.max(np.abs(m[s][k] - np.array(c[k]))))
    print(f"max |recomputed - csv| over all FPR95/AUROC/AUPR cells = {worst:.3f} (percentage points)")
    assert worst < 0.1, "metric re-implementation does not match the official CSVs"


# ---------------------------------------------------------------- figures
def fig01_local_feature():
    """Concept 1: local feature absent (MCM) vs present (GL-MCM); L-MCM (local only) as extra reference."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    for j, (idn, title) in enumerate(IDS):
        data = {s: clip_csv(idn, s) for s in COL}
        names = OOD[idn] + ["AVG"]
        x = np.arange(len(names)); w = 0.26
        for i, (mi, mname) in enumerate([(0, "FPR95 (lower is better)"), (1, "AUROC (higher is better)")]):
            ax = axes[i, j]
            for k, s in enumerate(COL):
                ax.bar(x + (k - 1) * w, [data[s][n][mi] for n in names], w, color=COL[s], label=s)
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
            ax.set_ylabel(mname if j == 0 else "")
            if i == 0: ax.set_title(title)
            if mi == 1: ax.set_ylim(50, 100)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Without local features (MCM) vs with local features (GL-MCM); L-MCM = local only", y=1.02)
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT}/fig01_local_feature_MCM_vs_GLMCM.png", bbox_inches="tight"); plt.close(fig)


def fig02_before_after():
    """Headline before/after: MCM (no local features) -> GL-MCM (with local features), average over OOD sets."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, mi, title, fmt in [(axes[0], 0, "FPR95 (lower is better)", "{:.1f}"), (axes[1], 1, "AUROC (higher is better)", "{:.1f}")]:
        x = np.arange(len(IDS)); w = 0.34
        b = [clip_csv(i, "MCM")["AVG"][mi] for i, _ in IDS]
        a = [clip_csv(i, "GL-MCM")["AVG"][mi] for i, _ in IDS]
        r1 = ax.bar(x - w / 2, b, w, color=COL["MCM"], label="Before: MCM (no local features)")
        r2 = ax.bar(x + w / 2, a, w, color=COL["GL-MCM"], label="After: GL-MCM (with local features)")
        for r, v in list(zip(r1, b)) + list(zip(r2, a)):
            ax.text(r.get_x() + r.get_width() / 2, v + (1 if mi == 0 else 0.3), fmt.format(v), ha="center", fontsize=9)
        for k in range(len(IDS)):
            d = a[k] - b[k]
            ax.text(x[k], max(a[k], b[k]) + (7 if mi == 0 else 1.8), f"{d:+.1f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([t for _, t in IDS]); ax.set_title(title)
        if mi == 1: ax.set_ylim(80, 98)
        if mi == 0: ax.set_ylim(0, 85)
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("Effect of adding local features (averages over the OOD sets; bold = change)", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig02_before_after_summary.png", bbox_inches="tight"); plt.close(fig)


def fig03_temperature():
    """Concept 2: temperature scaling absent (tau=1) vs present (tau=0.01)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    for j, (idn, title) in enumerate(IDS):
        for i, (mi, mname) in enumerate([(0, "FPR95 (lower is better)"), (1, "AUROC (higher is better)")]):
            ax = axes[i, j]; x = np.arange(3); w = 0.36
            for k, s in enumerate(COL):
                v1 = clip_csv(idn, s, "1")["AVG"][mi]; v2 = clip_csv(idn, s, "0.01")["AVG"][mi]
                ax.bar(x[k] - w / 2, v1, w, color=COL[s])
                ax.bar(x[k] + w / 2, v2, w, color=COL[s], hatch="///", edgecolor="white", alpha=0.75)
                ax.text(x[k] - w / 2, v1 + 0.6, f"{v1:.1f}", ha="center", fontsize=8)
                ax.text(x[k] + w / 2, v2 + 0.6, f"{v2:.1f}", ha="center", fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(list(COL)); ax.set_ylabel(mname if j == 0 else "")
            if i == 0: ax.set_title(title)
            if mi == 1: ax.set_ylim(60, 100)
            else: ax.set_ylim(0, 100)
    fig.legend(handles=[Patch(facecolor="#888", label="tau = 1 (no temperature scaling)"),
                        Patch(facecolor="#888", hatch="///", edgecolor="white", alpha=0.75, label="tau = 0.01 (temperature scaling)")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Effect of the softmax temperature tau (averages over the OOD sets)", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/fig03_temperature_tau1_vs_tau001.png", bbox_inches="tight"); plt.close(fig)


def fig04_lambda():
    """Extra analysis: weight lambda of the local score (lambda=0 is exactly MCM)."""
    lams = np.r_[0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    md = ["### Effect of lambda on GL-MCM (tau = 1, averages over the OOD sets; lambda = 0 is exactly MCM)\n",
          "| lambda | " + " | ".join(f"{t} FPR95 / AUROC" for _, t in IDS) + " |", "|---|---|---|---|"]
    allres = {}
    for j, (idn, title) in enumerate(IDS):
        res = np.array([all_metrics(idn, "1", lam=l)["GL-MCM"]["AVG"] for l in lams])
        allres[idn] = res
        for i, (mi, mname) in enumerate([(0, "FPR95 (lower is better)"), (1, "AUROC (higher is better)")]):
            ax = axes[i, j]
            ax.plot(lams, res[:, mi], marker="o", color=COL["GL-MCM"], lw=2)
            ax.axvline(LAMBDA[idn], color="#555", ls="--", lw=1)
            ax.scatter([0], [res[0, mi]], color=COL["MCM"], zorder=5, s=60)
            ax.set_xlabel("lambda (weight of the local score)" if i == 1 else ""); ax.set_ylabel(mname if j == 0 else "")
            if i == 0: ax.set_title(title)
    fig.legend(handles=[Line2D([0], [0], color=COL["GL-MCM"], marker="o", lw=2, label="GL-MCM at different lambda"),
                        Line2D([0], [0], color=COL["MCM"], marker="o", ls="", ms=8, label="lambda = 0 (this is exactly MCM)"),
                        Line2D([0], [0], color="#555", ls="--", label="lambda used in the article")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("GL-MCM = MCM + lambda * L-MCM: effect of lambda (tau = 1, averages over the OOD sets)", y=1.03)
    for k, l in enumerate(lams):
        md.append(f"| {l:g} | " + " | ".join(f"{allres[i][k][0]:.2f} / {allres[i][k][1]:.2f}" for i, _ in IDS) + " |")
    os.makedirs("tables", exist_ok=True)
    open("tables/lambda_sweep.md", "w").write("\n".join(md) + "\n")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{OUT}/fig04_lambda_sweep.png", bbox_inches="tight"); plt.close(fig)


def fig05_roc_pr():
    for idn, title in IDS:
        for kind in ["roc", "pr"]:
            names = OOD[idn]
            fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.5), sharey=True)
            idr = scores(raw(idn, "ID"), "1", LAMBDA[idn])
            for ax, o in zip(axes, names):
                orr = scores(raw(idn, o), "1", LAMBDA[idn])
                for s in COL:
                    fpr, tpr, rec, prec = roc_pr(idr[s], orr[s])
                    if kind == "roc": ax.plot(fpr, tpr, color=COL[s], lw=1.8, label=s)
                    else: ax.plot(rec, prec, color=COL[s], lw=1.8, label=s)
                if kind == "roc":
                    ax.plot([0, 1], [0, 1], color="#aaa", lw=0.8, ls=":"); ax.axhline(0.95, color="#aaa", lw=0.8, ls="--")
                    ax.set_xlabel("False positive rate (OOD accepted as ID)")
                else:
                    ax.set_xlabel("Recall (ID found)")
                ax.set_title(o)
            axes[0].set_ylabel("True positive rate (ID found)" if kind == "roc" else "Precision")
            axes[0].legend(frameon=False, loc="lower right" if kind == "roc" else "lower left", fontsize=8)
            nm = "ROC curves" if kind == "roc" else "Precision-Recall curves"
            extra = "  (dashed line = 95% TPR, where FPR95 is read)" if kind == "roc" else "  (ID is the majority class, so precision stays high)"
            fig.suptitle(f"{nm}, {title}, tau = 1{extra}", y=1.03)
            fig.tight_layout(); fig.savefig(f"{OUT}/fig05_{kind}_curves_{idn}.png", bbox_inches="tight"); plt.close(fig)


def fig06_histograms():
    for idn, title in IDS:
        idr = scores(raw(idn, "ID"), "1", LAMBDA[idn])
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        for ax, s in zip(axes, COL):
            lo = min(idr[s].min(), *[scores(raw(idn, o), "1", LAMBDA[idn])[s].min() for o in OOD[idn]])
            hi = max(idr[s].max(), *[scores(raw(idn, o), "1", LAMBDA[idn])[s].max() for o in OOD[idn]])
            bins = np.linspace(lo, hi, 40)
            ax.hist(idr[s], bins=bins, density=True, color="#9aa5b1", alpha=0.65, label="ID")
            for c, o in zip(OODCOL, OOD[idn]):
                ax.hist(scores(raw(idn, o), "1", LAMBDA[idn])[s], bins=bins, density=True, histtype="step", color=c, lw=1.6, label=o)
            ax.set_title(s); ax.set_xlabel("score (higher = more in-distribution)")
        axes[0].set_ylabel("density"); axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Score distributions, {title}: the less ID and OOD overlap, the better the detector", y=1.04)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig06_score_histograms_{idn}.png", bbox_inches="tight"); plt.close(fig)


def fig07_ours_vs_article():
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for j, (idn, title) in enumerate(IDS):
        for i, (mi, mname) in enumerate([(0, "FPR95 (lower is better)"), (1, "AUROC (higher is better)")]):
            ax = axes[i, j]; x = np.arange(3); w = 0.36
            for k, s in enumerate(COL):
                ours = clip_csv(idn, s)["AVG"][mi]
                art = ART[idn][s]["AVG"][mi] if idn in ART else ART_IMAGENET[s][mi]
                ax.bar(x[k] - w / 2, ours, w, color=COL[s])
                ax.bar(x[k] + w / 2, art, w, color="white", edgecolor=COL[s], hatch="///", lw=1.2)
                ax.text(x[k] - w / 2, ours + 0.6, f"{ours:.1f}", ha="center", fontsize=8)
                ax.text(x[k] + w / 2, art + 0.6, f"{art:.1f}", ha="center", fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(list(COL)); ax.set_ylabel(mname if j == 0 else "")
            if i == 0: ax.set_title(title); ax.set_ylim(0, 100)
            else: ax.set_ylim(60, 100)
    fig.legend(handles=[Patch(facecolor="#888", label="ours (our runs)"), Patch(facecolor="white", edgecolor="#555", hatch="///", label="article (reported values)")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Our results vs the article (averages over the OOD sets)", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/fig07_ours_vs_article.png", bbox_inches="tight"); plt.close(fig)


# ImageNet table of the article: (name, group, implemented_by_us)
IMN = [("MSP", "Supervised (post-hoc)", True), ("Energy", "Supervised (post-hoc)", True), ("ODIN", "Supervised (post-hoc)", True),
       ("ViM", "Supervised (post-hoc)", False), ("KNN", "Supervised (post-hoc)", False), ("NPOS", "Supervised (post-hoc)", False),
       ("MCM", "Zero-shot (CLIP)", True), ("L-MCM", "Zero-shot (CLIP)", True), ("GL-MCM", "Zero-shot (CLIP)", True),
       ("CoOp_MCM", "Prompt learning", False), ("CoOp_GL-MCM", "Prompt learning", False),
       ("LoCoOp_MCM", "Prompt learning", False), ("LoCoOp_GL-MCM", "Prompt learning", False)]


def imagenet_ours():
    ours = {s: clip_csv("ImageNet", s)["AVG"] for s in COL}
    for s in ["MSP", "Energy"]:
        ours[s] = csv_rows(f"results/ImageNet/{s}/ViT-B_16_in21k_ft_in1k_ID_final_msp_energy/final_msp_energy.csv")["AVG"]
    ours["ODIN"] = csv_rows("results/ImageNet/ODIN/ViT-B_16_in21k_ft_in1k_ID_final_odin/final_odin.csv")["AVG"]
    return ours


def fig08_imagenet_all_methods():
    ours = imagenet_ours()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True)
    y = np.arange(len(IMN))[::-1]
    for ax, mi, title in [(axes[0], 0, "FPR95 (lower is better)"), (axes[1], 1, "AUROC (higher is better)")]:
        for yy, (name, grp, impl) in zip(y, IMN):
            art = ART_IMAGENET[name][mi]
            if impl:
                col = COL.get(name, "#6b7280")
                ax.barh(yy, ours[name][mi], 0.55, color=col)
                ax.plot([art], [yy], marker="D", color="black", ms=6, zorder=5)
                ax.text(max(ours[name][mi], art) + 2.2, yy, f"{ours[name][mi]:.1f}", va="center", fontsize=8)
            else:
                ax.barh(yy, art, 0.55, color="white", edgecolor="#888", hatch="///")
                ax.text(art + 1, yy, f"{art:.1f}", va="center", fontsize=8, color="#555")
        ax.set_title(title); ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in IMN])
        ax.set_xlim(0 if mi == 0 else 60, 105 if mi == 0 else 100)
        ax.set_xlabel("" if mi == 0 else "(axis starts at 60)")
    for yy, (name, grp, impl) in zip(y, IMN):
        if name in ("MCM", "CoOp_MCM"):
            axes[0].axhline(yy + 0.5, color="#bbb", lw=0.8)
    fig.legend(handles=[Patch(facecolor="#6b7280", label="implemented and run by us (ours)"),
                        Patch(facecolor="white", edgecolor="#888", hatch="///", label="NOT implemented: value taken from the article"),
                        Line2D([0], [0], marker="D", color="black", ls="", label="article value of the method we implemented")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.925), fontsize=9)
    fig.suptitle("ImageNet-1K benchmark (article's main table): which methods we ran and which we did not\n"
                 "ODIN: 2,000 ID / 500 per OOD set; MSP and Energy: 10,000 ID / 3,000 per OOD set; CLIP methods: full sets",
                 y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(f"{OUT}/fig08_imagenet_all_methods_implemented_vs_not.png", bbox_inches="tight"); plt.close(fig)


def fig09_gain_heatmap():
    fig, axes = plt.subplots(2, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [6, 6, 5]})
    for j, (idn, title) in enumerate(IDS):
        names = OOD[idn] + ["AVG"]
        m, g = clip_csv(idn, "MCM"), clip_csv(idn, "GL-MCM")
        for i, (mi, lab, lim, sign) in enumerate([(0, "FPR95 change\n(negative = better)", 30, -1), (1, "AUROC change\n(positive = better)", 7, 1)]):
            ax = axes[i, j]; ax.grid(False)
            d = np.array([[g[n][mi] - m[n][mi] for n in names]])
            ax.imshow(sign * d, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")     # green always means GL-MCM is better
            for c in range(len(names)):
                ax.text(c, 0, f"{d[0, c]:+.1f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(d[0, c]) > 0.7 * lim else "black")
            ax.set_yticks([0]); ax.set_yticklabels([lab] if j == 0 else [""])
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names if i == 1 else [""] * len(names), rotation=25, ha="right")
            if i == 0: ax.set_title(title)
    fig.suptitle("GL-MCM minus MCM for every OOD set (green = adding local features helped, red = it hurt)", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig09_gain_per_ood_set_heatmap.png", bbox_inches="tight"); plt.close(fig)


def fig10_checkpoint_effect():
    """Finding: the same post-hoc scores behave very differently on two ViT-B/16 checkpoints."""
    try:
        tv = {s: csv_rows(f"results/ImageNet/{s}/ViT-B_16_supervised_ID_final_msp_energy/final_msp_energy.csv")["AVG"] for s in ["MSP", "Energy"]}
    except FileNotFoundError:
        print("torchvision baseline results not found, skipping fig10"); return
    ours = imagenet_ours()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, mi, title in [(axes[0], 0, "FPR95 (lower is better)"), (axes[1], 1, "AUROC (higher is better)")]:
        x = np.arange(2); w = 0.26
        for k, (lab, vals, col) in enumerate([("torchvision ViT-B/16 (ImageNet-1K only)", [tv["MSP"][mi], tv["Energy"][mi]], "#9aa5b1"),
                                               ("timm ViT-B/16 (21k pre-train + 1k fine-tune)", [ours["MSP"][mi], ours["Energy"][mi]], "#2a78d6"),
                                               ("article", [ART_IMAGENET["MSP"][mi], ART_IMAGENET["Energy"][mi]], "white")]):
            bars = ax.bar(x + (k - 1) * w, vals, w, color=col, edgecolor="#555" if col == "white" else col, hatch="///" if col == "white" else None, label=lab)
            for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(["MSP", "Energy"]); ax.set_title(title)
        ax.set_ylim(0, 100) if mi == 0 else ax.set_ylim(60, 100)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Backbone checkpoint matters: same scores, different ViT-B/16 weights (ImageNet-1K, 4 OOD sets)", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig10_baseline_checkpoint_effect.png", bbox_inches="tight"); plt.close(fig)


def fig11_odin_same_subset():
    """MSP, Energy and ODIN on exactly the same 2,000 ID / 4x500 OOD images (fair comparison)."""
    d = {s: csv_rows(f"results/ImageNet/{s}/ViT-B_16_in21k_ft_in1k_ID_final_odin/final_odin.csv") for s in ["MSP", "Energy", "ODIN"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mi, title in [(axes[0], 0, "FPR95 (lower is better)"), (axes[1], 1, "AUROC (higher is better)")]:
        x = np.arange(3); w = 0.36
        for k, s in enumerate(["MSP", "Energy", "ODIN"]):
            v, a = d[s]["AVG"][mi], ART_IMAGENET[s][mi]
            ax.bar(x[k] - w / 2, v, w, color="#2a78d6"); ax.bar(x[k] + w / 2, a, w, color="white", edgecolor="#555", hatch="///")
            ax.text(x[k] - w / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8); ax.text(x[k] + w / 2, a + 0.8, f"{a:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(["MSP", "Energy", "ODIN"]); ax.set_title(title)
        ax.set_ylim(0, 100) if mi == 0 else ax.set_ylim(60, 100)
    fig.legend(handles=[Patch(facecolor="#2a78d6", label="ours (2,000 ID / 500 per OOD set)"), Patch(facecolor="white", edgecolor="#555", hatch="///", label="article")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("MSP, Energy and ODIN on the same images (ours) vs the article", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/fig11_msp_energy_odin_same_subset.png", bbox_inches="tight"); plt.close(fig)


ALL = [fig01_local_feature, fig02_before_after, fig03_temperature, fig04_lambda, fig05_roc_pr, fig06_histograms,
       fig07_ours_vs_article, fig08_imagenet_all_methods, fig09_gain_heatmap, fig10_checkpoint_effect, fig11_odin_same_subset]

if __name__ == "__main__":
    check()
    if "--check" in sys.argv:
        sys.exit(0)
    for f in ALL:
        f(); print("ok", f.__name__)
    print("figures written to", OUT)
