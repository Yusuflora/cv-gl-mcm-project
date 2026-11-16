"""Builds markdown result tables from results/**/final.csv and prints/writes them to docs/RESULT_TABLES.md.
Compares our numbers with the values reported in the GL-MCM article. Only uses the standard library."""
import csv, os

NAME = os.environ.get("RUN_NAME", "final")
ART = {
    'COCO_single': {
        'MCM': {'iNaturalist': (64.42, 90.75), 'SUN': (84.18, 81.81), 'Texture': (65.10, 86.62), 'IN22k': (78.50, 84.14), 'ood_voc': (77.62, 84.43), 'AVG': (73.96, 85.55)},
        'L-MCM': {'iNaturalist': (41.80, 92.46), 'SUN': (51.52, 89.24), 'Texture': (85.68, 74.87), 'IN22k': (75.50, 77.92), 'ood_voc': (72.02, 84.21), 'AVG': (65.30, 83.74)},
        'GL-MCM': {'iNaturalist': (34.72, 94.45), 'SUN': (61.96, 88.49), 'Texture': (72.08, 84.53), 'IN22k': (70.66, 84.49), 'ood_voc': (71.62, 87.72), 'AVG': (62.21, 87.94)}},
    'VOC_single': {
        'MCM': {'iNaturalist': (8.20, 98.23), 'SUN': (28.60, 94.68), 'Texture': (51.70, 91.45), 'IN22k': (51.40, 90.94), 'ood_coco': (54.50, 89.02), 'AVG': (38.88, 92.86)},
        'L-MCM': {'iNaturalist': (27.70, 94.97), 'SUN': (44.20, 88.87), 'Texture': (60.00, 86.23), 'IN22k': (54.10, 86.27), 'ood_coco': (57.00, 84.06), 'AVG': (48.72, 88.08)},
        'GL-MCM': {'iNaturalist': (4.20, 98.71), 'SUN': (23.10, 94.66), 'Texture': (43.00, 92.84), 'IN22k': (41.00, 92.38), 'ood_coco': (44.30, 90.48), 'AVG': (31.12, 93.81)}},
}
# ImageNet article averages (FPR95, AUROC) over iNaturalist, SUN, Places, Texture
ART_IMAGENET = {'MSP': (45.12, 88.25), 'Energy': (35.92, 92.26), 'ODIN': (47.75, 88.80), 'ViM': (50.20, 87.82), 'KNN': (42.19, 90.97),
                'NPOS': (37.93, 91.22), 'MCM': (42.74, 90.77), 'L-MCM': (73.59, 73.73), 'GL-MCM': (35.25, 91.51),
                'CoOp_MCM': (36.83, 91.93), 'CoOp_GL-MCM': (30.03, 92.59), 'LoCoOp_MCM': (33.98, 92.69), 'LoCoOp_GL-MCM': (28.68, 93.59)}
SCORES = ['MCM', 'L-MCM', 'GL-MCM']


def read(id_name, score, T):
    p = f"results/{id_name}/{score}/CLIP_ViT-B_16_T_{T}_ID_{NAME}/{NAME}.csv"
    if not os.path.exists(p):
        return None
    rows = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            k = r[''] if '' in r else list(r.values())[0]
            rows[k] = (float(r['FPR95']), float(r['AUROC']), float(r['AUPR']))
    n = len(rows)
    rows['AVG'] = tuple(sum(v[i] for v in rows.values()) / n for i in range(3))
    return rows


def table(id_name, T='1'):
    out = [f"### {id_name}, tau={T} (FPR95 lower is better, AUROC higher is better)\n"]
    first = read(id_name, 'MCM', T)
    if first is None:
        return f"### {id_name}, tau={T}: results not found yet\n"
    cols = list(first.keys())
    out.append("| Method | " + " | ".join(cols) + " |")
    out.append("|---|" + "---|" * len(cols))
    for s in SCORES:
        r = read(id_name, s, T)
        if r is None:
            continue
        out.append(f"| {s} (ours) | " + " | ".join(f"{r[c][0]:.2f} / {r[c][1]:.2f}" for c in cols) + " |")
        a = ART.get(id_name, {}).get(s)
        if a and T == '1':
            out.append(f"| {s} (article) | " + " | ".join(f"{a[c][0]:.2f} / {a[c][1]:.2f}" if c in a else "-" for c in cols) + " |")
    return "\n".join(out) + "\n"


def tau_table(id_name):
    out = [f"### {id_name}: effect of temperature tau (averages over OOD sets, FPR95 / AUROC)\n", "| Method | tau=1 | tau=0.01 |", "|---|---|---|"]
    for s in SCORES:
        a, b = read(id_name, s, '1'), read(id_name, s, '0.01')
        if a and b:
            out.append(f"| {s} | {a['AVG'][0]:.2f} / {a['AVG'][1]:.2f} | {b['AVG'][0]:.2f} / {b['AVG'][1]:.2f} |")
    return "\n".join(out) + "\n"


# Article Table 3 (ablation on tau, GL-MCM only): (FPR95, AUROC), checked against arXiv 2304.04521v4 (HTML version)
ART_TAU3 = {'COCO_single': {'0.01': (64.89, 86.70), '1': (62.21, 87.94)},
            'VOC_single': {'0.01': (43.92, 90.80), '1': (31.12, 93.81)},
            'ImageNet': {'0.01': (50.72, 84.70), '1': (35.25, 91.51)}}


def tau_article_table():
    out = ["### GL-MCM temperature ablation: ours vs article Table 3 (FPR95 / AUROC, average over OOD sets)\n",
           "| ID dataset | tau | Ours | Article (Table 3) |", "|---|---|---|---|"]
    for idn, d in ART_TAU3.items():
        for T in ['0.01', '1']:
            r = read(idn, 'GL-MCM', T)
            if r:
                out.append(f"| {idn} | {T} | {r['AVG'][0]:.2f} / {r['AVG'][1]:.2f} | {d[T][0]:.2f} / {d[T][1]:.2f} |")
    return "\n".join(out) + "\n"


IMN = [("MSP", "Supervised (post-hoc)"), ("Energy", "Supervised (post-hoc)"), ("ODIN", "Supervised (post-hoc)"),
       ("ViM", "Supervised (post-hoc)"), ("KNN", "Supervised (post-hoc)"), ("NPOS", "Supervised (post-hoc)"),
       ("MCM", "Zero-shot CLIP"), ("L-MCM", "Zero-shot CLIP"), ("GL-MCM", "Zero-shot CLIP"),
       ("CoOp_MCM", "Prompt learning"), ("CoOp_GL-MCM", "Prompt learning"), ("LoCoOp_MCM", "Prompt learning"), ("LoCoOp_GL-MCM", "Prompt learning")]


def imagenet_baseline_results():
    """(FPR95, AUROC, AUPR) averages of the methods we ran on ImageNet; None if the run is missing."""
    out = {}
    for s in SCORES:
        r = read('ImageNet', s, '1')
        out[s] = r['AVG'] if r else None
    runs = {'MSP': ('final_msp_energy', 'MSP'), 'Energy': ('final_msp_energy', 'Energy'), 'ODIN': ('final_odin', 'ODIN')}
    for s, (run, _) in runs.items():
        p = f"results/ImageNet/{s}/ViT-B_16_in21k_ft_in1k_ID_{run}/{run}.csv"
        if os.path.exists(p):
            rows = {}
            with open(p) as f:
                for r in csv.DictReader(f):
                    rows[list(r.values())[0]] = (float(r['FPR95']), float(r['AUROC']), float(r['AUPR']))
            out[s] = rows['AVG']
    return out


def baseline_table():
    ours = imagenet_baseline_results()
    note = {'MSP': 'ours: 10,000 ID / 3,000 per OOD set', 'Energy': 'ours: 10,000 ID / 3,000 per OOD set',
            'ODIN': 'ours: 2,000 ID / 500 per OOD set (ODIN is slow)', 'MCM': 'ours: full sets', 'L-MCM': 'ours: full sets', 'GL-MCM': 'ours: full sets, lambda=0.5',
            'ViM': 'NOT implemented (needs ImageNet training features)', 'KNN': 'NOT implemented (needs ImageNet training features)',
            'NPOS': 'NOT implemented (needs training)', 'CoOp_MCM': 'NOT implemented (needs 16-shot training)', 'CoOp_GL-MCM': 'NOT implemented (needs 16-shot training)',
            'LoCoOp_MCM': 'NOT implemented (needs 16-shot training)', 'LoCoOp_GL-MCM': 'NOT implemented (needs 16-shot training)'}
    out = ["### ImageNet-1K main table of the article: what we implemented and what we did not (averages over iNaturalist, SUN, Places, Texture)\n",
           "| Method | Group | Status | Ours FPR95 | Ours AUROC | Article FPR95 | Article AUROC | Note |", "|---|---|---|---|---|---|---|---|"]
    for name, grp in IMN:
        o = ours.get(name)
        a = ART_IMAGENET[name]
        status = "IMPLEMENTED" if o else "not implemented"
        of, oa = (f"{o[0]:.2f}", f"{o[1]:.2f}") if o else ("-", "-")
        out.append(f"| {name} | {grp} | {status} | {of} | {oa} | {a[0]:.2f} | {a[1]:.2f} | {note[name]} |")
    return "\n".join(out) + "\n"


def aupr_table():
    out = ["### AUPR (ours only; the article does not report AUPR), averages over OOD sets, tau=1\n", "| ID dataset | MCM | L-MCM | GL-MCM |", "|---|---|---|---|"]
    for idn in ['COCO_single', 'VOC_single', 'ImageNet']:
        vals = [read(idn, s, '1')['AVG'][2] if read(idn, s, '1') else None for s in SCORES]
        out.append(f"| {idn} | " + " | ".join(f"{v:.2f}" if v is not None else "-" for v in vals) + " |")
    return "\n".join(out) + "\n"


def write_long_csv():
    """One tidy file with every number we measured (for charts / for the report)."""
    os.makedirs('tables', exist_ok=True)
    rows = []
    for idn in ['COCO_single', 'VOC_single', 'ImageNet']:
        for T in ['1', '0.01']:
            for s in SCORES:
                r = read(idn, s, T)
                if r:
                    for k, v in r.items():
                        rows.append([idn, 'CLIP ViT-B/16', T, s, k, *v])
    ours = imagenet_baseline_results()
    for s in ['MSP', 'Energy', 'ODIN']:
        run = 'final_odin' if s == 'ODIN' else 'final_msp_energy'
        p = f"results/ImageNet/{s}/ViT-B_16_in21k_ft_in1k_ID_{run}/{run}.csv"
        if os.path.exists(p):
            with open(p) as f:
                for r in csv.DictReader(f):
                    rows.append(['ImageNet', f'ViT-B/16 in21k->in1k ({run})', '-', s, list(r.values())[0], float(r['FPR95']), float(r['AUROC']), float(r['AUPR'])])
    with open('tables/all_results_long.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id_dataset', 'backbone', 'tau', 'method', 'ood_set', 'FPR95', 'AUROC', 'AUPR'])
        w.writerows(rows)
    return len(rows)


if __name__ == '__main__':
    parts = ["# Result tables (generated by make_tables.py)\n", "Each cell: FPR95 / AUROC. AVG = mean over the OOD sets.\n"]
    for idn in ['COCO_single', 'VOC_single', 'ImageNet']:
        parts += [table(idn, '1'), tau_table(idn)]
    parts += [tau_article_table(), baseline_table(), aupr_table()]
    if os.path.exists('tables/lambda_sweep.md'):
        parts.append(open('tables/lambda_sweep.md').read())
    print('rows in tables/all_results_long.csv:', write_long_csv())
    os.makedirs('docs', exist_ok=True)
    open('docs/RESULT_TABLES.md', 'w').write("\n".join(parts))
    print("\n".join(parts))
