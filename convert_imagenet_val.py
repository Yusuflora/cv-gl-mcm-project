"""Convert HF ILSVRC/imagenet-1k validation parquet shards into
datasets/ImageNet/<000..999>/<n>.JPEG so torchvision ImageFolder
(sorted folder names) gives label == class index (zero-padded names)."""
import glob, os, sys
import pyarrow.parquet as pq

src = sys.argv[1] if len(sys.argv) > 1 else "datasets/hf_imagenet/data"
dst = sys.argv[2] if len(sys.argv) > 2 else "datasets/ImageNet"
files = sorted(glob.glob(os.path.join(src, "validation-*.parquet")))
print(f"{len(files)} shards found")
n_total = 0
for f in files:
    pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
        imgs = batch.column("image").to_pylist()
        labels = batch.column("label").to_pylist()
        for im, lab in zip(imgs, labels):
            d = os.path.join(dst, f"{lab:03d}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"val_{n_total:05d}.JPEG"), "wb") as fh:
                fh.write(im["bytes"])
            n_total += 1
    print(os.path.basename(f), "done, total", n_total, flush=True)
print("finished:", n_total, "images (expected 50000)")
