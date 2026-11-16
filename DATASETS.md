# Veri setleri: nereden indirilir, nasıl yerleştirilir

Veri setleri GitHub'a/zip'e KONMAZ (toplam on binlerce görüntü, onlarca GB). Tablo ve grafikleri yeniden üretmek için veri seti gerekmez
(`results/` ve `results_raw/` içindeki dosyalar yeterli). Veri setleri yalnızca CLIP koşularını sıfırdan tekrarlamak için gerekir.

Hepsi `GL-MCM/datasets/` altına şu yapıda yerleştirilir (kod bu klasör adlarını bekler):

```
datasets/
  ImageNet/          # 000 ... 999 (1000 klasör, toplam 50.000 görüntü)  = ImageNet-1K doğrulama seti   (6.4 GB)
  ID_COCO_single/    # 5000 görüntü, 60 sınıf klasörü (ör. 10_traffic_light)                          (243 MB)
  ID_COCO_multi/     # images/ içinde 5000 görüntü (bu çalışmada koşturulmadı)                         (231 MB)
  ID_VOC_single/     # 1000 görüntü, 14 sınıf klasörü (aeroplane, bicycle, ...)                       (36 MB)
  OOD_COCO/images/   # 1000 görüntü (VOC ID'nin OOD'si)                                               (52 MB)
  OOD_VOC/images/    # 4000 görüntü (COCO ID'nin OOD'si)                                              (145 MB)
  iNaturalist/images/  # 10.000 görüntü                                                               (3.8 GB)
  SUN/images/          # 10.000 görüntü                                                               (3.7 GB)
  Places/images/       # 10.000 görüntü                                                               (678 MB)
  Texture/images/<47 kategori klasörü>/   # DTD, 5.640 görüntü                                        (614 MB)
  ImageNet-22K/      # README.txt + 20 adet n0xxxxxxx sınıf klasörü, toplam 18.335 görüntü            (1.7 GB)
```
Önemli: kod `torchvision.datasets.ImageFolder` kullandığı için OOD klasörlerinin içinde en az bir alt klasör (burada `images`) olmalıdır; düz yığın görüntü klasörü çalışmaz.
(Görüntü sayıları bizim makinemizdeki klasörlerden sayılmıştır.)

## 1) Yazarların hazırladığı paket (COCO / VOC ID ve OOD_COCO, OOD_VOC)
- Kaynak: yazarların GitHub README'sinde verdiği Google Drive dosyası: https://drive.google.com/file/d/1he4jKi2BfyGT6rkcbFYlez7PbLMXTBMR/view?usp=sharing (dosya kimliği `1he4jKi2BfyGT6rkcbFYlez7PbLMXTBMR`; bizde `datasets.tar.gz`, ~706 MB).
- İndirme: tarayıcıdan veya `gdown 1he4jKi2BfyGT6rkcbFYlez7PbLMXTBMR`. Sonra `tar -xzf datasets.tar.gz`; içindeki klasörleri yukarıdaki adlarla `datasets/` altına koyun.

## 2) iNaturalist, SUN, Places (Huang ve ark. 2021 alt kümeleri)
- Yazarların README'si: "Huang ve ark. 2021'in hazırladığı alt kümeleri, https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset adresindeki talimatla indirin."
- Biz `https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/<ad>.tar.gz` biçimindeki adresten `iNaturalist.tar.gz` (~3.9 GB), `SUN.tar.gz` (~3.9 GB), `Places.tar.gz` (~0.7 GB) dosyalarını indirdik. Adresin hâlâ çalıştığını doğrulayın; çalışmıyorsa yukarıdaki depodaki güncel talimatı izleyin.
- Her biri 10.000 görüntülüktür ve ImageNet-1K ile çakışan sınıflar çıkarılmıştır (makalenin ve Huang ve ark.'nın tarifi).
- Klasör adları `iNaturalist/`, `SUN/`, `Places/` olmalı (kodda `places365` adı `Places` klasörüne bağlanır).

## 3) Texture = DTD (Describable Textures Dataset)
- Resmi site: Oxford VGG, `dtd-r1.0.1.tar.gz` (~625 MB), https://www.robots.ox.ac.uk/~vgg/data/dtd/ . 
- Açtıktan sonra `dtd/images/` klasörünü `datasets/Texture/images/` olarak yerleştirin (47 kategori klasörü, 5.640 görüntü). Kod ImageFolder köküne `Texture/images` verir.

## 4) ImageNet-22K OOD alt kümesi (Wang ve ark. 2021)
- Kaynak (yazarların README'si): https://drive.google.com/drive/folders/1BGMRQz3eB_npaGD46HC6K_uzt105HPRy (deposu: https://github.com/deeplearning-wisc/multi-label-ood).
- Klasörde iki dosya var: `ImagenetOOD_for_COCO_VOC.tar` (bizim kullandığımız) ve `ImagenetOOD_for_NUS-WIDE.zip` (kullanmadık). `gdown --folder <klasör kimliği>` ile indirilebilir.
- `ImagenetOOD_for_COCO_VOC.tar` açılınca elde edilen sınıf klasörleri (n0xxxxxxx; 20 klasör) ve `README.txt` `datasets/ImageNet-22K/` altında durur (toplam 18.335 görüntü). Bizim klasör `ImageFolder` ile okunuyor; açtıktan sonra `datasets/ImageNet-22K` içinde doğrudan n0... klasörlerinin görünmesi gerekir (tar içindeki `imdb`/`labels` klasörleri indirme sırasında oluşan ek dizinlerdir; gerekirse kontrol edin).

## 5) ImageNet-1K doğrulama seti (50.000 görüntü)
- Resmi dağıtım https://image-net.org/challenges/LSVRC/2012/ ; biz Hugging Face'teki `ILSVRC/imagenet-1k` veri setini kullandık (ücretsiz hesap + şartları kabul etme gerekir; her kişi kendi hesabıyla yapmalıdır, başkasının giriş anahtarı paylaşılmaz).
- Giriş ve indirme: `hf auth login` sonra `hf download ILSVRC/imagenet-1k --repo-type dataset --include "data/validation-*.parquet" --local-dir datasets/hf_imagenet` (komutun tam sözdizimini `hf download --help` ile kontrol edin).
- Parquet'ten klasörlere çevirme: `python convert_imagenet_val.py` (varsayılan: `datasets/hf_imagenet/data` → `datasets/ImageNet`). Klasör adları sıfır dolgulu 000-999'dur; böylece sıralama sınıf numarasıyla birebir aynı olur.
- ImageNet sınıf adları `data/ImageNet/imagenet_class_clean.npy` içinde depoda mevcuttur.

## Sorun çıkarsa
- Tablo/grafikleri veri olmadan yeniden üretmek için: `python make_figures.py && python make_tables.py` (`results/` ve `results_raw/` yeterli).
- Çok yavaş indirme: `curl -C -` (kaldığı yerden devam) kullandık.
- Disk (bizim makinede ölçülen): açılmış veri setleri ~17 GB, indirilen arşivler (`downloads`) ~12 GB, Hugging Face parquet dosyaları (`hf_imagenet`) ~6.3 GB. Dönüştürme bittikten sonra `downloads` ve `hf_imagenet` silinebilir.
