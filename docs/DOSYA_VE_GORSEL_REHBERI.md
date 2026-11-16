# Dosya ve görsel rehberi

Bu dosya, depodaki her dosyanın ne olduğunu ve her şeklin (figure) neyi gösterdiğini, nasıl okunacağını, hangi rakamı öne çıkaracağını ve rapora nerede koyulacağını anlatır. Rakamların hepsi `docs/RESULT_TABLES.md` ile aynıdır. Şekilleri **mutlaka açıp bakın**: bu rehberdeki "ne gösterir" kısmı, grafik koduna (`make_figures.py`) ve tablolara dayanır; renk/yerleşim ayrıntısı için şeklin kendisine bakın.

Renk kuralı (tüm şekillerde aynı): **MCM mavi, L-MCM turuncu, GL-MCM yeşil.**

## A. Klasör ve dosya haritası

| Yol | Ne olduğu |
|---|---|
| `README.md` | Yazarların orijinal README'si (bizim değil) |
| `START_HERE.md` | Bu paketin giriş sayfası |
| `docs/PROJE_ACIKLAMASI.md` | **Ana açıklama**: makale, adımlar, kod haritası, sonuçlar, sınırlamalar, rapor iskeleti, S&C |
| `docs/DOSYA_VE_GORSEL_REHBERI.md` | Bu dosya |
| `docs/KARARLAR_VE_NOTLAR.md` | Proje boyunca alınan kararların ham notları (bazı maddeler sonradan güncellendi; çelişki varsa PROJE_ACIKLAMASI geçerlidir) |
| `docs/RESULT_TABLES.md` | `make_tables.py` çıktısı: tüm tablolar |
| `docs/AGENT_PROMPT.md` | Yapay zekâ ajanına verilecek hazır talimat |
| `DATASETS.md` | Veri setleri nereden indirilir / nasıl yerleştirilir |
| `requirements.txt` | Sürümler |
| `eval_all_scores.py` | CLIP ile MCM/L-MCM/GL-MCM tek geçişli değerlendirme (ana kod) |
| `baselines_imagenet.py` | MSP, Energy, ODIN (ImageNet) |
| `convert_imagenet_val.py` | ImageNet doğrulama parquet → klasörler |
| `make_tables.py` | Tablolar (yalnızca standart kütüphane) |
| `make_figures.py` | Şekiller fig01-fig11 (numpy + matplotlib) |
| `qualitative_examples.py` | Şekiller fig12-fig13 (CLIP gerekir; koşturuldu, çıktılar `figures/` içinde) |
| `eval_ood_detection.py`, `clip/`, `utils/`, `scripts/`, `data/` | Yazarların orijinal kodu (küçük yamalarımızla; bkz. PROJE_ACIKLAMASI §4.2) |
| `figures/` | Tüm şekiller (PNG) |
| `tables/all_results_long.csv` | Tüm sonuçlar tek uzun CSV (ID veri seti, τ, yöntem, OOD seti, FPR95, AUROC, AUPR) |
| `tables/lambda_sweep.md` | λ taraması tablosu |
| `results/<ID>/<Yöntem>/<koşu>/` | Resmi CSV (CLIP: `final.csv`; baseline: `final_msp_energy.csv`, `final_odin.csv`) ve CLIP koşularında skor dağılımı PNG'leri (`MCM_SUN.png` vb.); ilk deneme klasörlerinde ayrıca `ood_eval_info.log` ve `test.csv` bulunur |
| `results_raw/` | Görüntü başına ham skorlar (`.npz`); şekiller ve tüm skorlar buradan yeniden hesaplanır |
| `results/GL-MCM_progress_report.pdf` | 5 Kasım'da hazırlanan ilk ilerleme raporu (yalnızca COCO_single/ood_voc; eski, nihai değil) |
| `results/before_after_MCM_vs_GL-MCM.png`, `results/ours_vs_article.png` | 12 Kasım'daki ilk sürüm şekiller (eski; nihai şekiller `figures/` içinde) |

`results/<ID>/` içindeki koşu klasör adları: `CLIP_ViT-B_16_T_<τ>_ID_<etiket>`. Etiketler: `final` (nihai koşu), `m1test` (M1'de aynı test koşusu; M5 ile aynı), `test` (ilk deneme). **Nihai sayılar yalnızca `final` etiketlidir.** Baseline klasörleri: `results/ImageNet/{MSP,Energy,ODIN}/`; `ViT-B_16_in21k_ft_in1k_*` = timm modeli (ana sonuç), `ViT-B_16_supervised_*` = torchvision modeli (checkpoint kıyası için ek bulgu).

Yeniden üretme (sırayla): `python make_figures.py` sonra `python make_tables.py` (veri seti gerekmez; λ tablosunu önce şekil betiği üretir). `python make_figures.py --check`: kendi metrik hesabının resmi CSV'lerle uyuştuğunu doğrular (en büyük fark 0.008 puan).

## B. Şekil rehberi

Her kayıt: dosya adı → ne gösterir → nasıl okunur → öne çıkacak rakamlar → rapordaki yeri → dikkat.

### fig01_local_feature_MCM_vs_GLMCM.png (Kavram 1, ana şekil)
- **Ne gösterir:** 2 satır x 3 sütun. Sütunlar: COCO, VOC, ImageNet-1K (ID). Üst satır FPR95 (düşük iyi), alt satır AUROC (yüksek iyi). Her OOD seti ve AVG için üç çubuk: MCM (yerel özellik yok), L-MCM (yalnızca yerel), GL-MCM (global + yerel).
- **Nasıl okunur:** Mavi (MCM) ile yeşil (GL-MCM) çubuğu karşılaştırın; yeşil FPR95'te kısa, AUROC'ta uzunsa yerel özellik yardım etmiştir.
- **Rakamlar:** AVG: COCO FPR95 73.80 → 62.34, VOC 42.66 → 30.32, ImageNet 42.26 → 35.33; AUROC 85.55 → 87.95, 92.56 → 93.65, 90.83 → 91.52. L-MCM ImageNet'te en kötü (73.49 / 73.80).
- **Rapor:** "Kavram 1: yerel özellik var/yok" bölümünün ilk şekli.
- **Dikkat:** AUROC panellerinin y ekseni 50-100 arası kesilmiştir (farkları görünür kılmak için; bunu altyazıda belirtin).

### fig02_before_after_summary.png (Kavram 1 özet)
- **Ne gösterir:** MCM'den GL-MCM'e geçişin üç ID veri setindeki ortalama değişimi (FPR95 ve AUROC), değişim değerleri yazılı.
- **Rakamlar:** ΔFPR95 -11.5 / -12.3 / -6.9; ΔAUROC +2.4 / +1.1 / +0.7.
- **Rapor:** Yerel özellik bölümünün özeti ve sonuç bölümü. "Öncesi/sonrası" kıyasının en sade hâli.

### fig03_temperature_tau1_vs_tau001.png (Kavram 2: sıcaklık)
- **Ne gösterir:** 2 satır x 3 sütun (FPR95 / AUROC; COCO, VOC, ImageNet). Her yöntem (MCM, L-MCM, GL-MCM) için iki çubuk: düz dolgu = τ=1 (sıcaklık ölçekleme yok), çizgili (///) = τ=0.01 (ölçekleme var). Değerler çubukların üstünde yazılı.
- **Rakamlar:** ImageNet GL-MCM 35.33/91.52 (τ=1) → 50.66/84.71 (τ=0.01); VOC GL-MCM 30.32/93.65 → 42.34/91.25; COCO GL-MCM 62.34/87.95 → 64.42/86.63. Tek istisna: COCO MCM'de FPR95 τ=0.01'de hafif iyi (73.12 vs 73.80), AUROC yine τ=1'de iyi.
- **Rapor:** "Kavram 2" bölümünün ana şekli. Makalenin Tablo 3'ü ile uyumu metinde vurgulanabilir (PROJE_ACIKLAMASI §6.4).

### fig04_lambda_sweep.png (ek analiz: λ)
- **Ne gösterir:** GL-MCM'de yerel skor ağırlığı λ'nın 0'dan 3'e etkisi (λ=0 tam MCM), üç ID veri seti.
- **Rakamlar:** COCO FPR95 en iyi λ=1.5 (61.52), AUROC en iyi λ=1 (87.95); VOC en iyi λ≈0.5-0.75 (30.12-30.10); ImageNet FPR95 en iyi λ=0.75 (34.74), AUROC en iyi λ=0.5 (91.51). Makalenin λ'ları (1.0 / 1.0 / 0.5) en iyiye çok yakın.
- **Dikkat:** tarama test verisi üzerinde yapıldı (doğrulama ayrımı yok); "makalenin λ'sı doğru" değil, "tutarlı" denmeli. Tablosu: `tables/lambda_sweep.md`.

### fig05_roc_curves_<ID>.png ve fig05_pr_curves_<ID>.png (ID = COCO_single, VOC_single, ImageNet)
- **Ne gösterir:** Her OOD seti için ROC eğrisi (x: OOD'nin ID sayılma oranı, y: ID'nin bulunma oranı) ve kesinlik-duyarlılık eğrisi; üç yöntem aynı eksende. ROC'ta kesikli yatay çizgi %95 TPR'yi gösterir: FPR95 bu çizginin eğri ile kesiştiği yerdeki x değeridir.
- **Nasıl okunur:** Eğri sol-üst köşeye ne kadar yakınsa o kadar iyi; AUROC = eğri altındaki alan = "ranking quality". PR eğrilerinde ID çoğunluk sınıfı olduğu için kesinlik yüksek kalır (altyazıda yazılı).
- **Rapor:** Hocanın "ranking quality'yi açıkla" isteği için doğrudan görsel destek: AUROC eşikten bağımsız sıralama kalitesidir, FPR95 tek bir eşikteki ölçüdür. 6 şekil (3 ID x ROC/PR).
- **Dikkat:** ImageNet'te L-MCM eğrisi diğerlerinin belirgin altında; COCO Texture'da GL-MCM, MCM'den geride kalır (bkz. fig09).

### fig06_score_histograms_<ID>.png
- **Ne gösterir:** ID skorlarının (gri dolgulu) ve her OOD setinin (renkli çizgi) skor dağılımı; MCM, L-MCM, GL-MCM için ayrı panel.
- **Nasıl okunur:** ID ile OOD dağılımları ne kadar az üst üste biniyorsa dedektör o kadar iyidir. Skor ekseni "yüksek = daha ID".
- **Dikkat:** τ=1'de softmax skorları yaklaşık 1/(sınıf sayısı) civarında küçük değerler alır (örn. ImageNet'te 1000 sınıf), bu yüzden eksen değerleri çok küçüktür; bu normaldir, hata değildir. GL-MCM skoru MCM + λ·L-MCM olduğu için ölçeği farklıdır; paneller arası eksenleri karşılaştırmayın, yalnızca ayrışmaya bakın.
- **Not:** `results/**/<Yöntem>_<OOD>.png` dosyaları resmi koddan çıkan KDE grafikleridir (farklı biçim, aynı fikir).

### fig07_ours_vs_article.png
- **Ne gösterir:** Bizim ortalamalarımız ile makalenin ortalamaları yan yana (üç ID veri seti, üç yöntem).
- **Rakamlar:** COCO farkları ≤0.2 puan; ImageNet MCM 42.26 vs 42.74, L-MCM 73.49 vs 73.59, GL-MCM 35.33 vs 35.25; VOC'ta daha büyük sapma (MCM FPR95 42.66 vs 38.88), ID yalnızca 1000 görüntü.
- **Rapor:** "Makaleyle kıyas / doğrulama" bölümü.

### fig08_imagenet_all_methods_implemented_vs_not.png
- **Ne gösterir:** Makalenin ImageNet tablosundaki 13 yöntem (FPR95 ve AUROC paneli). Uyguladıklarımız düz renkli çubuk (bizim sayı) + siyah elmas (makale değeri); uygulamadıklarımız çizgili beyaz çubuk (yalnızca makale değeri; "NOT implemented" göstergesi). Başlıkta alt küme boyutları yazılı.
- **Rapor:** Hocanın "makaledeki tabloyu göster, hangilerini uyguladığını/uygulamadığını belirt" isteğinin doğrudan karşılığı. Yanına §6.6 tablosunu koyun.
- **Dikkat:** MSP/Energy/ODIN alt kümelerde hesaplandı (§6.6); ODIN makaleyle uyuşmuyor, açıkça yazın.

### fig09_gain_per_ood_set_heatmap.png
- **Ne gösterir:** Her ID veri seti için, OOD seti başına GL-MCM eksi MCM farkı (üst satır FPR95, alt satır AUROC; hücre içinde sayı). Renk işaret-düzeltmelidir: **yeşil her zaman GL-MCM daha iyi**, kırmızı GL-MCM daha kötü.
- **Rakamlar/istisnalar:** COCO Texture: FPR95 +7.06 (65.00 → 72.06) ve AUROC -2.09 (kırmızı); ImageNet Texture AUROC -0.64; çoğu hücre yeşil.
- **Rapor:** "Her yerde yardımcı olmuyor" dürüstlüğü için şart; Texture OOD'lerinde yerel yamalar doku içinde sınıf adlarıyla eşleşebildiği için olası neden (test edilmemiş yorum) olarak anılabilir.

### fig10_baseline_checkpoint_effect.png
- **Ne gösterir:** Aynı MSP ve Energy skorları, iki farklı ViT-B/16 ağırlığında (torchvision yalnızca 1K = gri çubuk, timm 21k→1k = mavi çubuk) ve makale değeri (çizgili beyaz çubuk).
- **Rakamlar:** torchvision MSP 61.20/83.28, Energy 65.84/74.52; timm MSP 44.63/88.52, Energy 30.54/93.52; makale 45.12/88.25 ve 35.92/92.26.
- **Rapor:** Baseline bölümü; "checkpoint seçimi sonucu çok değiştiriyor" bulgusu. Makale checkpoint'i belirtmediği için timm "en yakın makul tahmin".

### fig11_msp_energy_odin_same_subset.png
- **Ne gösterir:** MSP, Energy, ODIN'in tamamen aynı görüntüler üzerinde (ID 2.000, 4x500 OOD) ölçümü ve makale ortalamaları.
- **Rakamlar:** MSP 43.55/88.94, Energy 29.75/93.94, ODIN 34.60/91.99 (makale 45.12/88.25, 35.92/92.26, 47.75/88.80).
- **Dikkat:** ODIN bizde makaleden belirgin iyi; sıralama farklı; neden bilinmiyor (hiperparametre/checkpoint belirtilmemiş). Alt küme küçük, sayılar oynak.

### fig12_examples_mcm_fooled_glmcm_correct_SUN.png ve fig13_examples_glmcm_fooled_mcm_correct_SUN.png (nitel örnekler)
- **Ne gösterir:** COCO_single ID eşiği (ID görüntülerinin %95'ini kabul eden eşik) ile rastgele 600 SUN görüntüsü tarandı: fig12'de MCM'nin yanlışlıkla ID saydığı ama GL-MCM'in doğru reddettiği OOD görüntüleri, fig13'te tersi. Görüntü üstünde 14x14 yerel skor haritası (her yama için en yüksek sınıf olasılığı) bindirilmiştir.
- **Durum:** Üretildi (`PYTORCH_ENABLE_MPS_FALLBACK=1 python qualitative_examples.py`, ~2 dk). Rastgele 600 SUN görüntüsünün sonucu: MCM kabul edip GL-MCM reddeden 138 görüntü, MCM reddedip GL-MCM kabul eden 12 görüntü. Eşikler: MCM 0.0173, GL-MCM 0.0352 (COCO_single ID skorlarının 5. yüzdeliği). Yani bu örnek kümede GL-MCM'in kararı değiştirdiği görüntülerin çoğunda OOD'yi doğru reddediyor (138'e 12); bu 600 görüntülük bir örnektir, genel oran değildir.
- **Nasıl okunur:** Her karenin başlığında MCM ve GL-MCM skoru ve eşiği yazılı; "global: ... | local: ..." satırı global olarak ve yerel olarak en çok eşleşen COCO sınıfını verir. Renkli katman 14x14 yerel skor haritasıdır; renk ölçeği her görüntüde ayrı ayrı normalize edilmiştir (parlak = bir sınıfla en iyi eşleşen yama).
- **Dikkat:** (1) τ=1'de softmax skorları 1/60 civarında olduğu için MCM ve GL-MCM değerleri eşiğe çok yakındır (fark 3.-4. ondalıkta); örnekler "sınırda" kararlardır. (2) Yerel haritalar gürültülüdür ve anlamlı bir nesne konturu çıkarmaz (MaskCLIP tarzı yerel özelliğin bilinen özelliği olabilir; test edilmedi); haritaları "model nesneyi buldu" diye yorumlamayın. (3) Bunlar kanıt değil, görsel örnektir. (4) Sol üstteki ilk örneklerde bile "global: remote" gibi anlamsız sınıflar görünür; çünkü OOD görüntülerde en yakın ID sınıfı zaten anlamsızdır.

## C. Hocanın isteklerine karşılık gelen şekil/tablo eşlemesi
| İstek | Şekil / tablo |
|---|---|
| Kavram 1 var/yok kıyası, bol görsel | fig01, fig02, fig05 (6 şekil), fig06 (3 şekil), fig09, (fig12-13) |
| Kavram 2 (sıcaklık) var/yok kıyası | fig03, §6.4 tabloları, makale Tablo 3 kıyası |
| Ek: λ etkisi | fig04, `tables/lambda_sweep.md` |
| "Ranking quality" açıklaması | fig05 ROC/PR + PROJE_ACIKLAMASI §2.3 |
| Makale tablosu, uygulanan/uygulanmayan yöntemler | fig08 + §6.6 tablosu |
| Makaleyle kıyas | fig07 + §6.1-6.3 (makale satırlı tablolar) |
| Baseline'lar | fig10, fig11, §6.6 |
| Kodun her bölümünün açıklaması | PROJE_ACIKLAMASI §4 |
| Çıktının her bölümünün açıklaması | bu rehber (B bölümü) + `results/` CSV'leri + terminal çıktıları (D bölümü aşağıda) |

## D. Terminal çıktısı nasıl okunur (raporda "çıktının her bölümü" için)
`eval_all_scores.py` çalışırken: ilerleme çubuğu (`tqdm`: yüzde, işlenen batch/toplam batch, saniye/batch), `=== OOD set: <ad>` (hangi OOD kümesi taranıyor), `[cache] <yol>` (bu parça daha önce hesaplanmış, diskten okundu), `[burn] ...` (rastgele alt küme çekimi ilerletildi, görüntü işlenmedi), her OOD seti bittikten sonra `--- T=<τ>, lambda=<λ>: results so far (<n> OOD sets: ...)` ve ardından sonuçların yazıldığı klasörler. CSV'nin sütunları: ilk sütun OOD seti adı, sonra FPR95, AUROC, AUPR (yüzde); son satır AVG.
`baselines_imagenet.py` çıktısı: `--- MSP: FPR95 xx.xx  AUROC xx.xx  (n OOD sets)` satırları o ana kadar bitmiş OOD setlerinin **kümülatif ortalamasını** verir (set bazlı değildir; set bazlı sayılar CSV'dedir).
