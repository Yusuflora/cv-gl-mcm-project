# GL-MCM projesi: baştan sona açıklama (COME421 Midterm)

> Bu dosya, projeyi hiç görmemiş birinin (ya da onun yapay zeka asistanının) **ödev raporunu yazabilmesi** için hazırlandı.
> Önce her şey sade dille anlatılır, sonra ayrıntıya girilir. Sayıların tamamı bu repodaki dosyalardan gelir (`results/`, `tables/`, `docs/RESULT_TABLES.md`).
> Yardımcı dosyalar: `docs/DOSYA_VE_GORSEL_REHBERI.md` (her dosya ve her grafik ne), `docs/KARARLAR_VE_NOTLAR.md` (verilen kararların günlüğü), `docs/AGENT_PROMPT.md` (yapay zekaya verilecek talimat), `DATASETS.md` (veri setleri).

---------------------------------------------------------------------------------------------------

## 0. En kısa özet (5 cümle)

1. Elimizde bir fotoğrafa bakıp "bu, tanıdığım türden bir fotoğraf mı (ID), yoksa garip / tanımadığım bir şey mi (OOD, out-of-distribution)" diyen bir yöntem ailesi var. Buna **OOD detection** (dağılım dışı örnek tespiti) denir.
2. Seçtiğimiz makale (**GL-MCM**, IJCV 2025) bunu, önceden eğitilmiş **CLIP** modeliyle ve ek bir eğitim yapmadan (**zero-shot**) yapıyor; sadece sınıf adları ("cat", "dog" ...) veriliyor.
3. Üç yöntem var: **MCM** (fotoğrafa bütün olarak bakar), **L-MCM** (fotoğrafın küçük yamalarına bakar), **GL-MCM** (ikisini toplar).
4. Biz bu yöntemleri sıfırdan çalıştırdık (COCO, VOC, ImageNet-1K), makaledeki sayıları COCO ve ImageNet'te neredeyse birebir, VOC'ta küçük sapmalarla yeniden ürettik ve hocanın istediği iki "var / yok" kıyasını yaptık: (1) yerel özellik yok / var, (2) sıcaklık (temperature) ölçekleme yok / var.
5. Ayrıca makalenin ImageNet tablosundaki MSP, Energy ve ODIN yöntemlerini de çalıştırdık; ViM, KNN, NPOS, CoOp, LoCoOp eğitim gerektirdiği için **uygulamadık** (makaledeki değerler raporda bu şekilde etiketlenir).

---------------------------------------------------------------------------------------------------

## 1. Ödev nedir? (resmi metin ve hocanın beklentisi)

**Ders:** COME421/1 Introduction to Computer Vision, Üsküdar Üniversitesi. Hoca: Asst. Prof. Dr. Gamze Uslu. Midterm notun %45'i; final %55'i (finalde aynı makalenin devamı gelecek).

**Resmi ödev metni (özet çeviri):**
- Seçtiğiniz makalede karşılaştırılan yöntemlerin **Python uygulamasını** verin. Bu yöntemleri makalenin kullandığı **veri setlerinde** çalıştırın. Sonuçlarınızı makalenin kullandığı **aynı metriklerle** karşılaştırın. Sonuçları **tablo ve grafiklerle** verin.
- Kodunuzun **her bölümünü** ve kodun **çıktısının her bölümünü** açıklayan bir **rapor** yazın.
- Tek bir zip/rar dosyası olarak, STIX'teki ödev linkinden yükleyin. Son tarih: **19.11.2026 01:00** (yani 18 Kasım gecesi). Geç teslim kabul edilmez; link dışı teslim kabul edilmez.
- Grup çalışması: gruptan tek kişi yükler. Raporun kapak sayfasında **her üyenin ad, soyad ve numarası** bulunmalı; teslimden sonra üye eklenemez; kapakta olmayan üye 0 alır.
- Dosya sınırı: dosya başına en fazla 50 MB.

**Hocanın sözlü geri bildirimleri (ilerleme raporumuz üzerine; kullanıcının aktarımıyla):**
1. "Ranking quality" skorunun ne olduğunu açıklayın. (Cevap: AUROC. Raporda AUROC'un anlamı anlatılmalı; AUPR de var; FPR95 ise tek bir eşikteki hatadır.)
2. Makaledeki karşılaştırma tablosunu gösterin ve **hangi yöntemleri uyguladığınızı, hangilerini uygulamadığınızı** belirtin.
3. Makalenin deneysel sonuçlar tablosundaki yöntemleri uygulayın.
4. Hocanın **temel isteği**: seçilen makaleden **2 kavram** seçin ve bunların **kullanıldığı / kullanılmadığı** durumlar için **öncesi-sonrası** kıyası yapın; kıyası **ayrıntılı** yapın ve **bolca görselleştirilmiş çıktı** verin. (Hoca en çok buna dikkat ediyor.)

**Bizim seçtiğimiz iki kavram** (ablation, yani "bir parçayı çıkarınca ne oluyor" analizi):
1. **Yerel (local) özellik kullanımı:** yok = MCM, var = GL-MCM. (L-MCM = sadece yerel, ek referans.)
2. **Sıcaklık ölçekleme (temperature scaling, τ):** yok = τ=1, var = τ=0.01.
- Ek analiz: yerel skorun ağırlığı **λ** taraması (λ=0 tam olarak MCM'dir).
- Dil uyarısı: "yöntemi geliştirdik" demeyin. Doğrusu: "makalenin yöntemini uyguladık ve iki bileşenin etkisini ablation olarak ölçtük."

---------------------------------------------------------------------------------------------------

## 2. Makale ne yapıyor? (önce sade, sonra teknik)

### 2.1 Sade anlatım
Bir "bekçi" düşünün. Ona sadece şunu söylüyoruz: "Burada kedi, köpek, at, sandalye gibi 60 şey tanıyorsun." Sonra bir fotoğraf gösteriyoruz. Fotoğraf bu listedekilerden birine benziyorsa bekçi "tanıdım" (ID), hiçbirine benzemiyorsa "yabancı" (OOD) demeli. Bekçi için ayrıca eğitim yok; kullanılan model **CLIP**, internetten toplanmış milyonlarca fotoğraf-yazı çiftiyle önceden eğitilmiş. CLIP hem fotoğrafı hem metni aynı "anlam uzayına" çevirir; fotoğraf ile "a photo of a cat" yazısı birbirine yakınsa fotoğraf kedidir.

- **MCM:** fotoğrafın tamamına bakar, en çok benzediği sınıfın benzerlik puanını alır (yüksek puan = tanıdık).
- **L-MCM:** fotoğrafı 14x14 = 196 küçük yamaya böler, her yamanın en çok benzediği sınıfa bakar ve en yüksek puanı alır.
- **GL-MCM:** ikisini toplar: `GL-MCM = MCM + λ · L-MCM`.
- Neden yerel? Gerçek fotoğraflarda (COCO, VOC) bir tanıdık nesne ile bir yabancı nesne aynı karede olabilir. Bütüne bakınca yabancı nesne gözden kaçar; yamalara bakınca yakalanır.

### 2.2 Teknik anlatım
- `f(x)`: görüntü özelliği (CLIP görüntü kodlayıcısı), `g(c_i)`: i. sınıfın metin özelliği ("a photo of a {class}"). Özellikler L2 normalize edilir; benzerlik `s_i = cos(f(x), g(c_i))`.
- **MCM** (global): `S_MCM(x) = max_i exp(s_i/τ) / Σ_j exp(s_j/τ)`. τ=1 iken softmax neredeyse düzdür (örn. 60 sınıfta değerler ≈ 1/60 ≈ 0.0167 civarındadır); bu yüzden skor histogramlarında değerler çok küçük görünür. Sıralama yine bilgilidir.
- **L-MCM** (yerel): her yama özelliği `f_local(x)[h,w]` için aynı softmax alınır (sınıflar üzerinde) ve tüm yamalar ile sınıflar üzerinde max alınır: `S_L-MCM(x) = max_{h,w,i} softmax_i(cos(f_local[h,w], g(c_i))/τ)`.
- **GL-MCM:** `S_GL-MCM(x) = S_MCM(x) + λ · S_L-MCM(x)`. Makalenin ayarı: λ=1.0 (COCO, VOC), λ=0.5 (ImageNet), τ=1.
- **Karar:** skor eşiğin üstündeyse ID, altındaysa OOD. Metrikler eşikten bağımsız (AUROC, AUPR) ya da belirli bir eşikte (FPR95) hesaplanır.
- Kodda yerel özellik (`clip/model.py`): `VisionTransformer.forward` iki şey döndürür: (1) **global özellik** = CLS belirteci (`ln_post` + projeksiyon), (2) **yerel özellik haritası** (14x14 ızgara, ViT-B/16 ve 224x224 giriş). Yerel harita şöyle üretilir: son Transformer bloğunda (`ResidualAttentionBlock.forward`) her yama belirteci için dikkat (attention) karışımı **atlanır**; yama kendi `value` vektörünü `out_proj`'tan geçirir, artık bağlantı (`v += x`) ve MLP uygulanır, sonra `ln_post` ve görüntü-metin projeksiyonu gelir. Yani her yama, tüm görüntüyle karışmadan kendi yerel içeriğini metin uzayına taşır (MaskCLIP tarzı bir yaklaşım; makalenin kendi anlatımına bakın).

### 2.3 Metrikler (hocanın "ranking quality" sorusu burada)
Tüm metriklerde **ID = pozitif sınıf**, yüksek skor = daha çok ID.
- **AUROC** (ROC eğrisinin altındaki alan, %): ranking quality. "Rastgele seçilen bir ID görüntüsünün skoru, rastgele seçilen bir OOD görüntüsünün skorundan yüksek olma olasılığı". Eşikten bağımsızdır. %50 = yazı-tura, %100 = mükemmel. **Yüksek iyi.**
- **FPR95** (FPR at 95% TPR, %): ID görüntülerinin %95'ini doğru kabul edecek eşik seçilir; bu eşikte OOD görüntülerinin yüzde kaçı yanlışlıkla ID sanılıyor? Tek bir çalışma noktasıdır, eşiğe duyarlıdır. **Düşük iyi.**
- **AUPR** (Precision-Recall eğrisinin altındaki alan, %): makale raporlamıyor, kodumuz raporluyor. ID çoğunluk sınıfı olduğundan değerler yüksek çıkar (özellikle ImageNet'te ~98). **Yüksek iyi.**
- Raporlanan "AVG": OOD setleri üzerinden basit ortalama.

---------------------------------------------------------------------------------------------------

## 3. Yaptığımız her adım (sırayla)

> Tarihler 2026 Kasım'dandır (git geçmişinden: ilk fork 5 Kasım, ilk ilerleme raporu 5 Kasım, güncelleme 12 Kasım; nihai sonuçlar, şekiller ve açıklama dosyaları 17 Kasım'da GitHub'a gönderildi). Hesaplamalar Apple Silicon MacBook'larda (MPS) yapıldı.

**Adım 1 - Makale.** Ödevde kullanılan IJCV makalesi: Miyai, Yu, Irie, Aizawa, "GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot Out-of-Distribution Detection", IJCV 2025 (arXiv 2304.04521). Yazarların kodu GitHub'da `AtsuMiyai/GL-MCM` (kod, MCM projesinin kodunu temel alıyor).

**Adım 2 - Depo ve ortam.** Yazarların kodu bizim depomuza (`Yusuflora/cv-gl-mcm-project`) fork edildi. Yazarların kodu NVIDIA GPU (CUDA) için yazılmış; biz Mac'te çalışabilmesi için `cuda` yerine Apple **MPS** kullanacak şekilde yamaladık (`eval_ood_detection.py`: CUDA kontrolü kaldırıldı; `utils/train_eval_util.py` ve `utils/detection_util.py`: `.cuda()` yerine `.to("mps")`). Ortam `uv` ile Python 3.12 olarak kuruldu (yazarlar Python 3.8 + PyTorch 1.10 ve A100/V100 GPU ile çalışmış; biz daha yeni sürümlerde çalıştırdık, sonuçlar yine makaleyle uyuştu). Orijinal kodda zaten olan `--lambda_local` (λ, varsayılan 1.0) bayrağına ek olarak tek bir ekleme yaptık: `--out_list` (hangi OOD setlerinin koşacağını seçmek). `.gitignore`'a `.venv/` ve `datasets/` eklendi (ilk denemede `results/` de ignore'daydı; nihai sonuçlar depoda tutulsun diye 17 Kasım'da çıkarıldı). Ayrıca `utils/plot_util.py` içinde her grafikten sonra `plt.close('all')` çağrısı eklendi (bellek sızıntısını önlemek için).

**Adım 3 - İlk deneme koşuları (M5 bilgisayar, 5 Kasım).** COCO_single (ID) ve `ood_voc` (OOD) üzerinde MCM, L-MCM ve GL-MCM çalıştırıldı. τ=1 sonuçları makaleyle uyumluydu (ood_voc için FPR95/AUROC: MCM 77.48/84.42, L-MCM 72.10/84.21, GL-MCM 71.67/87.71; makale 77.62/84.43, 72.02/84.21, 71.62/87.72; ilk koşunun AUPR değerleri: 89.53, 87.43, 91.66). Aynı gün bir **ilerleme raporu** PDF'i (`results/GL-MCM_progress_report.pdf`, 5 sayfa) depoya eklendi; 12 Kasım'da "before/after" ve "bizim sonuç - makale" grafikleriyle güncellendi (`results/before_after_MCM_vs_GL-MCM.png`, `results/ours_vs_article.png`; commit bdb9564). Bu ilk dosyalar yalnızca COCO_single / `ood_voc` içindir; nihai sonuçlar aşağıdaki adımlarla üretildi.

**Adım 4 - Hocanın geri bildirimi** (bölüm 1'deki dört madde): ranking quality'yi açıkla, makale tablosunu göster ve hangi yöntemleri uyguladığını belirt, deneysel sonuç tablosundaki yöntemleri uygula, 2 kavramı var/yok diye kıyasla ve bolca görsel ver.

**Adım 5 - Makine değişikliği.** Çalışma M5 bilgisayardan M1 MacBook Air'e (8 GB RAM) taşındı. Doğrulama: M1'de aynı test koşusu (COCO → `ood_voc`, τ=1; sonuç klasörleri `..._ID_m1test`) M5'teki ilk koşuyla **birebir aynı** sonucu verdi.

**Adım 6 - Veri setlerinin toplanması.** (ayrıntı: `DATASETS.md`)
- Yazarların yayımladığı paket: COCO_single (ID, 5000 görüntü), COCO_multi (ID, 5000), VOC_single (ID, 1000), `OOD_COCO` (1000), `OOD_VOC` (4000).
- iNaturalist, SUN, Places: 10.000'er görüntülük alt kümeler (Huang ve ark. 2021'in ImageNet-1K ile çakışan sınıfları çıkarılmış alt kümeleri).
- Texture: DTD (Describable Textures), 47 sınıf, 5.640 görüntü.
- ImageNet-22K: 18.335 görüntülük OOD alt kümesi (Wang ve ark. 2021'in multi-label-ood deposundan).
- ImageNet-1K doğrulama seti (50.000 görüntü, 1000 sınıf): Hugging Face'teki `ILSVRC/imagenet-1k` (hesap ve şart kabulü gerekir), parquet'ten klasörlere çevrildi (`convert_imagenet_val.py`; klasör adları 000-999 sıfır dolgulu, böylece sıralama sınıf numarasıyla birebir aynı).

**Adım 7 - Tek geçişli değerlendirme scripti (`eval_all_scores.py`).** Yazarların kodu her skor için (MCM, L-MCM, GL-MCM) CLIP'i her görüntüde yeniden çalıştırıyor. Üç skor aslında aynı CLIP çıktısının farklı birleşimi olduğundan, biz CLIP'i **bir kez** çalıştırıp ham global ve yerel "max softmax" değerlerini `results_raw/` içine kaydettik. Böylece üç skor, iki sıcaklık (τ=1, 0.01) ve her λ değeri hiç yeni hesaplama gerektirmeden bu ham değerlerden çıkıyor (3x hızlanma + bedava ablation).

**Adım 8 - Bellek önlemleri.** M1'in belleği 8 GB olduğu için büyük koşulara bellek önlemleriyle başlandı (kayıtlarda bir donma olayı yazılı değil, önlem amaçlıdır): batch 32, 2 veri yükleme işçisi, her adımdan sonra MPS bellek temizliği (`torch.mps.empty_cache()`).

**Adım 9 - CLIP koşuları (M1).** Hepsi τ ∈ {1, 0.01}, sonuçlar her OOD setinden sonra yazılır (kesilse de kayıp olmaz):
- COCO_single (ID 5000; her OOD seti 5000 rastgele görüntü, `ood_voc` 4000): iNaturalist 15:53 dk, SUN 14:15, Texture 8:33, IN22k 9:38. 
- VOC_single (ID 1000, her OOD 1000): her set yaklaşık 1:35-1:55 dk, toplam ~10 dk.
- ImageNet-1K (λ=0.5; ID 50.000 tam; OOD iNaturalist 10.000, SUN 10.000, Places 10.000, Texture 5.640 tam): ID 1 sa 42 dk; iNaturalist 25 dk, SUN 24 dk, Places 25.5 dk, Texture 14 dk.
- Rastgele alt kümelerin birden çok çalıştırmada aynı kalması için `--burn` bayrağı eklendi (önceki çağrıların rastgele sayı üretimlerini ilerletir).

**Adım 10 - Baseline'lar (ImageNet).** Makalenin ImageNet tablosundaki MSP, Energy ve ODIN yöntemleri `baselines_imagenet.py` ile yazıldı (ayrıntı bölüm 4.3 ve 7).
- İlk deneme: torchvision ViT-B/16 (IMAGENET1K_V1; yalnızca ImageNet-1K ile, yoğun düzenlileştirmeyle eğitilmiş). MSP ve Energy makaleden belirgin kötü çıktı (MSP FPR95 61.20, Energy 65.84); ODIN bu modelde ~29 sn/16 görüntü hızıyla çok yavaştı, yarıda durdurduk.
- Makale kontrolü: makale bu baseline sayılarını **kendisi üretmemiş**, başka makalelerden almış (MSP/Energy: Ming ve ark. 2022; ODIN, ViM, KNN, NPOS: Tao ve ark. 2023) ve hangi modeli kullandığını açıkça yazmıyor ("ImageNet-1K üzerinde fine-tune edilmiş", MSP satırı "ViT-B"). O çizgideki Fort ve ark. 2021 çalışması ViT-B/16'yı ImageNet-21k ile önceden eğitip kendi ID veri setinde fine-tune ediyor.
- Düzeltme: ImageNet-21k ön eğitimli, ImageNet-1K'da fine-tune edilmiş ViT-B/16 (`timm`: `vit_base_patch16_224.orig_in21k_ft_in1k`) kullanıldı. MSP ve Energy makaleye yaklaştı (bölüm 6).
- Süre kısıtı nedeniyle alt kümeler: MSP ve Energy için ID 10.000 / her OOD seti 3.000; ODIN için ID 2.000 / her OOD seti 500.

**Adım 11 - Tablolar ve grafikler.** `make_tables.py` ve `make_figures.py` ile tüm tablolar ve 11 grafik dosyası grubu (fig01-fig11; toplam 18 PNG) üretildi; nitel örnekler (fig12-fig13) ayrıca `qualitative_examples.py` ile üretildi (toplam 20 PNG). Grafik kodu, metrikleri ham skorlardan yeniden hesaplar ve resmi CSV'lerle uyuştuğunu (en fazla 0.008 puan fark) doğrular. `qualitative_examples.py` gerçek görüntülerden örnek ve yerel skor haritası figürleri üretir.

**Karşılaşılan sorunlar ve çözümleri (raporda "zorluklar" bölümüne uygun):**
1. CUDA yerine MPS: kod yaması.
2. 8 GB RAM sınırı: küçük batch, az veri yükleme işçisi, her batch sonunda MPS bellek temizliği.
3. Bazı veri sunucuları çok yavaş: kesintiye dayanıklı (`curl -C -`) arka plan indirmeleri.
4. Aynı rastgele alt kümeyi korumak: `--burn` bayrağı.
5. Baseline modeli seçimi: torchvision → timm geçişi (yukarıda).
6. ODIN'in hızı: geri-yayılım yüzünden küçük alt küme.

---------------------------------------------------------------------------------------------------

## 4. Kod haritası (raporda "kodun her bölümünü açıkla" kısmı için)

### 4.1 `eval_all_scores.py` (CLIP, MCM / L-MCM / GL-MCM, tek geçiş)
Bölümler (yukarıdan aşağı):
1. **Başlık ve importlar:** modül açıklaması; `clip` (değiştirilmiş CLIP, global + yerel özellik döndürür), `utils.*` yardımcıları.
2. **`parse()`:** komut satırı argümanları: `--in_dataset` (COCO_single, COCO_multi, VOC_single, ImageNet), `--root-dir` (veri klasörü, varsayılan `./datasets`), `--name` (çıktı klasörü etiketi, varsayılan `all`; biz `final` kullandık), `--seed` (1), `-b` batch (32), `--workers` (2), `--CLIP_ckpt` (ViT-B/16, RN50, RN101), `--Ts` (virgülle ayrılmış sıcaklıklar, ör. `1,0.01`), `--lambda_local` (verilmezse 1.0; ImageNet için 0.5), `--out_list` (hangi OOD setleri), `--num_ood_sumple` / `--num_id_sample` (rastgele alt küme boyutu, -1 = hepsi), `--burn`, `--model` (yalnızca çıktı yolundaki etiket, varsayılan CLIP). Sabit `DEVICE = "mps"`.
3. **`tkey(T)`:** sıcaklığı dosya/anahtar adına çevirir (1 → "1", 0.01 → "0.01").
4. **`extract(net, loader, text_feats, Ts)`:** asıl hesap. Her batch için: görüntüyü CLIP'ten geçirir, global ve yerel özelliği alır, L2 normalize eder, metin özellikleriyle çarpar (`og = gf @ text.T`; `ol = lf @ text.T`), her sıcaklık için softmax alır (global: sınıflar üzerinde; yerel: her yama için sınıflar üzerinde), global için sınıf üzerinde max, yerel için tüm 196 yama ve sınıflar üzerinde max alır (`flatten(1).max`). Çıktı: her T için iki dizi (`g_<T>`, `l_<T>`). Bellek için her batch sonunda tensörleri siler ve MPS belleğini boşaltır.
5. **`cached(path, fn)`:** sonuç `.npz` dosyası varsa diskten okur, yoksa `fn()` ile hesaplayıp kaydeder (kesilen koşular kaldığı yerden devam eder).
6. **`scores_from_raw(raw, T, lam)`:** ham değerlerden üç skoru üretir: MCM = g, L-MCM = l, GL-MCM = g + λ·l.
7. **`write_outputs(args, id_raw, ood_raw)`:** her T ve her skor için her OOD setinde `get_measures` ile FPR95, AUROC, AUPR hesaplar; CSV (`save_as_dataframe`, son satır AVG) yazar; skor dağılımı PNG'lerini (`plot_distribution`, orijinal koddaki gibi işaret çevrilmiş skorlarla) yalnızca ilk sıcaklık için yazar. Her OOD seti bittikten sonra çağrıldığı için kesilen bir koşuda bile o ana kadarki sonuçlar diskte olur. Çıktı yolu: `results/<ID veri seti>/<SKOR>/CLIP_ViT-B_16_T_<T>_ID_<name>/`.
8. **`main()`:** modeli yükler; metin özelliklerini bir kez üretir ("a photo of a {sınıf}"); ID özelliklerini çıkarır (önbellekli); `--burn` için rastgele alt küme çekimlerini ilerletir; her OOD seti için özellik çıkarır, kaydeder ve çıktıları yeniler.

### 4.2 Yazarların orijinal dosyaları (bizim yamalarımızla)
- `eval_ood_detection.py`: yazarların ana betiği; skorları tek tek hesaplar (bizim betik bunun hızlı hâli). Bizim yamalarımız: CUDA kontrolü kaldırıldı, `--out_list` eklendi (`--lambda_local` orijinalde vardı).
- `clip/model.py`, `clip/*`: OpenAI CLIP'in değiştirilmiş hâli; `VisionTransformer.forward` global ve yerel özellik döndürür (bölüm 2.2).
- `utils/detection_util.py`: `get_measures` (AUROC, AUPR, FPR95; `fpr_and_fdr_at_recall`), `get_ood_scores_clip` (orijinal skor hesabı), `print_measures`.
- `utils/train_eval_util.py`: `set_model_clip` (CLIP yükler, ön işleme), `set_val_loader` (ID yükleyici), `set_ood_loader_ImageNet` (OOD yükleyici; alt küme seçimi `get_subset_with_len` ile `np.random.shuffle`).
- `utils/common.py`: sınıf adı listeleri (COCO_single 60 sınıf, COCO_multi 58, VOC_single 14, ImageNet 1000; ImageNet adları `data/ImageNet/imagenet_class_clean.npy`; kodla sayıldı), `setup_seed`, `get_test_labels`.
- `utils/plot_util.py`: skor dağılımı histogramı (yamamız: `plt.close('all')`, bellek sızıntısını önlemek için).
- `utils/file_ops.py`: CSV yazımı (`save_as_dataframe`).
- `scripts/eval_*.sh`: yazarların örnek çalıştırma komutları (COCO'da OOD başına 5000, VOC'ta 1000 örnek).

### 4.3 `baselines_imagenet.py` (MSP, Energy, ODIN)
1. Ağırlıklar: `--arch tv` (torchvision ViT-B/16) veya `--arch timm` (`vit_base_patch16_224.orig_in21k_ft_in1k`, 21k ön eğitim + 1k fine-tune).
2. `make_loader`: ImageFolder + rastgele alt küme (sabit tohumlu permütasyonun ilk k elemanı; böylece küçük alt küme büyük olanın içinde kalır).
3. `extract`: bir batch için: `logits = model(x)`. **MSP** = `max softmax(logits)`. **Energy** = `logsumexp(logits)` (T=1). **ODIN** = girdiye küçük bir bozucu ekleyip (`T=1000`, `eps=0.0014`, gradyanın işareti, kanal standart sapmasına bölünür) yeniden `max softmax(logits/T)` alır.
4. `cached`, `write_outputs`: CLIP betiğiyle aynı mantık; sonuçlar `results/ImageNet/<MSP|Energy|ODIN>/<model etiketi>_ID_<run>/`.
5. ViM ve KNN **uygulanmadı** (ImageNet eğitim seti özellikleri gerekir).

### 4.4 Diğer dosyalar
- `convert_imagenet_val.py`: Hugging Face parquet dosyalarını `datasets/ImageNet/000..999/` klasörlerine yazar.
- `make_tables.py`: `results/**.csv` → `docs/RESULT_TABLES.md`, `tables/all_results_long.csv`; makale değerleri (bu dosyada elle girilmiş, kaynak: makalenin tabloları) ile yan yana kıyas.
- `make_figures.py`: tüm grafikler; metrik kodunu (numpy) resmi CSV'lerle doğrular.
- `qualitative_examples.py`: gerçek görüntü örnekleri ve yerel skor haritası (CLIP gerektirir).

---------------------------------------------------------------------------------------------------

## 5. Deney kurulumu özeti
| Madde | Değer |
|---|---|
| Model | CLIP ViT-B/16 (zero-shot); baseline'lar: ViT-B/16 (timm, 21k→1k) |
| ID veri setleri | COCO_single (60 sınıf, 5000 görüntü), VOC_single (14 sınıf, 1000 görüntü), ImageNet-1K val (1000 sınıf, 50.000 görüntü) |
| OOD (COCO ID) | iNaturalist, SUN, Texture, IN22k (rastgele 5000'er), ood_voc (4000, tamamı) |
| OOD (VOC ID) | iNaturalist, SUN, Texture, IN22k (rastgele 1000'er), ood_coco (1000, tamamı) |
| OOD (ImageNet ID) | iNaturalist 10.000, SUN 10.000, Places 10.000, Texture 5.640 (tamamı) |
| λ | 1.0 (COCO, VOC), 0.5 (ImageNet); λ taraması 0-3 |
| τ | 1 (ana), 0.01 (ablation) |
| Rastgele tohum | 1 (alt küme seçimi) |
| Metrikler | FPR95 (↓), AUROC (↑), AUPR (↑); ID pozitif sınıf |
| Donanım | Apple M1, 8 GB RAM, MPS (önceki denemeler: M5, 32 GB) |

---------------------------------------------------------------------------------------------------

## 6. Sonuçlar (hepsi `docs/RESULT_TABLES.md` ve `tables/all_results_long.csv` içinde; sayılar CSV dosyalarından üretilmiştir)

Her hücre: FPR95 / AUROC. FPR95 düşük iyi, AUROC yüksek iyi. AVG = OOD setleri üzerinden ortalama. "Makale" satırları makalenin tablolarından elle girilmiştir (`make_tables.py` içinde `ART` sözlüğü).

### 6.1 COCO_single (τ=1, λ=1)
| Yöntem | iNaturalist | SUN | Texture | IN22k | ood_voc | AVG |
|---|---|---|---|---|---|---|
| MCM (bizim) | 64.06 / 90.76 | 84.08 / 81.79 | 65.00 / 86.64 | 78.38 / 84.15 | 77.48 / 84.42 | 73.80 / 85.55 |
| MCM (makale) | 64.42 / 90.75 | 84.18 / 81.81 | 65.10 / 86.62 | 78.50 / 84.14 | 77.62 / 84.43 | 73.96 / 85.55 |
| L-MCM (bizim) | 41.76 / 92.46 | 51.76 / 89.33 | 85.72 / 74.89 | 75.60 / 77.92 | 72.10 / 84.21 | 65.39 / 83.76 |
| L-MCM (makale) | 41.80 / 92.46 | 51.52 / 89.24 | 85.68 / 74.87 | 75.50 / 77.92 | 72.02 / 84.21 | 65.30 / 83.74 |
| GL-MCM (bizim) | 34.90 / 94.46 | 62.32 / 88.53 | 72.06 / 84.55 | 70.74 / 84.49 | 71.67 / 87.71 | 62.34 / 87.95 |
| GL-MCM (makale) | 34.72 / 94.45 | 61.96 / 88.49 | 72.08 / 84.53 | 70.66 / 84.49 | 71.62 / 87.72 | 62.21 / 87.94 |

### 6.2 VOC_single (τ=1, λ=1) - ID yalnızca 1000 görüntü, sapma daha büyük
| Yöntem | iNaturalist | SUN | Texture | IN22k | ood_coco | AVG |
|---|---|---|---|---|---|---|
| MCM (bizim) | 11.70 / 97.98 | 31.50 / 94.11 | 53.80 / 91.89 | 56.30 / 90.29 | 60.00 / 88.53 | 42.66 / 92.56 |
| MCM (makale) | 8.20 / 98.23 | 28.60 / 94.68 | 51.70 / 91.45 | 51.40 / 90.94 | 54.50 / 89.02 | 38.88 / 92.86 |
| L-MCM (bizim) | 28.80 / 95.06 | 45.60 / 88.53 | 60.70 / 85.78 | 53.10 / 87.42 | 59.20 / 84.04 | 49.48 / 88.17 |
| L-MCM (makale) | 27.70 / 94.97 | 44.20 / 88.87 | 60.00 / 86.23 | 54.10 / 86.27 | 57.00 / 84.06 | 48.72 / 88.08 |
| GL-MCM (bizim) | 4.40 / 98.63 | 23.50 / 94.17 | 40.10 / 92.81 | 39.10 / 92.38 | 44.50 / 90.28 | 30.32 / 93.65 |
| GL-MCM (makale) | 4.20 / 98.71 | 23.10 / 94.66 | 43.00 / 92.84 | 41.00 / 92.38 | 44.30 / 90.48 | 31.12 / 93.81 |

### 6.3 ImageNet-1K (CLIP, τ=1, λ=0.5) - ID 50.000 tam, OOD setleri tam
| Yöntem | iNaturalist | SUN | places365 | Texture | AVG |
|---|---|---|---|---|---|
| MCM | 30.70 / 94.63 | 37.39 / 92.57 | 43.66 / 89.96 | 57.29 / 86.18 | 42.26 / 90.83 |
| L-MCM | 48.98 / 87.00 | 73.32 / 76.66 | 79.10 / 71.93 | 92.57 / 59.61 | 73.49 / 73.80 |
| GL-MCM | 17.50 / 96.44 | 30.79 / 93.45 | 37.67 / 90.64 | 55.34 / 85.54 | 35.33 / 91.52 |
(Makale ortalamaları: MCM 42.74 / 90.77, L-MCM 73.59 / 73.73, GL-MCM 35.25 / 91.51. Makale ImageNet için set bazlı sayıları bu tabloda göstermiyor olabilir; biz yalnızca ortalamayı kıyaslıyoruz.)

### 6.4 İki kavramın öncesi-sonrası (özet)
**Kavram 1 - Yerel özellik (yok = MCM, var = GL-MCM; L-MCM referans).** Ortalama farklar (GL-MCM eksi MCM):
| ID veri seti | ΔFPR95 | ΔAUROC |
|---|---|---|
| COCO_single | 62.34 - 73.80 = **-11.5** (iyileşme) | 87.95 - 85.55 = **+2.4** |
| VOC_single | 30.32 - 42.66 = **-12.3** | 93.65 - 92.56 = **+1.1** |
| ImageNet-1K | 35.33 - 42.26 = **-6.9** | 91.52 - 90.83 = **+0.7** |

Set bazında istisnalar (rapor bunları gizlememeli): COCO Texture'da GL-MCM, MCM'den kötü (FPR95 65.00 → 72.06, AUROC 86.64 → 84.55); ImageNet Texture'da AUROC çok az düşüyor (86.18 → 85.54; FPR95 ise 57.29 → 55.34 iyileşiyor); COCO SUN'da tek başına L-MCM, GL-MCM'den daha iyi FPR95 veriyor (51.76 vs 62.32). Yani yerel özellik her OOD setinde kazandırmıyor; ortalamada kazandırıyor. L-MCM tek başına tutarsız: COCO'da MCM'den iyi FPR95 (65.39 vs 73.80) ama düşük AUROC (83.76 vs 85.55), VOC'ta ve ImageNet'te belirgin kötü (ImageNet AVG 73.49 / 73.80). Makalenin bulgusuyla uyumlu: yerel skor tek başına zayıf, global ile birleşince en iyi.

*Yorum (test edilmemiş açıklama, rapor "olası neden" diye yazmalı):* ImageNet görüntüleri çoğunlukla ortada tek bir nesne içerir; yerel yamalar bu tür görüntülerde çok sayıda sınıfla eşleşebildiği için L-MCM tek başına ayırt edici değil. COCO_single görüntülerinde ID nesnesi küçük ve yanında başka nesneler vardır; yerel skor ID nesnesini daha iyi yakalar.

**Kavram 2 - Sıcaklık (τ=1 vs τ=0.01), ortalamalar FPR95 / AUROC:**
| ID | Yöntem | τ=1 | τ=0.01 |
|---|---|---|---|
| COCO | MCM | 73.80 / 85.55 | 73.12 / 84.08 |
| COCO | L-MCM | 65.39 / 83.76 | 66.92 / 82.99 |
| COCO | GL-MCM | 62.34 / 87.95 | 64.42 / 86.63 |
| VOC | MCM | 42.66 / 92.56 | 44.56 / 90.36 |
| VOC | L-MCM | 49.48 / 88.17 | 55.96 / 85.82 |
| VOC | GL-MCM | 30.32 / 93.65 | 42.34 / 91.25 |
| ImageNet | MCM | 42.26 / 90.83 | 56.01 / 83.82 |
| ImageNet | L-MCM | 73.49 / 73.80 | 79.97 / 67.93 |
| ImageNet | GL-MCM | 35.33 / 91.52 | 50.66 / 84.71 |

**Makalenin kendi τ ablation'ı (makale Tablo 3, GL-MCM; arXiv 2304.04521v4 HTML sürümünden kontrol edildi) ile kıyas:**
| ID | τ | Makale FPR95 / AUROC | Bizim FPR95 / AUROC |
|---|---|---|---|
| COCO | 0.01 | 64.89 / 86.70 | 64.42 / 86.63 |
| COCO | 1.0 | 62.21 / 87.94 | 62.34 / 87.95 |
| VOC | 0.01 | 43.92 / 90.80 | 42.34 / 91.25 |
| VOC | 1.0 | 31.12 / 93.81 | 30.32 / 93.65 |
| ImageNet-1K | 0.01 | 50.72 / 84.70 | 50.66 / 84.71 |
| ImageNet-1K | 1.0 | 35.25 / 91.51 | 35.33 / 91.52 |

Önemli: sıcaklık kıyası makalenin kendi ablation'ıdır (Tablo 3); makale yalnızca GL-MCM için verir, MCM ve L-MCM için τ=0.01 sonuçları bizim ek ölçümümüzdür. Makalenin yorumu: τ ile softmax ölçekleme ID/OOD ayrımı için önemlidir; τ=1.0 üç ID veri setinde de daha iyi. (Makale Tablo 3 değerleri `make_tables.py` içine `ART_TAU3` olarak eklendi ve `docs/RESULT_TABLES.md`'de de var. Makale sayıları arXiv HTML sürümünden alındı; rapor yazarken orijinal PDF'den bir kez daha doğrulayın.)

τ=1 neredeyse her durumda daha iyi (tek istisna: COCO MCM'de FPR95 τ=0.01'de hafifçe iyi, 73.12 vs 73.80; AUROC yine τ=1'de daha iyi). Kayda değer: makale τ=1 kullanıyor; küçük τ softmax'ı keskinleştirip skoru 1'e doğru itiyor ve ID/OOD farkını eziyor. (Bu açıklama genel mantıktır; deneysel olarak ayrı doğrulanmadı.)

**Ek - λ taraması (GL-MCM, τ=1; λ=0 tam olarak MCM):** (`tables/lambda_sweep.md`, fig04)
| λ | COCO FPR95 / AUROC | VOC | ImageNet |
|---|---|---|---|
| 0 | 73.80 / 85.55 | 42.66 / 92.56 | 42.26 / 90.83 |
| 0.1 | 71.75 / 86.28 | 40.08 / 93.03 | 39.86 / 91.14 |
| 0.25 | 68.33 / 87.05 | 33.62 / 93.48 | 37.46 / 91.42 |
| 0.5 | 65.56 / 87.71 | 30.12 / 93.78 | 35.32 / 91.51 |
| 0.75 | 63.26 / 87.93 | 30.10 / 93.79 | 34.74 / 91.30 |
| 1 | 62.34 / 87.95 | 30.32 / 93.65 | 35.23 / 90.90 |
| 1.5 | 61.52 / 87.72 | 32.18 / 93.25 | 37.07 / 89.80 |
| 2 | 61.90 / 87.41 | 33.58 / 92.81 | 39.30 / 88.58 |
| 3 | 62.01 / 86.82 | 36.76 / 92.04 | 44.34 / 86.30 |

COCO'da en iyi FPR95 λ=1.5'te (61.52), en iyi AUROC λ=1'de; VOC'ta en iyi λ≈0.5-0.75; ImageNet'te en iyi FPR95 λ=0.75 (34.74), en iyi AUROC λ=0.5. Makalenin λ seçimi (1.0 / 1.0 / 0.5) bu taramada en iyiye çok yakın (aşırı uyum yok; bizim tarama ID/OOD test kümesi üzerinde yapıldı, doğrulama kümesi ayrımı yok, bu yüzden "en iyi λ" kesin genelleme sayılmamalı).

### 6.5 AUPR (yalnızca bizim; makale AUPR vermiyor), τ=1, OOD ortalaması
| ID | MCM | L-MCM | GL-MCM |
|---|---|---|---|
| COCO_single | 88.37 | 84.09 | 89.57 |
| VOC_single | 93.32 | 87.99 | 94.08 |
| ImageNet | 98.18 | 93.59 | 98.27 |

### 6.6 Makalenin ImageNet tablosu: uyguladıklarımız ve uygulamadıklarımız (FPR95 / AUROC, 4 OOD seti ortalaması)
| Yöntem | Grup | Durum | Bizim | Makale | Not |
|---|---|---|---|---|---|
| MSP | Denetimli (post-hoc) | UYGULANDI | 44.63 / 88.52 | 45.12 / 88.25 | ID 10.000, her OOD 3.000 |
| Energy | Denetimli (post-hoc) | UYGULANDI | 30.54 / 93.52 | 35.92 / 92.26 | ID 10.000, her OOD 3.000 |
| ODIN | Denetimli (post-hoc) | UYGULANDI | 34.60 / 91.99 | 47.75 / 88.80 | ID 2.000, her OOD 500 |
| ViM | Denetimli (post-hoc) | uygulanmadı | - | 50.20 / 87.82 | ImageNet eğitim özellikleri gerekir |
| KNN | Denetimli (post-hoc) | uygulanmadı | - | 42.19 / 90.97 | ImageNet eğitim özellikleri gerekir |
| NPOS | Denetimli (post-hoc) | uygulanmadı | - | 37.93 / 91.22 | eğitim gerekir |
| MCM | Zero-shot CLIP | UYGULANDI | 42.26 / 90.83 | 42.74 / 90.77 | tam kümeler |
| L-MCM | Zero-shot CLIP | UYGULANDI | 73.49 / 73.80 | 73.59 / 73.73 | tam kümeler |
| GL-MCM | Zero-shot CLIP | UYGULANDI | 35.33 / 91.52 | 35.25 / 91.51 | tam kümeler, λ=0.5 |
| CoOp_MCM | İstem öğrenme | uygulanmadı | - | 36.83 / 91.93 | 16-shot eğitim gerekir |
| CoOp_GL-MCM | İstem öğrenme | uygulanmadı | - | 30.03 / 92.59 | 16-shot eğitim gerekir |
| LoCoOp_MCM | İstem öğrenme | uygulanmadı | - | 33.98 / 92.69 | 16-shot eğitim gerekir |
| LoCoOp_GL-MCM | İstem öğrenme | uygulanmadı | - | 28.68 / 93.59 | 16-shot eğitim gerekir |

Baseline ayrıntıları (bölüm 3 adım 10 ve `docs/KARARLAR_VE_NOTLAR.md` 10-13): 
- **Checkpoint etkisi (fig10):** torchvision ViT-B/16 (yalnızca 1K, yoğun düzenlileştirme) MSP 61.20 / 83.28, Energy 65.84 / 74.52 verdi; timm (21k ön eğitim + 1k fine-tune) MSP 44.63 / 88.52, Energy 30.54 / 93.52. Makale baseline sayılarını kendisi üretmemiş, başka makalelerden almış; hangi checkpoint olduğu açık değil. timm seçimi "en yakın makul tahminimiz" (Fort ve ark. 2021 çizgisiyle uyumlu), kesin eşleşme iddia edilmez.
- **Aynı alt kümede adil kıyas (fig11; ID 2.000 / OOD 4x500):** MSP 43.55 / 88.94, Energy 29.75 / 93.94, ODIN 34.60 / 91.99 (AUPR: 96.65, 98.21, 97.42). Bizde ODIN makaleden belirgin iyi (34.60 vs 47.75 FPR95) ve sıralama farklı (bizde ODIN MSP'den iyi; makalede kötü). Alt küme gürültüsü (~2-3 puan FPR95) bu farkı açıklamaz. Olası neden: makalenin ODIN hiperparametreleri ve checkpoint'i belirtilmemiş (biz T=1000, ε=0.0014 kullandık). Raporda bu "birebir tutmadı, nedeni belirsiz" diye dürüstçe yazılmalı.
- Alt küme boyutu nedeniyle MSP/Energy/ODIN sayıları MCM/GL-MCM kadar kesin değildir (AUROC ~±1, FPR95 ~±2-3 puan).

### 6.7 Sonuçların makaleyle uyumu (özet)
- COCO ve ImageNet'te MCM/L-MCM/GL-MCM ortalamaları makaleyle çok yakın (COCO'da farklar ≤0.2 puan; ImageNet MCM 42.26 vs 42.74, GL-MCM 35.33 vs 35.25). Bu, uygulamanın ve veri hazırlığının doğru olduğunun güçlü kanıtıdır.
- VOC'ta ortalama sapma daha büyük (MCM FPR95 42.66 vs 38.88; GL-MCM 30.32 vs 31.12). VOC ID yalnızca 1000 görüntü olduğundan FPR95 doğal olarak oynak; bu yüzden VOC'ta set bazlı farklar (ör. iNaturalist MCM 11.70 vs 8.20) beklenir. Genel eğilim (GL-MCM en iyi) aynıdır.
- Bunlar "birebir tekrar üretim" değil, rastgele alt kümeler ve tek tohumla yapılan bir tekrar üretimdir; küçük sapmalar normaldir.

---------------------------------------------------------------------------------------------------

## 7. Uygulanan ve uygulanmayan yöntemler (özet)
- **Uygulandı (kendi çalıştırdığımız sayılar):** MCM, L-MCM, GL-MCM (CLIP ViT-B/16; COCO_single, VOC_single, ImageNet-1K); MSP, Energy, ODIN (ImageNet-1K).
- **Uygulanmadı (rapor "makaleden alındı, bizim uygulamamız değil" diye etiketlemeli):** ViM, KNN (ImageNet eğitim özellikleri gerekir), NPOS (eğitim gerekir), CoOp, LoCoOp ve bunların GL-MCM'li hâlleri (16-shot istem öğrenimi gerekir). Neden uygulanmadığı: bu yöntemler eğitim (CoOp, LoCoOp, NPOS) veya ImageNet eğitim seti özellikleri (ViM, KNN) gerektirir ve vize süresi içinde yapılmadı; final ödevi aynı makalenin devamıdır.
- **Makalenin tabloları (arXiv v4 HTML sürümünden kontrol edildi) ve bizim durumumuz:**
  - **Tablo 1 (ImageNet-1K ana sonuçlar):** tamamı yukarıdaki 6.6 tablosunda; 13 yöntemden 6'sı uygulandı (MSP, Energy, ODIN, MCM, L-MCM, GL-MCM), 7'si uygulanmadı (ViM, KNN, NPOS, CoOp_MCM, CoOp_GL-MCM, LoCoOp_MCM, LoCoOp_GL-MCM). MSP/Energy makalede "Ming ve ark. 2022'den", ODIN/ViM/KNN/NPOS "Tao ve ark. 2023'ten" alınmış.
  - **Tablo 2 (MS-COCO ve Pascal-VOC):** yalnızca MCM, L-MCM, GL-MCM satırları var; hepsi uygulandı (COCO_single ve VOC_single).
  - **Tablo 3 (τ ablation, makale):** uyguladık (kavram 2; bizim fazladan MCM/L-MCM τ=0.01 ölçümümüz var).
  - **Uygulanmayan/tekrar üretilmeyen diğer makale deneyleri:** Tablo 4 (SAN ile yerel özellik; CLIPN, SAN), Tablo 5 (CLIP-ResNet-50), Tablo 6 (Grounding DINO karşılaştırması), Ek Tablo A (skor fonksiyonu ablation'ı, (i)-(vii)), Ek Tablo B (COCO'dan çıkarılan ID verisi sayıları), Ek Tablo C (COCO_multi, ResNet-50 ve ViT-B). COCO_multi veri seti indirildi (5000 görüntü) ama koşturulmadı.
  - Raporda bu satırlar "yapmadık, vize kapsamı dışında bıraktık" diye açıkça yazılmalı.

## 8. Dürüst sınırlamalar ve sapmalar (rapor bunları saklamamalı)
1. Tek tohum (seed=1), tek rastgele alt küme; güven aralığı hesaplanmadı. Farkların istatistiksel anlamlılığı test edilmedi.
2. VOC_single'da ID yalnızca 1000 görüntü; FPR95 ~2-4 puan oynak (makaleden sapma bundan).
3. MSP/Energy: ID 10.000, OOD 3.000 alt küme; ODIN: ID 2.000, OOD 500. Makale tam ImageNet-1K val (50.000) ve tam OOD setleri kullanmış olabilir (bunu makalenin kendisi ayrıca belirtmiyorsa bilinmez).
4. Baseline checkpoint'i makalede belirtilmemiş; timm modeli tahmindir. ODIN hiperparametreleri de bilinmiyor; ODIN sonucu makaleyle uyuşmuyor (bizde daha iyi).
5. ViM, KNN, NPOS, CoOp, LoCoOp (+GL) uygulanmadı.
6. λ taramasında seçim ve değerlendirme aynı veri üzerinde; makalenin λ'sı korunmuştur (sonradan seçilmedi).
7. Sıcaklık ablation'ında yalnızca iki değer (1 ve 0.01) denendi (makale de yalnızca bu ikisini veriyor); aradaki değerler denenmedi.
8. Yerel özellik açıklamaları ("yorum" etiketli yerler) test edilmiş nedensel iddia değildir; nitel örnek figürleri (fig12-fig13) kanıt değil, görsel örnektir.
9. Zaman ölçümleri M1 MacBook Air'e özgüdür.
10. Nitel örnek şekiller (fig12-fig13) tek bir OOD setinin (SUN) rastgele 600 görüntüsünden seçilmiştir; skorlar eşiğe çok yakındır ve yerel haritalar gürültülüdür. Kanıt değil, görsel örnek olarak sunulmalıdır (`docs/DOSYA_VE_GORSEL_REHBERI.md`).
11. İlerleme raporundaki (5 Kasım/12 Kasım) ilk sonuçlar yalnızca COCO_single / `ood_voc` içindir; bu belgedeki nihai sayılar onların yerine geçer.

---------------------------------------------------------------------------------------------------

## 9. Rapor iskeleti ve yazım kuralları (kızlar ve onların yapay zekâ ajanları için)

### 9.1 Önerilen bölümler
1. Kapak sayfası: üniversite, ders (COME421/1 Introduction to Computer Vision), vize ödevi, makale adı, ekip üyelerinin her biri için ad soyad + öğrenci numarası (eksik üye = sıfır), tarih. Öğretim üyesi: Asst. Prof. Dr. Gamze Uslu.
2. Giriş: sıfır-atış (zero-shot) OOD tespiti nedir, neden gerekli (bölüm 2).
3. Makalenin özeti: MCM, L-MCM, GL-MCM formülleri, yerel özelliğin nasıl çıkarıldığı (bölüm 2).
4. Değerlendirme metrikleri: FPR95, AUROC ("ranking quality"), AUPR (bölüm 2.3). Hocanın istediği "ranking quality" açıklaması burada olmalı.
5. Veri setleri ve kurulum (bölüm 5, `DATASETS.md`).
6. Kod açıklaması: her kod bölümü ayrı ayrı (bölüm 4). Hoca "kodun her bölümünü açıkla" istiyor.
7. Makalenin karşılaştırma tablosu ve hangi yöntemlerin uygulandığı/uygulanmadığı (bölüm 6.6, 7). Hocanın istediği.
8. Deney sonuçları: bölüm 6.1-6.3 tabloları ve ilgili şekiller (`docs/DOSYA_VE_GORSEL_REHBERI.md` her şeklin nerede kullanılacağını söyler). Her çıktı bölümü için açıklama: bu hoca "çıktının her bölümünü açıkla" istiyor.
9. İki kavramın öncesi-sonrası karşılaştırması (hocanın temel isteği, en çok dikkat ettiği yer): yerel özellik yok/var (fig01, fig02, fig05, fig06, fig09, fig12-13) ve sıcaklık (fig03). Ek: λ taraması (fig04).
10. Makaleyle kıyas: fig07, fig08, bölüm 6.7.
11. Baseline'lar ve checkpoint bulgusu: fig10, fig11, bölüm 6.6.
12. Sınırlamalar ve zorluklar: bölüm 8 ve bölüm 3'ün sonundaki "karşılaşılan sorunlar".
13. Sonuç.
14. Kaynakça: Miyai ve ark. (GL-MCM, IJCV 2025, arXiv 2304.04521), Ming ve ark. 2022 (MCM, arXiv 2211.13445), Radford ve ark. 2021 (CLIP), Huang ve ark. 2021 (arXiv 2105.01879), Wang ve ark. 2021 (arXiv 2109.14162), Fort ve ark. 2021 (arXiv 2106.03004), Hendrycks ve Gimpel (MSP), Liu ve ark. (Energy), Liang ve ark. (ODIN), Cimpoi ve ark. (DTD; Texture), Deng ve ark. (ImageNet). Kaynakçadaki ayrıntıları (başlık, yıl, konferans) yazarken doğrulayın.
15. Ek: kod listesi, depo bağlantısı (GitHub).

### 9.2 Yazım kuralları (kullanıcı ile varılan kararlar)
- Rapor ekibin kendi cümleleriyle, sade ve doğal bir akademik dille yazılmalı (kalıp cümle, gereksiz abartı ve aşırı madde işaretlendirme yok). Yapay zekâ kullanımı dersin kuralına uygun olmalı ve ekip her cümleyi anlayıp sahiplenmelidir. Uzunluk dolgu cümleyle değil, kod ve çıktı açıklamaları, tablolar ve çok sayıda grafikle sağlanmalı.
- Rakamlar yalnızca bu pakette bulunanlardır (`docs/RESULT_TABLES.md`, `tables/`, `results/**.csv`). Yeni rakam uydurulmaz, yuvarlama değiştirilmez.
- "Geliştirdik" DENMEZ. Doğrusu: "makalenin yöntemini uyguladık ve iki bileşenin (yerel özellik, sıcaklık) etkisini ablation olarak ölçtük."
- Uygulanmayan yöntemler (ViM, KNN, NPOS, CoOp, LoCoOp) uygulanmış gibi yazılmaz; makaledeki değerler "makaleden alındı" diye etiketlenir.
- GL-MCM'in iki kıyasta birden kullanılması: kavram 1 = yerel özellik (yok: MCM, var: GL-MCM, referans L-MCM); kavram 2 = sıcaklık (τ=1 vs τ=0.01). Aynı yöntemi iki kez kıyasladık izlenimi verilmemeli.
- Dört ana başlık: MCM, L-MCM, GL-MCM, sıcaklık kıyası.
- Sıcaklık kıyasının makalenin kendi Tablo 3'ü olduğunu ve bizim sonuçlarımızın onunla uyuştuğunu belirtin.
- İstisnalar (COCO Texture, SUN'da L-MCM, ODIN uyumsuzluğu) gizlenmez.
- "Yorum/olası neden" olarak işaretlenen cümleler nedensel iddia gibi yazılmaz.
- Metin dili: ekip karar verir (ödev İngilizce makale üzerine; ders dili ve hocanın tercihini kontrol edin).
- Dersin yapay zekâ kullanım kuralını kontrol edin ve ona göre davranın.

---------------------------------------------------------------------------------------------------

## 10. Hocanın sorabileceği sorular ve kısa cevaplar (sunum/sözlü için)
1. **"Ranking quality nedir?"** AUROC: rastgele seçilen bir ID görüntüsünün skorunun rastgele seçilen bir OOD görüntüsünün skorundan yüksek olma olasılığı. Eşikten bağımsızdır. FPR95 ise tek eşikte (ID'lerin %95'i doğru kabul edilirken) OOD'lerin yüzde kaçının yanlışlıkla ID sayıldığıdır (tek nokta). AUPR kesinlik-duyarlılık eğrisinin altındaki alan.
2. **"Neden GL-MCM, MCM'den iyi?"** Global skor görüntünün tamamını tek vektöre sıkıştırır; küçük ID nesnesi başka nesneler arasında kaybolabilir. Yerel skor en iyi yamayı yakalar. İkisinin toplamı hem merkezi tek nesneyi hem de küçük nesneyi kapsar. Sonuçlarımız: COCO FPR95 -11.5, VOC -12.3, ImageNet -6.9 puan.
3. **"Her OOD setinde iyileşiyor mu?"** Hayır: COCO Texture'da kötüleşiyor, COCO SUN'da L-MCM daha iyi, ImageNet Texture'da AUROC çok az düşüyor (bölüm 6.4).
4. **"L-MCM neden ImageNet'te çok kötü?"** Olası neden: ImageNet'te ID görüntüsü tek ve büyük nesnedir, yerel yamalar çok sınıfla eşleşebilir; yerel max OOD'de de yüksek çıkar. Test edilmedi.
5. **"τ neden 1?"** Makale τ=1 kullanıyor ve kendi Tablo 3'ünde τ=0.01 ile kıyaslıyor; bizim GL-MCM sonuçlarımız o tabloyla uyuşuyor. Biz ayrıca MCM ve L-MCM için de τ=0.01'i ölçtük; neredeyse her durumda daha kötü çıktı.
6. **"λ nasıl seçildi?"** Makalenin değerleri (1.0 COCO/VOC, 0.5 ImageNet) kullanıldı; biz ayrıca 0-3 taraması yapıp makalenin seçimlerinin en iyiye yakın olduğunu gördük.
7. **"Tüm yöntemleri uyguladınız mı?"** Hayır: MCM, L-MCM, GL-MCM, MSP, Energy, ODIN uygulandı; ViM, KNN, NPOS, CoOp, LoCoOp uygulanmadı (eğitim/ImageNet-train özellikleri gerekir). Makale değerleri etiketli olarak gösterildi.
8. **"Neden VOC'ta makaleden sapma var?"** VOC ID yalnızca 1000 görüntü; FPR95 oynak.
9. **"ODIN neden makaleden farklı?"** Makalenin ODIN hiperparametreleri/checkpoint'i belirtilmemiş; biz T=1000, ε=0.0014 kullandık. Fark alt küme gürültüsünden büyük; nedeni kesin bilinmiyor.
10. **"Yerel özellik nasıl çıkarılıyor? Eğitim var mı?"** Eğitim yok (zero-shot, training-free). CLIP'in son blokunda dikkat karışımı atlanıp her yama için değer (value) projeksiyonu kullanılır (bölüm 2.2).
11. **"Kodu siz mi yazdınız?"** Orijinal depo yazarlarındır; biz MPS uyumu, tek geçişli değerlendirme betiği (`eval_all_scores.py`), baseline betiği, tablo/grafik betikleri ve veri dönüştürme betiği ekledik (bölüm 4). Kim neyi yazdı sorusunda dürüst cevap verin.
12. **"Neden bu kadar küçük alt küme?"** Donanım (8 GB, MPS) ve süre; tam ImageNet baseline koşuları saatler sürüyor.

---------------------------------------------------------------------------------------------------

## 11. Teslim bilgileri
- Ders: COME421/1 Introduction to Computer Vision, Üsküdar Üniversitesi; öğretim üyesi Asst. Prof. Dr. Gamze Uslu.
- Teslim: tek zip/rar dosyası, STIX bağlantısından (dosya başına ≤50 MB); son tarih 19.11.2026 01:00 (yani 18 Kasım gecesi); başka kanaldan veya geç teslim kabul edilmiyor.
- Kapak sayfası: her üyenin ad, soyad ve numarası (eksik = sıfır). Ekip: Mehmet Yusuf İzci (220201052), Meryem Kaşıkçı, İlayda Yalçınöz, Sude Ünal (numaralar ekipçe doldurulacak).
- Vize %45, final %55 (final aynı makalenin devamı).
- 18 Kasım 14:00 çevrimiçi ders (Meet), devam zorunluluğu %70.
- Raporu yazma, kapak ve STIX yüklemesi ekip arkadaşlarının işidir; kodu, sonuçları ve şekilleri hazırlama Yusuf'un.
- Zip içeriği (öneri): rapor (PDF/Word), `figures/`, `tables/`, `docs/`, kod dosyaları, `results/` (CSV'ler; histogram PNG'leri), `results_raw/` (npz), `requirements.txt`, `DATASETS.md`, `README`. Veri setleri zip'e girmez (çok büyük; `DATASETS.md` ile indirilir).
