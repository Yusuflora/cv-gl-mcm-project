# GL-MCM VİZE PROJESİ: RAPOR YAZIMI İÇİN TAM PAKET (tek dosya)

Bu dosya, COME421/1 Introduction to Computer Vision vize ödevinin **tüm deney işini** anlatan eksiksiz pakettir. Deneyler bitti; tablolar ve görseller hazır. Senin (yapay zekâ asistanının) görevi, bu bilgilerle ekibin **raporunu yazmaya yardım etmek**.

**Projenin GitHub deposu (kod, sonuçlar, tüm şekiller, tablolar):** https://github.com/Yusuflora/cv-gl-mcm-project
Depoya erişebiliyorsan `START_HERE.md` dosyasından başla ve şekilleri `figures/` klasöründen al. Erişemiyorsan bu dosyadaki bilgiler yeterlidir; görseller ayrıca "GORSELLER.zip" olarak verilmiştir ve dosya adları aşağıdaki "Dosya ve görsel rehberi" bölümündeki adlarla birebir aynıdır.

## Bu dosyanın içindekiler (sırayla)
1. YAPAY ZEKÂ İÇİN TALİMAT (yazım kuralları, nelere dikkat edileceği)
2. PROJE AÇIKLAMASI: ödev, makale, yapılan her adım, kodun her bölümü, kurulum, tüm sonuçlar, sınırlamalar, rapor iskeleti, hocanın olası soruları
3. DOSYA VE GÖRSEL REHBERİ: her dosya ve her şekil neyi gösterir, hangi rakamları öne çıkarır, raporda nereye konur
4. TÜM SONUÇ TABLOLARI (rakamların tek kaynağı)
5. KARARLAR VE NOTLAR (ham karar günlüğü; çelişki olursa 2. bölüm geçerlidir)
6. VERİ SETLERİ

Notlar: Metin içinde geçen "bu depodaki dosya" ifadeleri, GitHub deposundaki dosyalardır; içerikleri bu belgede zaten verilmiştir. Rakamlar yalnızca bu belgedeki tablolardan alınmalıdır.




====================================================================================================
# BÖLÜM 1. YAPAY ZEKÂ İÇİN TALİMAT
(kaynak: docs/AGENT_PROMPT.md)
====================================================================================================

# Yapay zekâ asistanına verilecek talimat (kopyala-yapıştır)

> Önce ekip olarak dersin yapay zekâ kullanım kuralını kontrol edin. Kural izin vermiyorsa veya sınırlıyorsa buna uyun. Üretilen metindeki her cümleyi teslimden önce kendiniz okuyup anlayın: hoca sözlü soru sorabilir (`docs/PROJE_ACIKLAMASI.md` bölüm 10).

---

Sen bir üniversite bilgisayarlı görü dersi vize raporu yazımında yardımcı oluyorsun. Elimizde bitmiş bir deney projesi var (GitHub deposu ve içindeki dosyalar). Görevin, bu projenin RAPORUNU yazmaya yardım etmek. Deneyleri tekrar koşturma, yeni sayı üretme, hiçbir sayıyı uydurma.

## Önce oku (sırayla, tamamını)
1. `START_HERE.md`
2. `docs/PROJE_ACIKLAMASI.md` (ana kaynak; ödev metni, makale açıklaması, adımlar, kod haritası, sonuçlar, sınırlamalar, rapor iskeleti, hocanın olası soruları)
3. `docs/DOSYA_VE_GORSEL_REHBERI.md` (her dosya ve her şekil: ne gösterir, hangi rakam, raporda nerede)
4. `docs/RESULT_TABLES.md` ve `tables/` (tüm sayılar)
5. `docs/KARARLAR_VE_NOTLAR.md` (ham karar notları; PROJE_ACIKLAMASI ile çelişirse PROJE_ACIKLAMASI geçerlidir)
6. `DATASETS.md`, kod dosyaları (`eval_all_scores.py`, `baselines_imagenet.py`, `make_tables.py`, `make_figures.py`, `qualitative_examples.py`, `convert_imagenet_val.py`) ve `figures/` içindeki şekiller.

## Ödev
COME421/1 Introduction to Computer Vision, vize ödevi. Makale: Miyai ve ark., "GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot OOD Detection", IJCV 2025. İstenen: makalede karşılaştırılan yöntemleri uygulamak, makalenin veri setlerinde çalıştırmak, aynı metriklerle (FPR95, AUROC) kıyaslamak, tablolar ve grafikler; ayrıca kodun her bölümünü ve çıktının her bölümünü açıklayan bir rapor. Hocanın temel isteği: makaleden iki kavramı seçip "kullanılan / kullanılmayan" öncesi-sonrası karşılaştırmasını ayrıntılı ve bol görsel çıktıyla vermek (yerel özellik yok/var; sıcaklık τ=1 / τ=0.01).

## Yazım kuralları (kesin)
1. Rakamlar yalnızca depodaki dosyalardan alınır (`docs/RESULT_TABLES.md`, `tables/`, `results/**.csv`). Hiçbir sayıyı uydurma, yuvarlamayı değiştirme, "yaklaşık" diye başka rakam yazma.
2. Yöntemleri doğru etiketle. UYGULANDI: MCM, L-MCM, GL-MCM, MSP, Energy, ODIN. UYGULANMADI: ViM, KNN, NPOS, CoOp, LoCoOp ve GL-MCM'li hâlleri; bunların makaledeki değerleri varsa "makaleden alındı, bizim uygulamamız değil" diye yaz. Makalenin diğer deneylerini (Tablo 4, 5, 6, Ek Tablo A-C) yaptık diye yazma.
3. "Yöntemi geliştirdik" yazma. Doğrusu: "makalenin yöntemini uyguladık ve iki bileşenin (yerel özellik, sıcaklık) etkisini ablation olarak ölçtük." Makalenin sonuçlarını aştığımızı iddia etme; sonuçlarımız makaleyle uyumlu.
4. Sınırlamaları ve istisnaları saklama (PROJE_ACIKLAMASI §8): VOC'ta ID yalnızca 1000 görüntü; MSP/Energy/ODIN alt kümelerde; ODIN makaleyle uyuşmuyor; COCO Texture'da GL-MCM, MCM'den kötü; baseline checkpoint'i makalede belirtilmediği için timm modeli tahmin; tek tohum, güven aralığı yok.
5. "Yorum/olası neden" olarak işaretli açıklamaları kesin nedensel iddia gibi yazma ("büyük olasılıkla", "olası neden" kullan).
6. GL-MCM'i iki kez kıyaslamış gibi görünme: Kavram 1 = yerel özellik (MCM yok / GL-MCM var, L-MCM yalnızca-yerel referans); Kavram 2 = sıcaklık (τ=1 / τ=0.01). Dört ana başlık: MCM, L-MCM, GL-MCM, sıcaklık kıyası.
7. Hocanın istekleri raporda açıkça karşılanmalı: (a) "ranking quality" = AUROC açıklaması (+ AUPR, FPR95'in tek eşikli olduğu); (b) makaledeki ImageNet tablosu ve hangi yöntemlerin uygulanıp uygulanmadığı; (c) kodun her bölümünün açıklaması; (d) çıktının her bölümünün açıklaması; (e) iki kavram için öncesi-sonrası kıyası, çok sayıda şekil (hepsini `figures/` içinden, DOSYA_VE_GORSEL_REHBERI'ndeki eşlemeye göre kullan).
8. Dil ve ton: sade, doğal, öğrenci düzeyinde akademik Türkçe veya ekibin seçtiği dil. Kalıp cümle, gereksiz abartı, aşırı madde işareti ve dolgu yok. Uzunluk; kod bölümü bölümü açıklama, her çıktının açıklaması, tablolar ve çok sayıda şekille sağlanmalı, dolgu cümleyle değil.
9. Kapak sayfası: üniversite, ders, vize ödevi, makale adı, ekibin HER ÜYESİNİN ad, soyad ve numarası (eksik üye sıfır alır), tarih. Bu bilgileri bana ekip verecek; kendin uydurma.
10. Kaynakça: PROJE_ACIKLAMASI §9.1'deki listeyi kullan, başlık/yıl/konferans ayrıntılarını doğrulamadan yazma; emin olmadığın kaynakta "doğrulanacak" diye not bırak.
11. Her şekli rapora koyarken şekil numarası, altyazı ve metin içinde yorum yaz. Şekle bakamıyorsan DOSYA_VE_GORSEL_REHBERI'ndeki açıklamayı kullan ama şekil dosyasını açıp kontrol et.
12. Emin olmadığın veya depoda bulamadığın bilgi için tahmin yürütme; "BİLGİ YOK: ..." diye işaretleyip bana sor.

## Çıktı olarak istediğim
Önce rapor iskeletini (bölüm başlıkları ve her bölümde hangi şekil/tablo/rakam geçecek) çıkar, ben onaylayınca bölümleri tek tek yaz. Her bölüm sonunda, kullandığın tüm rakamları ve kaynak dosyalarını bir liste olarak ver ki kontrol edebileyim.


====================================================================================================
# BÖLÜM 2. PROJE AÇIKLAMASI
(kaynak: docs/PROJE_ACIKLAMASI.md)
====================================================================================================

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


====================================================================================================
# BÖLÜM 3. DOSYA VE GÖRSEL REHBERİ
(kaynak: docs/DOSYA_VE_GORSEL_REHBERI.md)
====================================================================================================

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


====================================================================================================
# BÖLÜM 4. TÜM SONUÇ TABLOLARI
(kaynak: docs/RESULT_TABLES.md)
====================================================================================================

# Result tables (generated by make_tables.py)

Each cell: FPR95 / AUROC. AVG = mean over the OOD sets.

### COCO_single, tau=1 (FPR95 lower is better, AUROC higher is better)

| Method | iNaturalist | SUN | Texture | IN22k | ood_voc | AVG |
|---|---|---|---|---|---|---|
| MCM (ours) | 64.06 / 90.76 | 84.08 / 81.79 | 65.00 / 86.64 | 78.38 / 84.15 | 77.48 / 84.42 | 73.80 / 85.55 |
| MCM (article) | 64.42 / 90.75 | 84.18 / 81.81 | 65.10 / 86.62 | 78.50 / 84.14 | 77.62 / 84.43 | 73.96 / 85.55 |
| L-MCM (ours) | 41.76 / 92.46 | 51.76 / 89.33 | 85.72 / 74.89 | 75.60 / 77.92 | 72.10 / 84.21 | 65.39 / 83.76 |
| L-MCM (article) | 41.80 / 92.46 | 51.52 / 89.24 | 85.68 / 74.87 | 75.50 / 77.92 | 72.02 / 84.21 | 65.30 / 83.74 |
| GL-MCM (ours) | 34.90 / 94.46 | 62.32 / 88.53 | 72.06 / 84.55 | 70.74 / 84.49 | 71.67 / 87.71 | 62.34 / 87.95 |
| GL-MCM (article) | 34.72 / 94.45 | 61.96 / 88.49 | 72.08 / 84.53 | 70.66 / 84.49 | 71.62 / 87.72 | 62.21 / 87.94 |

### COCO_single: effect of temperature tau (averages over OOD sets, FPR95 / AUROC)

| Method | tau=1 | tau=0.01 |
|---|---|---|
| MCM | 73.80 / 85.55 | 73.12 / 84.08 |
| L-MCM | 65.39 / 83.76 | 66.92 / 82.99 |
| GL-MCM | 62.34 / 87.95 | 64.42 / 86.63 |

### VOC_single, tau=1 (FPR95 lower is better, AUROC higher is better)

| Method | iNaturalist | SUN | Texture | IN22k | ood_coco | AVG |
|---|---|---|---|---|---|---|
| MCM (ours) | 11.70 / 97.98 | 31.50 / 94.11 | 53.80 / 91.89 | 56.30 / 90.29 | 60.00 / 88.53 | 42.66 / 92.56 |
| MCM (article) | 8.20 / 98.23 | 28.60 / 94.68 | 51.70 / 91.45 | 51.40 / 90.94 | 54.50 / 89.02 | 38.88 / 92.86 |
| L-MCM (ours) | 28.80 / 95.06 | 45.60 / 88.53 | 60.70 / 85.78 | 53.10 / 87.42 | 59.20 / 84.04 | 49.48 / 88.17 |
| L-MCM (article) | 27.70 / 94.97 | 44.20 / 88.87 | 60.00 / 86.23 | 54.10 / 86.27 | 57.00 / 84.06 | 48.72 / 88.08 |
| GL-MCM (ours) | 4.40 / 98.63 | 23.50 / 94.17 | 40.10 / 92.81 | 39.10 / 92.38 | 44.50 / 90.28 | 30.32 / 93.65 |
| GL-MCM (article) | 4.20 / 98.71 | 23.10 / 94.66 | 43.00 / 92.84 | 41.00 / 92.38 | 44.30 / 90.48 | 31.12 / 93.81 |

### VOC_single: effect of temperature tau (averages over OOD sets, FPR95 / AUROC)

| Method | tau=1 | tau=0.01 |
|---|---|---|
| MCM | 42.66 / 92.56 | 44.56 / 90.36 |
| L-MCM | 49.48 / 88.17 | 55.96 / 85.82 |
| GL-MCM | 30.32 / 93.65 | 42.34 / 91.25 |

### ImageNet, tau=1 (FPR95 lower is better, AUROC higher is better)

| Method | iNaturalist | SUN | places365 | Texture | AVG |
|---|---|---|---|---|---|
| MCM (ours) | 30.70 / 94.63 | 37.39 / 92.57 | 43.66 / 89.96 | 57.29 / 86.18 | 42.26 / 90.83 |
| L-MCM (ours) | 48.98 / 87.00 | 73.32 / 76.66 | 79.10 / 71.93 | 92.57 / 59.61 | 73.49 / 73.80 |
| GL-MCM (ours) | 17.50 / 96.44 | 30.79 / 93.45 | 37.67 / 90.64 | 55.34 / 85.54 | 35.33 / 91.52 |

### ImageNet: effect of temperature tau (averages over OOD sets, FPR95 / AUROC)

| Method | tau=1 | tau=0.01 |
|---|---|---|
| MCM | 42.26 / 90.83 | 56.01 / 83.82 |
| L-MCM | 73.49 / 73.80 | 79.97 / 67.93 |
| GL-MCM | 35.33 / 91.52 | 50.66 / 84.71 |

### GL-MCM temperature ablation: ours vs article Table 3 (FPR95 / AUROC, average over OOD sets)

| ID dataset | tau | Ours | Article (Table 3) |
|---|---|---|---|
| COCO_single | 0.01 | 64.42 / 86.63 | 64.89 / 86.70 |
| COCO_single | 1 | 62.34 / 87.95 | 62.21 / 87.94 |
| VOC_single | 0.01 | 42.34 / 91.25 | 43.92 / 90.80 |
| VOC_single | 1 | 30.32 / 93.65 | 31.12 / 93.81 |
| ImageNet | 0.01 | 50.66 / 84.71 | 50.72 / 84.70 |
| ImageNet | 1 | 35.33 / 91.52 | 35.25 / 91.51 |

### ImageNet-1K main table of the article: what we implemented and what we did not (averages over iNaturalist, SUN, Places, Texture)

| Method | Group | Status | Ours FPR95 | Ours AUROC | Article FPR95 | Article AUROC | Note |
|---|---|---|---|---|---|---|---|
| MSP | Supervised (post-hoc) | IMPLEMENTED | 44.63 | 88.52 | 45.12 | 88.25 | ours: 10,000 ID / 3,000 per OOD set |
| Energy | Supervised (post-hoc) | IMPLEMENTED | 30.54 | 93.52 | 35.92 | 92.26 | ours: 10,000 ID / 3,000 per OOD set |
| ODIN | Supervised (post-hoc) | IMPLEMENTED | 34.60 | 91.99 | 47.75 | 88.80 | ours: 2,000 ID / 500 per OOD set (ODIN is slow) |
| ViM | Supervised (post-hoc) | not implemented | - | - | 50.20 | 87.82 | NOT implemented (needs ImageNet training features) |
| KNN | Supervised (post-hoc) | not implemented | - | - | 42.19 | 90.97 | NOT implemented (needs ImageNet training features) |
| NPOS | Supervised (post-hoc) | not implemented | - | - | 37.93 | 91.22 | NOT implemented (needs training) |
| MCM | Zero-shot CLIP | IMPLEMENTED | 42.26 | 90.83 | 42.74 | 90.77 | ours: full sets |
| L-MCM | Zero-shot CLIP | IMPLEMENTED | 73.49 | 73.80 | 73.59 | 73.73 | ours: full sets |
| GL-MCM | Zero-shot CLIP | IMPLEMENTED | 35.33 | 91.52 | 35.25 | 91.51 | ours: full sets, lambda=0.5 |
| CoOp_MCM | Prompt learning | not implemented | - | - | 36.83 | 91.93 | NOT implemented (needs 16-shot training) |
| CoOp_GL-MCM | Prompt learning | not implemented | - | - | 30.03 | 92.59 | NOT implemented (needs 16-shot training) |
| LoCoOp_MCM | Prompt learning | not implemented | - | - | 33.98 | 92.69 | NOT implemented (needs 16-shot training) |
| LoCoOp_GL-MCM | Prompt learning | not implemented | - | - | 28.68 | 93.59 | NOT implemented (needs 16-shot training) |

### AUPR (ours only; the article does not report AUPR), averages over OOD sets, tau=1

| ID dataset | MCM | L-MCM | GL-MCM |
|---|---|---|---|
| COCO_single | 88.37 | 84.09 | 89.57 |
| VOC_single | 93.32 | 87.99 | 94.08 |
| ImageNet | 98.18 | 93.59 | 98.27 |

### Effect of lambda on GL-MCM (tau = 1, averages over the OOD sets; lambda = 0 is exactly MCM)

| lambda | COCO (ID) FPR95 / AUROC | VOC (ID) FPR95 / AUROC | ImageNet-1K (ID) FPR95 / AUROC |
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


====================================================================================================
# BÖLÜM 5. KARARLAR VE NOTLAR
(kaynak: docs/KARARLAR_VE_NOTLAR.md)
====================================================================================================

# Kararlar ve notlar (rapor açıklama metnine girecek ayrıntılar)

Bu dosya, proje sırasında verilen her karar ve her önemli ayrıntı için çalışma notudur. Son açıklama metni bu dosyadaki her maddeyi içermelidir.

## 1. Ödevin kapsamı ve hocanın beklentisi
- Ödev (COME421 Midterm): seçilen makalede (GL-MCM, IJCV 2025) karşılaştırılan yöntemlerin Python uygulaması, makalenin veri setlerinde çalıştırılması, aynı metriklerle (FPR95, AUROC) kıyas, tablo ve grafik; ayrıca kodun her bölümünü ve çıktının her bölümünü açıklayan rapor.
- Hocanın sözlü geri bildirimi: (a) "ranking quality" skorunun ne olduğunu söyleyin (= AUROC; AUPR de var; FPR95 tek eşikli bir ölçü), (b) makaledeki karşılaştırma tablosunu gösterin ve hangi yöntemleri uyguladığımızı, hangilerini uygulamadığımızı belirtin, (c) deneysel sonuçlar tablosundaki yöntemleri uygulayın.
- Hocanın temel isteği: seçilen makaleden 2 kavram seçip "kullanılırsa / kullanılmazsa" diye öncesi-sonrası kıyası yapmak; kıyasları ayrıntılı yapmak ve bolca görselleştirilmiş çıktı vermek (en çok buna dikkat ediyor).

## 2. Seçilen iki kavram (ablation / bileşen etkisi analizi)
1. Yerel (local) özellik kullanımı: yok = MCM, var = GL-MCM. L-MCM (sadece yerel) ek referans olarak gösterilir.
2. Sıcaklık ölçekleme (temperature, tau): tau=1 (ölçekleme yok) ile tau=0.01 (ölçekleme var) karşılaştırması.
- GL-MCM'i iki kıyasta birden kullanmamak için ikinci kavram olarak global özellik yerine sıcaklık seçildi (aynı yöntemi iki kez karşılaştırdık izlenimini önlemek için).
- Ek analiz: yerel skor ağırlığı lambda taraması (ham skorlardan hesaplanır, yeni koşu gerektirmez).
- Dil: "yöntemi geliştirdik" DENMEZ. Doğrusu: makalenin yöntemini uyguladık ve iki bileşenin etkisini ablation olarak ölçtük.

## 3. Uygulanan ve uygulanmayan yöntemler (makalenin ImageNet tablosu)
- Uygulandı: MCM, L-MCM, GL-MCM (CLIP ViT-B/16, zero-shot); MSP, Energy, ODIN (ImageNet-1K'da fine-tune edilmiş ViT-B/16). NOT: ilk denemede torchvision ağırlıkları kullanıldı; ANA baseline sonuçları timm (in21k ön eğitim + in1k fine-tune) modelidir, bkz. madde 10-13. torchvision sonuçları yalnızca checkpoint etkisini gösteren ek bulgudur.
- Uygulanmadı: ViM, KNN (ImageNet eğitim seti özellikleri gerektirir), NPOS, CoOp, LoCoOp ve bunların GL-MCM'li hâlleri (ImageNet-1K eğitim alt kümesi ve eğitim gerektirir). Makaledeki değerler raporda "makaleden alındı, bizim uygulamamız değil" diye açıkça etiketlenmelidir.

## 4. Sonuçların makaleyle uyumu
- COCO_single, VOC_single ve ImageNet-1K'da MCM / L-MCM / GL-MCM ortalamaları makaleyle çok yakın (ayrıntı docs/RESULT_TABLES.md).
- VOC'ta ID seti 1000 görüntü olduğu için makaleden ortalamada ~4 puana kadar (MCM FPR95 42.66 vs 38.88) sapma var (örneklem küçük, FPR95 oynak).

## 5. Deney kurulumu ayrıntıları
- Model: CLIP ViT-B/16. Skorlar: MCM (global, max softmax), L-MCM (yerel, 14x14 yama üzerinde max), GL-MCM = MCM + lambda * L-MCM. lambda=1.0 (COCO, VOC), 0.5 (ImageNet). tau=1 (ana), tau=0.01 (ablation). ID = pozitif sınıf, yüksek skor = daha ID.
- Metrikler: FPR95 (düşük iyi), AUROC (yüksek iyi, "ranking quality"), AUPR (yüksek iyi, makale raporlamıyor).
- COCO_single: ID 5000 görüntü; OOD alt kümeleri rastgele 5000 (iNaturalist, SUN, Texture, IN22k), ood_voc 4000 (tamamı). VOC_single: ID 1000; OOD 1000'er (iNaturalist, SUN, Texture, IN22k, ood_coco). ImageNet: ID = ImageNet-1K doğrulama 50.000 görüntü; OOD = iNaturalist 10.000, SUN 10.000, Places 10.000, Texture 5.640 (hepsi tam).
- Rastgele alt küme tohumu 1; birden çok çağrıda aynı alt kümeyi korumak için --burn bayrağı kullanıldı.
- Temel hesaplama tek geçişte yapıldı (eval_all_scores.py): CLIP bir kez çalışır, ham global/yerel skorlar results_raw/ içine kaydedilir; üç skor, tau ve lambda ham değerlerden hesaplanır. Orijinal kod her skor için CLIP'i yeniden çalıştırıyordu.
- Orijinal koddan farklar: CUDA yerine Apple MPS, bellek için batch 32 / 2 işçi, plt.close('all') yaması.
- Donanım: MacBook (8 GB RAM, Apple M1, MPS); önceki kısım M5'te yapıldı (M1 ile bire bir aynı sonuç doğrulandı).

## 6. Baseline'lar için örneklem kısıtları (dürüstçe belirtilecek)
- MSP ve Energy: ID 10.000, her OOD setinden 3.000 rastgele görüntü.
- ODIN: ID 2.000, her OOD setinden 500 (geri-yayılım MPS'te çok yavaş, ~29 sn / 16 görüntü). ODIN sayıları daha oynak; MSP/Energy ile aynı veri üzerinde birebir kıyas değil.
- Bu yüzden makaledeki yöntem sıralaması bizde farklı çıkabilir; sebep alt küme gürültüsüdür (AUROC ~±1, FPR95 ~±2-3 puan).

## 7. Rapor biçimi ve teslim (sizin tarafınız)
- Rapor arkadaşlar tarafından yazılacak; doğal ve sade bir akademik görünümde olmalı.
- Kapak sayfası: grubun tüm üyelerinin ad, soyad, numarası (eksik üye = 0). Teslim: tek zip/rar, STIX linkinden, 19.11.2026 01:00 (18 Kasım gecesi) öncesi.
- Kod ve sonuçlar da teslim edilmeli. 18 Kasım 14:00 çevrimiçi ders (Meet), devam zorunluluğu %70.

## 8. Son kontrol (17 Kasım'da YAPILDI)
- Paket, sanki projeyi biz yapmamışız gibi baştan sona okundu; bayat cümleler, doğrulanamayan iddialar ve çelişkiler düzeltildi. Temiz bir kopyada tablo ve şekillerin yeniden üretildiği doğrulandı.

## 9. Küçük alt küme (ODIN) uyarısı, ayrıntı
- Alt küme küçük olduğu için ODIN sonuçları daha oynak: AUROC için yaklaşık ±1 puan, FPR95 için yaklaşık ±2-3 puan sapma beklenir.
- Makalede MSP ile ODIN arasındaki fark (FPR95 45.12 ve 47.75) bu sapmadan küçük; bu yüzden bizde MSP / ODIN / Energy sıralaması makaledekinden farklı çıkabilir. Böyle çıkarsa raporda bu durum "alt küme (örneklem) gürültüsü" olarak açıklanmalı, yöntem hatası gibi yazılmamalı.
- İsteğe bağlı iyileştirme (YAPILMADI, zaman yetmedi): ODIN daha büyük bir kümede (ID 4.000 / OOD 1.000, ~4 saat) yeniden koşturulabilirdi.

## 10. Baseline modeli sorunu ve çözümü (önemli bulgu)
- Makale MSP/Energy sayılarını kendisi üretmedi: "Ming ve ark. 2022" makalesinden aldı; ODIN/ViM/KNN/NPOS sayıları "Tao ve ark. 2023"ten. Hepsi "ImageNet-1K üzerinde fine-tune edilmiş" modeller (MSP satırı "ViT-B"). Hiperparametre verilmemiş. Bu bilgi makalenin Tablo 1 açıklamasından.
- İlk denememiz: torchvision ViT-B/16 (IMAGENET1K_V1; yalnızca ImageNet-1K ile, yoğun düzenlileştirmeyle eğitilmiş). MSP ve Energy sonuçları makaleden belirgin kötü çıktı (alt küme 10.000 ID / 3.000 OOD):
  - MSP: FPR95 61.20, AUROC 83.28 (makale 45.12 / 88.25)
  - Energy: FPR95 65.84, AUROC 74.52 (makale 35.92 / 92.26)
  - Bu sonuçlar results/ImageNet/{MSP,Energy}/ViT-B_16_supervised_ID_final_msp_energy/ altında duruyor; raporda "eğitim şekli ve checkpoint OOD skorlarını çok etkiliyor" bulgusu olarak kullanılabilir. ODIN bu modelle yavaş olduğu için yarıda durduruldu (sonuç yok).
- Düzeltme: ImageNet-21K ile ön eğitilip ImageNet-1K'da fine-tune edilmiş ViT-B/16 (timm: vit_base_patch16_224.orig_in21k_ft_in1k) ile baseline'lar yeniden koşuldu (baselines_imagenet.py --arch timm). Sonuçlar ViT-B_16_in21k_ft_in1k_... klasörlerinde.
- Dürüst not: makalenin tam hangi checkpoint'i kullandığı belli değil; timm modeli "en yakın makul tahminimiz".
- Kaynak kontrolü (16 Kasım): GL-MCM makalesi (arXiv 2304.04521v4) model adını vermiyor (MSP satırı yalnızca "ViT-B"). Sayıların alındığı MCM makalesi (Ming ve ark. 2022, arXiv 2211.13445) ilgili satırları "ViT-B / ViT-L, fine-tuned" diye etiketliyor; hangi ön eğitim veri setini kullandığı görünen metinde yazmıyor. O satırların dayandığı Fort ve ark. 2021 (arXiv 2106.03004) ise ViT-B_16'yı ImageNet-21k ile ön eğitilmiş checkpoint'lerden başlatıp kendi ID veri setinde (orada CIFAR) fine-tune ediyor. Yani "21k ile ön eğitim + ID setinde fine-tune" bu çizginin standart kurulumu; timm modelimiz (orig_in21k_ft_in1k) bunun ImageNet-1K karşılığıdır. ImageNet-1K için tam checkpoint makalede AÇIKÇA yazmıyor, raporda bu "makul tahmin" olarak belirtilmeli.

## 11. Raporun anlatım kuralları (kullanıcı ile varılan)
- L-MCM, "yerel özellik" kıyasında (madde 2.1) yardımcı üçüncü sonuç olarak raporda yer alır: MCM (sadece global) / L-MCM (sadece yerel) / GL-MCM (ikisi birlikte). Asıl kıyas MCM ile GL-MCM; L-MCM "yerel özellik tek başına kötü, global ile birleşince en iyi" bulgusunu gösterir (ImageNet'te L-MCM en kötü).
- Dört ana başlık: MCM, L-MCM, GL-MCM, sıcaklık (tau=1 ve tau=0.01) kıyası.
- Anlatım kendi cümlelerimizle, doğal ve sade olabilir; ama rakamlar, iddialar ve "uyguladık/geliştirdik" ifadeleri büyütülmez. Uygulanmayan yöntemler (CoOp, LoCoOp, NPOS, ViM, KNN) uygulanmış gibi yazılmaz.
- Rapor uzunluğu dolgu cümleleriyle değil, daha fazla grafik, tablo ve ayrıntılı kıyas açıklamasıyla sağlanır (hoca bolca görselleştirilmiş çıktı ve ayrıntılı kıyas istiyor).
- "Geliştirdik" yerine: "makalenin yöntemini uyguladık ve iki bileşenin (yerel özellik, sıcaklık) etkisini ablation olarak ölçtük."

## 12. timm (in21k ön eğitim + in1k fine-tune) ile baseline sonuçları
- MSP ve Energy, ID 10.000 / her OOD seti 3.000 görüntü, 4 OOD seti ortalaması:
  - MSP: FPR95 44.63, AUROC 88.52 (makale 45.12 / 88.25) -> neredeyse birebir
  - Energy: FPR95 30.54, AUROC 93.52 (makale 35.92 / 92.26) -> aynı eğilim, bizde biraz daha iyi
  - Ayrıntılı set bazlı sayılar results/ImageNet/{MSP,Energy}/ViT-B_16_in21k_ft_in1k_ID_final_msp_energy/ altındaki CSV dosyalarında (terminal çıktısındaki "results so far" satırları kümülatif ortalamadır, set bazlı değildir).
- Bu sonuç, model seçiminin (21k ön eğitim + 1k fine-tune) makaledeki baseline'lara yakın olduğunu destekliyor; torchvision modeli (yalnızca 1k, yoğun düzenlileştirme) MSP 61.20 / Energy 65.84 FPR95 ile çok daha kötüydü (madde 10).
- ODIN (ID 2.000 / OOD 500) koşusu bu ölçümden sonra başladı.

## 13. ODIN sonucu ve MSP/Energy/ODIN adil kıyası (ID 2.000 / OOD 500 alt kümesi)
- ODIN, MSP ve Energy aynı koşuda (final_odin) aynı 2.000 ID / 4x500 OOD görüntüsü üzerinde hesaplandı; adil kıyas bu üçlüdür (4 OOD seti ortalaması, FPR95 / AUROC / AUPR):
  - MSP: 43.55 / 88.94 / 96.65
  - Energy: 29.75 / 93.94 / 98.21
  - ODIN: 34.60 / 91.99 / 97.42
- Makale (Tablo 1, ImageNet ortalamaları): MSP 45.12 / 88.25, Energy 35.92 / 92.26, ODIN 47.75 / 88.80.
- Gözlem: MSP ve Energy makaleye yakın. ODIN bizde makaleden belirgin iyi (34.60 vs 47.75 FPR95); bu fark alt küme gürültüsüyle (~2-3 puan) açıklanamayacak kadar büyük. Olası nedenler: makalenin dayandığı ODIN uygulamasının hiperparametreleri (sıcaklık, perturbation) ve checkpoint'i bilinmiyor (biz T=1000, eps=0.0014 kullandık). Sıralama da farklı: bizde ODIN, MSP'den iyi; makalede ODIN, MSP'den kötü. Raporda bu dürüstçe "makalenin ODIN ayarları belirtilmediği için birebir tutmadı" diye yazılmalı.
- Tüm hesaplamalar 16 Kasım gece (~23:45) itibarıyla tamamlandı; grafik, tablo ve paket hazırlığı 17 Kasım'da bitti.


====================================================================================================
# BÖLÜM 6. VERİ SETLERİ
(kaynak: DATASETS.md)
====================================================================================================

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
