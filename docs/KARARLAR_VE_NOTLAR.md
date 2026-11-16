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
- VOC'ta ID seti 1000 görüntü olduğu için makaleden ~2 puana kadar sapma var (örneklem küçük, FPR95 oynak).

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

## 8. Yapılacak son kontrol (UNUTMA)
- Paketi arkadaşlara vermeden önce: bütün projeyi, kodu, sonuçları, bu notları ve sohbetin ham kaydını sanki projeyi biz yapmamışız gibi baştan sona incele. Açıklama metnini bu gözle yeniden yaz ve eleştir: eksik, yanlış, çelişen, atlanmış ayrıntı var mı? Kullanıcı özellikle uyardı: hiçbir bilgi atlanmayacak.
- Kullanıcı kuralı: sohbet iki kez sıkıştırıldı; ayrıntı kaybı riski yüzünden her karar diske (bu dosya) yazılır.

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
- Tüm hesaplamalar 16 Kasım gece (~23:45) itibarıyla tamamlandı: sırada grafik, tablo ve paket hazırlığı var.
