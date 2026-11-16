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
