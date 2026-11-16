# GL-MCM vize projesi: buradan başlayın

Bu depo, COME421/1 (Introduction to Computer Vision) vize ödevinin deney kısmını içerir: Miyai ve ark. "GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot OOD Detection" (IJCV 2025) makalesinin MCM / L-MCM / GL-MCM yöntemlerinin ve ImageNet tablosundaki MSP / Energy / ODIN yöntemlerinin uygulanması, makalenin veri setlerinde çalıştırılması, iki kavramın (yerel özellik, sıcaklık) var/yok kıyası. Deneyler, tablolar ve şekiller hazırdır. Rapor yazımı ekip arkadaşlarına aittir.

## Okuma sırası
1. `docs/PROJE_ACIKLAMASI.md`: her şeyin ayrıntılı anlatımı (ana kaynak).
2. `docs/DOSYA_VE_GORSEL_REHBERI.md`: her dosya ve şekil.
3. `docs/RESULT_TABLES.md`, `tables/`: sayılar.
4. `docs/AGENT_PROMPT.md`: yapay zekâ asistanı kullanacaksanız hazır talimat (önce dersin kuralını kontrol edin).
5. `docs/KARARLAR_VE_NOTLAR.md`: ham karar notları.
6. `DATASETS.md`: veri setleri (sadece deneyleri sıfırdan tekrarlayacaksanız).

## Sık işler
- Tablo ve şekilleri yeniden üret (veri seti gerekmez): `pip install numpy matplotlib` sonra sırayla `python make_figures.py` ve `python make_tables.py` (sıra önemli: λ tablosunu önce şekil betiği üretir)
- Metrik doğrulaması: `python make_figures.py --check`
- Nihai sonuçlar: `results/<ID>/<Yöntem>/..._final/` ve `results_raw/`

## Teslim (ekip tarafından yapılacak)
- Tek zip/rar, STIX bağlantısından, dosya başına ≤50 MB, son tarih **19.11.2026 01:00** (18 Kasım gecesi). Geç veya başka kanaldan teslim kabul edilmiyor.
- Kapak sayfasında her üyenin ad, soyad ve numarası olmalı (eksik üye sıfır alır).
- Zip'e rapor + kod + `figures/` + `tables/` + `results/` + `results_raw/` + `docs/` + `DATASETS.md` + `requirements.txt` girer; veri setleri girmez.
- 18 Kasım 14:00 çevrimiçi ders (Meet), devam %70.
