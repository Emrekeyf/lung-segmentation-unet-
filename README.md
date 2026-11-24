# Akciğer Röntgenlerinde Segmentasyon ve Referans Maske Analizi

Eğitim ve portföy amaçlı **iki modüllü** Bilgisayarla Görü çalışması:

* `segmentasyon/`: Median/CLAHE/Otsu, hazır `lungs\_segmentation` ResNet34 modeline dayalı maske çıkarma, karşılaştırma ve Montgomery veri kümesiyle deneyler.
* `jsrt\_dogrulama/`: JSRT görüntüleri üzerinde maske görselleştirme, veri kontrolü ve klasik/modern segmentasyon karşılaştırması.

**Bu depoda yeni bir U-Net eğitilmemektedir.** Model, `lungs\_segmentation` paketinin önceden eğitilmiş modelidir. İncelenen akademik deneylerden düzenlenen betiklerin çalışması için kullanıcı tarafından sağlanan veriler gereklidir.

## Veri klasörleri

Görseller ve maskeler, paylaşım koşulları doğrulanmadığı için GitHub'a eklenmemiştir. Dosyaları **yerel** olarak aşağıdaki konumlara yerleştirin:

```text
data/
  rontgen\_512.png             # Kaynağı doğrulanmış, izinli örnek
  stego.png                  # Opsiyonel karşılaştırma
  JSRT/
    jpg/                    # JPCLN001.jpg ...
    masks/                  # JPCLN001.tif ... (ham maskeler)
    yeni\_maskeler/          # JPCLN001.png ... (kökeni henüz doğrulanmadı)
  MontgomerySet/
    CXR\_png/
    ManualMask/leftMask/
    ManualMask/rightMask/
  test\_goruntuleri/
    COVID/
    Normal/
```

`outputs/` klasörü otomatik oluşur ve GitHub'a eklenmez. Betiklerde yollar proje köküne göre tanımlanmıştır; Windows'ta Türkçe klasör isimleri için güvenli OpenCV okuma/yazma yardımcıları eklenmiştir.

## Kurulum ve örnek kullanım

Python 3.11 önerilir. Önce sanal ortam oluşturup bağımlılıkları yükleyin. PyTorch'un uygun CPU/CUDA sürümünü bilgisayarınıza göre seçmek gerekebilir.

```cmd
py -3.11 -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe segmentasyon\\modern\_segmentasyon.py
.venv\\Scripts\\python.exe jsrt\_dogrulama\\referans\_kaynak\_kontrol.py
```

`segmentasyon/modern\_segmentasyon.py`, `data/rontgen\_512.png` ve isteğe bağlı `data/stego.png` dosyasından `outputs/modern\_sonuclar/` altına maske, ROI ve overlay üretir. Eğitim yapmaz.

## Bilimsel sınırlamalar

* `segmentasyon/toplu\_analiz.py` iki **tahmin** yönteminin (Otsu ve U-Net) örtüşmesini karşılaştırır. Bu skor **gerçek maske doğruluğu değildir**.
* `jsrt\_dogrulama/jsrt\_gercek\_dogruluk\_analizi.py` içindeki `yeni\_maskeler` klasörünün uzman referansı olduğuna dair yeterli kaynak belgesi yoktur. Bu deneylerin Dice/IoU değerleri bağımsız klinik doğrulama iddiası değildir.
* `data/JSRT/masks` TIFF dosyalarının tek taraflı maske içerebildiği gözlenmiştir. Ham ve işlenmiş maskeler doğrudan eşdeğer kabul edilmemelidir.
* `segmentasyon/gercek\_dogruluk\_analizi.py` Montgomery `ManualMask` yapısını temel alır; veri seti izinleri ve maske eşleşmeleri yerel olarak doğrulanmalıdır.
* Bu arşivde kodlar **Python sözdizimi testinden** geçirilmiştir; veri seti ve önceden eğitilmiş model bu çalışma ortamında bulunmadığından tüm deneyler uçtan uca yürütülmemiştir.

**Klinik kullanım için geliştirilmemiştir.** Veri seti lisanslarını kontrol etmeden görüntü veya hasta verisi yayımlamayın.

\## JSRT Deneysel Sonuçları (247 Görüntü)



| Ölçüt | Klasik Otsu | Hazır U-Net |

|---|---:|---:|

| Ortalama Dice | 0.6436 | 0.8667 |

| Ortalama IoU | 0.4796 | 0.7682 |



\- Başarıyla işlenen görüntü: 247/247

\- İşlem hatası: 0

\- Karşılaştırma: `yeni\_maskeler` klasöründeki işlenmiş PNG maskeleri



\*\*Bilimsel sınırlılık:\*\* PNG maskelerinin bağımsız uzman referansı olduğu henüz doğrulanmamıştır. Bu değerler klinik doğruluk kanıtı değil, mevcut maskelere göre deneysel örtüşme sonuçlarıdır.

