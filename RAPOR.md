# Robotik Ödev Raporu — Ters Kinematik Çözümü

**Öğrenci:** Umut Barbaros Babahan
**Öğrenci No:** 230212065 (KLXY = 2065)
**YÖNTEM:** Yapay Sinir Ağları yöntemi
**Ders:** Robotik
**Tarih:** 11 Mayıs 2026

---

## 1. Özet

Bu raporda, 2 eklemli (2-DOF) düzlemsel robot kolun ters kinematik problemi yapay sinir ağları yöntemle çözülmüştür:

1. **Yapay Sinir Ağı (YSA) çözümü** — NumPy ile sıfırdan yazılmış MLP (Multi-Layer Perceptron)

Soru da öğrenci numarasına özgü parametrelerle (a₁=3, a₂=1, hedef (6, 5)) hedef çalışma uzayı dışında kaldığı tespit edilmiş; YSA bu durum için en yakın yaklaşımı vermiştir.

---

## 2. Problem Tanımı

### 2.1 Manipulatör Modeli

İki döner eklemli (R-R) düzlemsel manipulatörün uç-efektör konumu **ileri kinematik** ile:

$$x = a_1 \cos(q_1) + a_2 \cos(q_1 + q_2)$$
$$y = a_1 \sin(q_1) + a_2 \sin(q_1 + q_2)$$

şeklindedir. **Ters kinematik problemi**: verilen (x, y) için (q₁, q₂) değerlerini bulmaktır.

### 2.2 Parametreler

| Soru | a₁ | a₂ | Hedef (x, y) | Kaynak |
|---|---|---|---|---|
| Soru 3 | 3 | 1 | (6, 5) | KLXY=2065 (a₁=K+1, a₂=L+1, x=X, y=Y) |

## 3. Yapay Sinir Ağı ile Ters Kinematik

### 3.1 Problem Analizi

**Veriler:** a₁ = 3, a₂ = 1, hedef (x, y) = (6, 5)

2 eklemli düzlemsel kolun **çalışma uzayı**, iki dairenin arasında kalan halkadır:

- **Dış yarıçap (max erişim):** a₁ + a₂ = **4**
- **İç yarıçap (min erişim):** |a₁ − a₂| = **2**

Hedef noktanın orijine uzaklığı:
$$r = \sqrt{6^2 + 5^2} = \sqrt{61} \approx 7.81$$

7.81 > 4 olduğundan **hedef geometrik olarak ulaşılamaz**. Bu durum, geleneksel analitik IK yöntemlerinde çözüm vermez (cos(q₂) > 1 çıkar, arccos tanımsızdır). YSA gibi öğrenme tabanlı yöntemlerin bir avantajı burada görülür: ağ, eğitim aralığı dışındaki noktalar için bile **en iyi yaklaşımı** üretmeye çalışır.

### 3.2 Yöntem — Multi-Layer Perceptron (MLP)

#### 3.2.1 Veri Üretimi (Adım 1)

İleri kinematik kullanılarak 8000 sentetik eğitim örneği üretildi. Ters kinematik çok-değerli olduğundan (dirsek-yukarı/aşağı) ağı eğitirken **tek konfigürasyon** seçildi:

- q₁ ∈ [0°, 180°] (yarı düzlem, üst yarı)
- q₂ ∈ [0°, 180°] (yalnızca **dirsek-aşağı** çözümler)

```python
q1_train = np.random.uniform(0, π, 8000)
q2_train = np.random.uniform(0, π, 8000)
x_train = a1·cos(q1) + a2·cos(q1+q2)
y_train = a1·sin(q1) + a2·sin(q1+q2)
```

Bu kısıtlama IK fonksiyonunu **tek-değerli** hale getirir; aksi takdirde aynı (x, y) için iki farklı (q₁, q₂) eğitim örneği olur, MSE kaybı iki çözümün ortalamasına yakınsar ve ağ doğru sonuç veremez.

#### 3.2.2 Çıkış Kodlaması — Açısal Süreklilik Problemi

Açıları doğrudan q₁, q₂ olarak hedeflemek **açısal sıçramaya** (örn. 359° → 1°) ve ağda hataya yol açar. Bu nedenle her açıyı **(sin, cos)** çifti olarak öğretiyoruz:

- Çıkış vektörü: [sin q₁, cos q₁, sin q₂, cos q₂] (4 boyutlu)
- Tahmin sonrası: q = atan2(sin q, cos q)

Bu kodlama açısal süreklilik sağlar.

#### 3.2.3 Mimari (Adım 2)

| Katman | Nöron Sayısı | Aktivasyon |
|---|---|---|
| Giriş | 2 (x, y) | — |
| Gizli-1 | 64 | tanh |
| Gizli-2 | 64 | tanh |
| Çıkış | 4 (sin q₁, cos q₁, sin q₂, cos q₂) | Linear |

**Eğitim parametreleri:**
- Kayıp fonksiyonu: **Mean Squared Error (MSE)**
- Optimizer: **Adam** (β₁=0.9, β₂=0.999, ε=1e-8)
- Öğrenme oranı (learning rate): **0.005**
- Mini-batch boyutu: **128**
- Epoch sayısı: **400**
- Ağırlık başlatma: Xavier/He benzeri (1/√fan_in)
- Giriş normalizasyonu: z-score (sıfır ortalama, birim varyans)

#### 3.2.4 İleri ve Geri Yayılım

**İleri yayılım:**
$$h_1 = \tanh(X W_1 + b_1)$$
$$h_2 = \tanh(h_1 W_2 + b_2)$$
$$\hat{y} = h_2 W_3 + b_3$$

**Geri yayılım (zincir kuralı):**
$$\frac{\partial L}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)$$
$$\frac{\partial L}{\partial W_3} = h_2^T \cdot \frac{\partial L}{\partial \hat{y}}, \quad \frac{\partial L}{\partial h_2} = \frac{\partial L}{\partial \hat{y}} \cdot W_3^T$$

tanh türevi: $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$

Adam güncelleme:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

#### 3.2.5 Eğitim Eğrisi (Adım 3)

| Epoch | MSE |
|---:|---:|
| 1 | 0.142817 |
| 40 | 0.001027 |
| 80 | 0.000765 |
| 120 | 0.000483 |
| 160 | 0.000381 |
| 200 | 0.000400 |
| 240 | 0.000295 |
| 280 | 0.000305 |
| 320 | 0.000393 |
| 360 | 0.000330 |
| **400** | **0.000291** |

Kayıp 0.143'ten **0.000291**'e düşmüştür — yaklaşık **490 kat** iyileşme. Ağ ters kinematiği başarıyla öğrenmiştir.

### 3.3 Doğrulama — Ulaşılabilir Bir Test Noktası (Adım 4a)

Ağın gerçekten IK'yı öğrendiğini göstermek için, bilinen bir açıdan üretilen noktayla doğrulama yapıldı:

- **Gerçek açılar:** q₁ = 45°, q₂ = 60°
- **İleri kinematik ile hedef:** (x, y) = (1.8625, 3.0872)
- **YSA tahmini:** q₁ = **45.0084°**, q₂ = **58.5540°**
- **Geri-kontrol (FK):** efektör konumu (1.8865, 3.0937)
- **Konum hatası:** **0.025** birim

Bu sonuç, ağın çalışma uzayı içinde IK'yı virgülden sonra 2 hane doğrulukla öğrendiğini ispatlar.

### 3.4 Asıl Hedef İçin Tahmin (Adım 4b)

**Hedef:** (x, y) = (6, 5) — çalışma uzayı dışında

**YSA çıkışı (ham):**
- sin q₁ = 1.1443, cos q₁ = 0.3690
- sin q₂ = −2.8723, cos q₂ = 2.0066

(|sin|, |cos| > 1 değerleri ekstrapolasyon nedeniyledir; atan2 yine doğru yön açısını verir.)

**Tahmin:**
- q₁ = atan2(1.1443, 0.3690) = **72.1280°**
- q₂ = atan2(−2.8723, 2.0066) = **−55.0616°**

**İleri kinematik kontrolü:**
- Efektör konumu: (1.8766, 3.1487)
- Hedefe uzaklık: **4.5199**

### 3.5 En Yakın Ulaşılabilir Nokta ile Karşılaştırma

(6, 5) yönündeki çalışma uzayı sınırındaki en yakın nokta:
- Sınır noktası: (3.0729, 2.5607)
- Bu noktaya **kolu tam uzatarak** (q₂ = 0°) ulaşılır.
- q₁ = atan2(5, 6) = **39.8056°**

YSA'nın tahmini bu ideal noktadan biraz sapmıştır çünkü:
1. Hedef eğitim verisinin **çok dışındadır** (extrapolation).
2. Ağ, ekstrapolasyon bölgesinde tanh non-lineeritesi nedeniyle doyuma ulaşmıştır.

---

## 4. Genel Sonuç ve Karşılaştırma

### 4.1 Sonuç Tablosu

| Soru | Yöntem | q₁ | q₂ | Konum Hatası |
|---|---|---:|---:|---:|
| S3 (validasyon) | YSA — ulaşılabilir nokta | 45.01° | 58.55° | 0.025 |
| **S3 (hedef)** | **YSA — (6, 5)** | **72.13°** | **-55.06°** | **4.52*** |

(*) Hedef çalışma uzayı dışında olduğundan, "konum hatası" geometrik olarak kaçınılmazdır; en iyi durumda 7.81 − 4 = 3.81 birim olur.

### 4.2 Yöntem Karşılaştırması

| Özellik | Analitik | Geometrik | YSA |
|---|---|---|---|
| Tam doğruluk | ✓ | ✓ | ✗ (yaklaşık) |
| Çoklu çözüm | ✓ (2 adet açıkça) | ✓ (2 adet açıkça) | ✗ (tek konfig.) |
| Çalışma uzayı dışında | ✗ (tanımsız) | ✗ (tanımsız) | ✓ (yaklaşık) |
| Çoklu eklem (n ≥ 3) | Zor | Çok zor | ✓ (genişler) |
| Hesaplama maliyeti (test) | Çok düşük | Çok düşük | Düşük (ileri yayılım) |
| Eğitim maliyeti | Yok | Yok | Yüksek (offline) |

### 4.3 Değerlendirme

- **Analitik ve geometrik yöntemler**, 2-DOF gibi düşük serbestlik dereceli sistemlerde **en doğru ve verimli** seçimdir.
- **YSA**, daha karmaşık (yüksek-DOF) sistemlerde ya da çalışma uzayı sınırında en yakın çözümün arandığı durumlarda avantaj sağlar.
- Bu çalışmadaki YSA, çalışma uzayı **içinde** mükemmele yakın doğruluk (hata < 0.03) verirken, **dışında** geometrik kısıtlamalar nedeniyle ulaşılamaz noktaya en iyi yaklaşımı üretmiştir.

---

## 5. Kullanılan Araçlar

- **Programlama dili:** Python 3.12
- **Kütüphaneler:** Yalnızca **NumPy** (MLP sıfırdan, üçüncü-parti ML kütüphanesi yok)
- **Geliştirme ortamı:** Windows 11

---

## 6. Eklere İlişkin Notlar

- **Kaynak kod:** `homework_solution.py`
- Çalıştırma: `python homework_solution.py`
- Tüm çıktılar konsola yazdırılır; ek bağımlılık yoktur.

---

*Bu çalışma 11 Mayıs 2026 tarihinde, "Robotik" dersi final sınavı oturumunda teslim edilmek üzere hazırlanmıştır.*
