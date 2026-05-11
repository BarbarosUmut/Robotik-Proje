# Robotik Ödev — 2 Eklemli Robot Kol Ters Kinematik

İki eklemli (2-DOF) düzlemsel robot kolun **ters kinematik** problemini üç farklı yöntemle çözen ödev projesi.

**Öğrenci:** Umut Barbaros Babahan (No: 230212065, KLXY = 2065)
**Ders:** Robotik

## Problem

Verilen uç-efektör konumu (x, y) için q₁ ve q₂ eklem açılarını bulmak. İleri kinematik:

```
x = a₁·cos(q₁) + a₂·cos(q₁ + q₂)
y = a₁·sin(q₁) + a₂·sin(q₁ + q₂)
```

## İçerik

| Dosya | Açıklama |
|---|---|
| `odev1ve2.py` | **Soru 1 & 2** — Analitik (kosinüs teoremi + atan2) ve geometrik ters kinematik. a₁=3, a₂=2, hedef (2, 3.5). |
| `homework_solution.py` | **Soru 3** — Yapay Sinir Ağı (NumPy ile sıfırdan MLP) ile ters kinematik. a₁=3, a₂=1, hedef (6, 5). |
| `RAPOR.md` | Yöntemlerin detaylı açıklaması, denklem türetimleri ve sonuç karşılaştırması. |
| `HOMEWORK.pdf` | Orijinal ödev metni. |

## Kurulum & Çalıştırma

```bash
pip install -r requirements.txt

# Soru 1 ve 2 (analitik + geometrik)
python odev1ve2.py

# Soru 3 (yapay sinir ağı)
python homework_solution.py
```

## YSA Mimarisi (Soru 3)

- Giriş: 2 nöron (x, y)
- Gizli katmanlar: 64 → 64 (tanh aktivasyon)
- Çıkış: 4 nöron — [sin q₁, cos q₁, sin q₂, cos q₂] (açısal süreklilik için)
- Optimizer: Adam, lr=0.005, batch=128, 400 epoch
- Eğitim verisi: 8000 sentetik örnek (ileri kinematikten)

## Sonuçlar

| Yöntem | q₁ | q₂ |
|---|---:|---:|
| Soru 1/2 — dirsek aşağı | 31.73° | 74.29° |
| Soru 1/2 — dirsek yukarı | 88.78° | -74.29° |
| Soru 3 — YSA (hedef (6,5)) | 72.13° | -55.06° |

> **Not:** Soru 3'teki hedef (6, 5) noktası kol erişim alanı dışındadır (√61 ≈ 7.81 > a₁+a₂ = 4). YSA bu durumda en yakın yaklaşımı verir. Detaylar için bkz. `RAPOR.md`.

## Gereksinimler

- Python 3.12+
- NumPy

Üçüncü-parti makine öğrenmesi kütüphanesi (PyTorch, TensorFlow, scikit-learn vb.) kullanılmamıştır — MLP tamamen NumPy ile sıfırdan yazılmıştır.
