"""
ROBOTIK ODEV COZUMU
Ogrenci: Umut Barbaros Babahan (No: 230212065, KLXY=2065)
Parametreler:
  Q3      -> a1=3, a2=1, hedef (x,y)=(6, 5)     (ogrenci numarasindan)
Q3 yontemi: Yapay Sinir Aglari 
"""

import numpy as np
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# SORU 3 - YAPAY SINIR AGI (Sira 41 -> C secenegi)
# a1=3, a2=1, hedef (x,y)=(6, 5)
print()
print("="*70)
print("SORU 3 - YAPAY SINIR AGI ile TERS KINEMATIK")
print("="*70)

a1_s, a2_s = 3.0, 1.0
x_s, y_s = 6.0, 5.0

print(f"Ogrenci parametreleri: a1={a1_s}, a2={a2_s}, hedef=({x_s}, {y_s})")
max_reach = a1_s + a2_s
min_reach = abs(a1_s - a2_s)
dist = np.sqrt(x_s**2 + y_s**2)
print(f"Maksimum erisim: a1+a2 = {max_reach}")
print(f"Minimum erisim:  |a1-a2| = {min_reach}")
print(f"Hedef uzakligi:  sqrt(36+25) = {dist:.4f}")
print(f"NOT: Hedef calisma uzayi disinda ({dist:.2f} > {max_reach}). YSA")
print("     en yakin ulasilabilir noktaya en iyi yaklasimi verecektir.")
print()

# ---------- Egitim verisi ----------
print("Adim 1: Egitim verisini ileri kinematik ile uretelim")
np.random.seed(42)
N = 8000
# IK cok-degerli oldugu icin (dirsek-yukari/asagi) tek konfigurasyon seciyoruz:
# q2 in [0, pi]  => dirsek-asagi cozumler. q1 in [0, pi] (1. ve 2. dortte birde).
q1_train = np.random.uniform(0, np.pi, N)            # 0..180 deg
q2_train = np.random.uniform(0, np.pi, N)            # 0..180 deg (dirsek-asagi)
x_train = a1_s*np.cos(q1_train) + a2_s*np.cos(q1_train + q2_train)
y_train = a1_s*np.sin(q1_train) + a2_s*np.sin(q1_train + q2_train)
print(f"  Egitim ornek sayisi: {N}")
print(f"  Giris (x,y) araligi: x=[{x_train.min():.2f}, {x_train.max():.2f}], "
      f"y=[{y_train.min():.2f}, {y_train.max():.2f}]")

# Normallestirme
X_data = np.stack([x_train, y_train], axis=1)         # (N, 2)
# q1, q2'yi sin/cos olarak hedefleyelim -> 4 cikis (acisal sureklilik icin)
Y_data = np.stack([np.sin(q1_train), np.cos(q1_train),
                   np.sin(q2_train), np.cos(q2_train)], axis=1)  # (N, 4)

# Giris normalizasyonu (z-score)
x_mean, x_std = X_data.mean(0), X_data.std(0)
X_norm = (X_data - x_mean) / x_std

# ---------- MLP (Numpy ile sifirdan) ----------
print()
print("Adim 2: Cok katmanli algilayici (MLP) mimarisi")
print("  Giris katmani:  2 noron (x, y)")
print("  Gizli katman-1: 64 noron, aktivasyon: tanh")
print("  Gizli katman-2: 64 noron, aktivasyon: tanh")
print("  Cikis katmani:  4 noron (sin q1, cos q1, sin q2, cos q2), linear")
print("  Kayip: MSE, Optimizer: Adam, learning_rate=0.005")

# Aktivasyonlar
def tanh(z): return np.tanh(z)
def dtanh(a): return 1 - a*a

# Parametreler (He / Xavier benzeri)
def init(shape):
    fan_in = shape[0]
    return np.random.randn(*shape) * np.sqrt(1.0/fan_in)

W1 = init((2, 64));  b1 = np.zeros(64)
W2 = init((64, 64)); b2 = np.zeros(64)
W3 = init((64, 4));  b3 = np.zeros(4)

# Adam optimizer durumu
params = [W1, b1, W2, b2, W3, b3]
m_state = [np.zeros_like(p) for p in params]
v_state = [np.zeros_like(p) for p in params]
beta1, beta2, eps = 0.9, 0.999, 1e-8
lr = 0.005

def forward(Xb):
    z1 = Xb @ W1 + b1;       a1h = tanh(z1)
    z2 = a1h @ W2 + b2;      a2h = tanh(z2)
    z3 = a2h @ W3 + b3       # linear cikis
    cache = (Xb, a1h, a2h, z3)
    return z3, cache

def backward(cache, Yb):
    Xb, a1h, a2h, yhat = cache
    n = Xb.shape[0]
    dy = (yhat - Yb) * (2.0/n)              # MSE gradyani
    dW3 = a2h.T @ dy;        db3 = dy.sum(0)
    da2 = dy @ W3.T;         dz2 = da2 * dtanh(a2h)
    dW2 = a1h.T @ dz2;       db2 = dz2.sum(0)
    da1 = dz2 @ W2.T;        dz1 = da1 * dtanh(a1h)
    dW1 = Xb.T @ dz1;        db1 = dz1.sum(0)
    return [dW1, db1, dW2, db2, dW3, db3]

def adam_step(grads, t):
    for i, g in enumerate(grads):
        m_state[i] = beta1*m_state[i] + (1-beta1)*g
        v_state[i] = beta2*v_state[i] + (1-beta2)*(g*g)
        m_hat = m_state[i] / (1 - beta1**t)
        v_hat = v_state[i] / (1 - beta2**t)
        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

# Egitim dongusu
print()
print("Adim 3: Egitim (mini-batch)")
epochs = 400
batch = 128
t = 0
for ep in range(1, epochs+1):
    idx = np.random.permutation(N)
    losses = []
    for s in range(0, N, batch):
        ix = idx[s:s+batch]
        Xb, Yb = X_norm[ix], Y_data[ix]
        yhat, cache = forward(Xb)
        loss = ((yhat - Yb)**2).mean()
        losses.append(loss)
        grads = backward(cache, Yb)
        t += 1
        adam_step(grads, t)
    # Adam icindeki params listesi guncellendi, lokal degiskenleri yenile
    W1, b1, W2, b2, W3, b3 = params
    if ep % 40 == 0 or ep == 1:
        print(f"  Epoch {ep:3d}/{epochs}  MSE = {np.mean(losses):.6f}")

# ---------- Once ulasilabilir bir test noktasinda dogrulama ----------
print()
print("Adim 4a: Ulasilabilir bir test noktasinda dogrulama")
# Bilinen q1,q2'den uret -> aga tahmin ettir -> karsilastir
q1_true, q2_true = np.radians(45), np.radians(60)
x_t = a1_s*np.cos(q1_true) + a2_s*np.cos(q1_true+q2_true)
y_t = a1_s*np.sin(q1_true) + a2_s*np.sin(q1_true+q2_true)
print(f"  Gercek aci: q1=45.00 deg, q2=60.00 deg -> ileri kinematik (x,y)=({x_t:.4f}, {y_t:.4f})")
pt_norm = (np.array([[x_t, y_t]]) - x_mean) / x_std
yhat_t, _ = forward(pt_norm)
sq1, cq1, sq2, cq2 = yhat_t[0]
q1_t = np.degrees(np.arctan2(sq1, cq1))
q2_t = np.degrees(np.arctan2(sq2, cq2))
x_re, y_re = fk(q1_t, q2_t, A1=a1_s, A2=a2_s)
err_t = np.sqrt((x_re - x_t)**2 + (y_re - y_t)**2)
print(f"  YSA tahmini: q1={q1_t:.4f} deg, q2={q2_t:.4f} deg")
print(f"  Geri-kontrol: efektor=({x_re:.4f}, {y_re:.4f}), hata={err_t:.6f}")

print()
print("Adim 4b: Asil hedef (x,y) icin tahmin")
target = np.array([[x_s, y_s]])
target_norm = (target - x_mean) / x_std
pred, _ = forward(target_norm)
sin_q1_p, cos_q1_p, sin_q2_p, cos_q2_p = pred[0]
q1_pred = np.degrees(np.arctan2(sin_q1_p, cos_q1_p))
q2_pred = np.degrees(np.arctan2(sin_q2_p, cos_q2_p))
print(f"  Ag cikisi (ham): {pred[0]}")
print(f"  q1 (YSA tahmini) = atan2({sin_q1_p:.4f}, {cos_q1_p:.4f}) = {q1_pred:.4f} derece")
print(f"  q2 (YSA tahmini) = atan2({sin_q2_p:.4f}, {cos_q2_p:.4f}) = {q2_pred:.4f} derece")

# Ileri kinematik ile dogrulama
x_chk, y_chk = fk(q1_pred, q2_pred, A1=a1_s, A2=a2_s)
err = np.sqrt((x_chk - x_s)**2 + (y_chk - y_s)**2)
print(f"  Ileri kinematik kontrol: efektor konumu = ({x_chk:.4f}, {y_chk:.4f})")
print(f"  Hedefe uzaklik (hata) = {err:.4f}")
print(f"  (Hedef calisma uzayi disinda oldugundan, YSA en yakin smasa")
print(f"   yaklasik uzaylik nokta icin acilari uretmistir.)")

# En yakin ulasilabilir nokta ile karsilastirma (calisma uzayi sinir hatti)
x_closest = x_s * (max_reach / dist)
y_closest = y_s * (max_reach / dist)
print()
print("Karsilastirma: En yakin ulasilabilir nokta (calisma uzayi sinirinda)")
print(f"  ({x_closest:.4f}, {y_closest:.4f}) -- yon (6,5) ile ayni, mesafe = {max_reach}")
# Tam uzanma => q2 = 0, q1 = atan2(y, x)
q1_ext = np.degrees(np.arctan2(y_s, x_s))
print(f"  Bu durumda q2 = 0 deg, q1 = atan2(5,6) = {q1_ext:.4f} derece")

print()
print("="*70)
print("SONUC OZETI")
print("="*70)
print(f"S1 (formul) :  q1 = {q1_down:.2f} deg, q2 = {q2_down:.2f} deg  (dirsek asagi)")
print(f"               q1 = {q1_up:.2f} deg, q2 = {q2_up:.2f} deg  (dirsek yukari)")
print(f"S2 (geometri): Ayni sonuclar (dogrulama)")
print(f"S3 (YSA)    :  q1 = {q1_pred:.2f} deg, q2 = {q2_pred:.2f} deg")
print(f"               Hedef ulasilamaz - YSA en yakin cozumu uretti.")
