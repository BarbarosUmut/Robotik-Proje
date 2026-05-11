"""
ROBOTIK ODEV COZUMU
Ogrenci: Umut Barbaros Babahan (No: 230212065, KLXY=2065)
Parametreler:
  Q1, Q2  -> a1=3, a2=2, hedef (x,y)=(2, 3.5)   (sabit, odevde verilen)
"""
import numpy as np
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# SORU 1 - Analitik (formul) ile ters kinematik
# a1=3, a2=2, hedef (x,y)=(2, 3.5)
print("="*70)
print("SORU 1 - ANALITIK TERS KINEMATIK (Formul)")
print("="*70)

a1, a2 = 3.0, 2.0
x, y = 2.0, 3.5

print(f"Veriler: a1={a1}, a2={a2}, hedef (x,y)=({x}, {y})")
print()
print("Adim 1: q2 acisi icin Cosinus Teoremi")
print("  cos(q2) = (x^2 + y^2 - a1^2 - a2^2) / (2*a1*a2)")

r2 = x**2 + y**2
print(f"  x^2 + y^2 = {x}^2 + {y}^2 = {r2}")
print(f"  a1^2 + a2^2 = {a1**2} + {a2**2} = {a1**2 + a2**2}")
print(f"  2*a1*a2 = {2*a1*a2}")

c2 = (r2 - a1**2 - a2**2) / (2*a1*a2)
print(f"  cos(q2) = ({r2} - {a1**2+a2**2}) / {2*a1*a2} = {c2:.6f}")

s2_pos = np.sqrt(1 - c2**2)
s2_neg = -s2_pos
print(f"  sin(q2) = +/- sqrt(1 - cos^2(q2)) = +/- {s2_pos:.6f}")

# Iki cozum: elbow-down (s2>0) ve elbow-up (s2<0)
q2_down = np.degrees(np.arctan2(s2_pos, c2))
q2_up   = np.degrees(np.arctan2(s2_neg, c2))
print(f"  -> q2 (dirsek asagi) = atan2({s2_pos:.4f}, {c2:.4f}) = {q2_down:.4f} derece")
print(f"  -> q2 (dirsek yukari) = atan2({s2_neg:.4f}, {c2:.4f}) = {q2_up:.4f} derece")

print()
print("Adim 2: q1 acisi")
print("  q1 = atan2(y, x) - atan2(a2*sin(q2), a1 + a2*cos(q2))")

phi = np.degrees(np.arctan2(y, x))
print(f"  atan2(y, x) = atan2({y}, {x}) = {phi:.4f} derece")

# Dirsek asagi
psi_down = np.degrees(np.arctan2(a2*s2_pos, a1 + a2*c2))
q1_down = phi - psi_down
print(f"  Dirsek asagi: atan2({a2*s2_pos:.4f}, {a1+a2*c2:.4f}) = {psi_down:.4f}")
print(f"               q1 = {phi:.4f} - {psi_down:.4f} = {q1_down:.4f} derece")

# Dirsek yukari
psi_up = np.degrees(np.arctan2(a2*s2_neg, a1 + a2*c2))
q1_up = phi - psi_up
print(f"  Dirsek yukari: atan2({a2*s2_neg:.4f}, {a1+a2*c2:.4f}) = {psi_up:.4f}")
print(f"                q1 = {phi:.4f} - ({psi_up:.4f}) = {q1_up:.4f} derece")

print()
print("SONUC (Soru 1):")
print(f"  Cozum-A (dirsek asagi):  q1 = {q1_down:.4f} deg, q2 = {q2_down:.4f} deg")
print(f"  Cozum-B (dirsek yukari): q1 = {q1_up:.4f} deg, q2 = {q2_up:.4f} deg")

# Dogrulama (ileri kinematik)
def fk(q1d, q2d, A1=a1, A2=a2):
    q1r, q2r = np.radians(q1d), np.radians(q2d)
    xx = A1*np.cos(q1r) + A2*np.cos(q1r+q2r)
    yy = A1*np.sin(q1r) + A2*np.sin(q1r+q2r)
    return xx, yy

print("Dogrulama (ileri kinematik):")
print(f"  Cozum-A:  ({fk(q1_down, q2_down)[0]:.4f}, {fk(q1_down, q2_down)[1]:.4f})")
print(f"  Cozum-B:  ({fk(q1_up, q2_up)[0]:.4f}, {fk(q1_up, q2_up)[1]:.4f})")

# SORU 2 - Geometrik cozum
print()
print("="*70)
print("SORU 2 - GEOMETRIK TERS KINEMATIK")
print("="*70)
print(f"Veriler: a1={a1}, a2={a2}, hedef (x,y)=({x}, {y})")
print()
print("Adim 1: Hedef noktanin orijine uzakligi (r)")
r = np.sqrt(x**2 + y**2)
print(f"  r = sqrt(x^2 + y^2) = sqrt({x**2} + {y**2}) = sqrt({r**2}) = {r:.6f}")

print()
print("Adim 2: Alpha acisi (hedefin x ekseni ile yaptigi aci)")
alpha = np.degrees(np.arctan2(y, x))
print(f"  alpha = atan2(y, x) = atan2({y}, {x}) = {alpha:.4f} derece")

print()
print("Adim 3: a1, a2, r kenarli ucgende beta acisi (a1 ile r arasi)")
print("  Cosinus teoremi: a2^2 = a1^2 + r^2 - 2*a1*r*cos(beta)")
print("  => cos(beta) = (a1^2 + r^2 - a2^2) / (2*a1*r)")
cos_beta = (a1**2 + r**2 - a2**2) / (2*a1*r)
beta = np.degrees(np.arccos(cos_beta))
print(f"  cos(beta) = ({a1**2} + {r**2:.4f} - {a2**2}) / (2*{a1}*{r:.4f}) = {cos_beta:.6f}")
print(f"  beta = arccos({cos_beta:.4f}) = {beta:.4f} derece")

print()
print("Adim 4: a1 ile a2 arasindaki ic aci (gamma)")
print("  Cosinus teoremi: r^2 = a1^2 + a2^2 - 2*a1*a2*cos(gamma)")
print("  => cos(gamma) = (a1^2 + a2^2 - r^2) / (2*a1*a2)")
cos_gamma = (a1**2 + a2**2 - r**2) / (2*a1*a2)
gamma = np.degrees(np.arccos(cos_gamma))
print(f"  cos(gamma) = ({a1**2+a2**2} - {r**2:.4f}) / {2*a1*a2} = {cos_gamma:.6f}")
print(f"  gamma = arccos({cos_gamma:.4f}) = {gamma:.4f} derece")

print()
print("Adim 5: q1 ve q2 acilarinin hesaplanmasi")
print("  Dirsek asagi (elbow-down): q1 = alpha - beta, q2 = 180 - gamma")
q1_geo_down = alpha - beta
q2_geo_down = 180 - gamma
print(f"    q1 = {alpha:.4f} - {beta:.4f} = {q1_geo_down:.4f} derece")
print(f"    q2 = 180 - {gamma:.4f} = {q2_geo_down:.4f} derece")

print("  Dirsek yukari (elbow-up): q1 = alpha + beta, q2 = -(180 - gamma)")
q1_geo_up = alpha + beta
q2_geo_up = -(180 - gamma)
print(f"    q1 = {alpha:.4f} + {beta:.4f} = {q1_geo_up:.4f} derece")
print(f"    q2 = -(180 - {gamma:.4f}) = {q2_geo_up:.4f} derece")

print()
print("SONUC (Soru 2): Soru 1 ile ayni sonuclar")
print(f"  Dirsek asagi:  q1 = {q1_geo_down:.4f} deg, q2 = {q2_geo_down:.4f} deg")
print(f"  Dirsek yukari: q1 = {q1_geo_up:.4f} deg, q2 = {q2_geo_up:.4f} deg")