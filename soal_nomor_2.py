import pandas as pd
import numpy as np
from scipy.stats import kruskal, friedmanchisquare

# 1. Membuat data
data = {
    "No": list(range(1, 16)),
    "Algoritma A": [1.5,1.8,1.6,1.7,1.9,2.0,2.2,1.4,1.6,1.7,1.8,1.9,2.1,1.5,1.7],
    "Algoritma B": [1.7,1.6,1.9,1.8,2.0,2.1,2.3,1.5,1.8,1.6,2.0,2.1,2.3,1.8,2.0],
    "Algoritma C": [2.0,1.9,2.1,2.2,2.3,2.4,2.5,1.7,2.0,1.9,2.2,2.3,2.6,2.1,2.2]
}
df = pd.DataFrame(data)
print("Tabel waktu eksekusi ketiga algoritma (dalam detik):\n")
print(df.to_string(index=False))

# 2. Ekstrak data per algoritma
alg_a = df["Algoritma A"]
alg_b = df["Algoritma B"]
alg_c = df["Algoritma C"]

# 3. Uji Kruskal-Wallis
stat_kw, p_kw = kruskal(alg_a, alg_b, alg_c)
print(f"\nHasil uji Kruskal-Wallis:\n  Statistik = {stat_kw:.3f}, p-value = {p_kw:.4f}")
if p_kw < 0.05:
    print("  => Ada perbedaan signifikan waktu eksekusi di antara ketiga algoritma (Kruskal-Wallis).")
else:
    print("  => Tidak ada perbedaan signifikan waktu eksekusi di antara ketiga algoritma (Kruskal-Wallis).")

# 4. Uji Friedman
stat_fr, p_fr = friedmanchisquare(alg_a, alg_b, alg_c)
print(f"\nHasil uji Friedman:\n  Statistik = {stat_fr:.3f}, p-value = {p_fr:.4f}")
if p_fr < 0.05:
    print("  => Ada perbedaan signifikan waktu eksekusi di antara ketiga algoritma (Friedman).")
else:
    print("  => Tidak ada perbedaan signifikan waktu eksekusi di antara ketiga algoritma (Friedman).")

# 5. Interpretasi (otomatis)
print("\nInterpretasi:")
print("- Jika p-value < 0.05 pada salah satu uji, berarti terdapat perbedaan kinerja (waktu eksekusi) yang signifikan antar algoritma.")
print("- Jika p-value >= 0.05 pada kedua uji, berarti tidak terdapat perbedaan kinerja yang signifikan.")
