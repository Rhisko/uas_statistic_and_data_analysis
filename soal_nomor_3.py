from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

# Matriks kontingensi (urutan: [ [a, b], [c, d] ])
table = np.array([[70, 15],
                  [5, 25]])

# Uji McNemar (pakai exact jika sampel kecil, di sini cukup pakai chi2)
result = mcnemar(table, exact=False, correction=True)

print(f"Statistik McNemar: {result.statistic}")
print(f"p-value: {result.pvalue:.4f}")

if result.pvalue < 0.05:
    print("Terdapat perbedaan signifikan diagnosis sebelum dan sesudah penerapan teknologi baru (Tolak H0).")
else:
    print("Tidak terdapat perbedaan signifikan diagnosis sebelum dan sesudah penerapan teknologi baru (Gagal Tolak H0).")
