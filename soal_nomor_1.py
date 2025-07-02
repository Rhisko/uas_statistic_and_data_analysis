import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

# ======== 1. Baca data ========
df = pd.read_csv('data_nilai_matematika.csv')

# ======== 2. Siapkan X dan y ========
X = df[['Waktu_Belajar_Jam', 'Tidur_Jam']]
y = df['Nilai_Ujian_Matematika']

# ======== 3. Model Regresi Berganda ========
model = LinearRegression()
model.fit(X, y)

print('Intercept (β₀):', model.intercept_)
print('Koefisien Waktu_Belajar_Jam (β₁):', model.coef_[0])
print('Koefisien Tidur_Jam (β₂):', model.coef_[1])
print('\nMakna Model:')
print('- Intercept adalah nilai prediksi jika waktu belajar dan tidur = 0.')
print('- Koefisien Waktu_Belajar_Jam menunjukkan perubahan rata-rata nilai ujian untuk setiap penambahan 1 jam belajar (dengan tidur tetap).')
print('- Koefisien Tidur_Jam menunjukkan perubahan rata-rata nilai ujian untuk setiap penambahan 1 jam tidur (dengan waktu belajar tetap).\n')

# ======== 4. Uji Asumsi Klasik ========

# (a) Normalitas Residual
X_const = sm.add_constant(X)
ols = sm.OLS(y, X_const).fit()
residuals = ols.resid

plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title('Histogram Residual')
plt.show()
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot Residual')
plt.show()

# (b) Homoskedastisitas
plt.figure(figsize=(6,4))
plt.scatter(ols.fittedvalues, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

# (c) Multikolinearitas
vif = pd.DataFrame()
vif['Variable'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print('\nNilai VIF (Multikolinearitas):\n', vif)

# (d) Autokorelasi
dw = durbin_watson(residuals)
print('Durbin-Watson:', dw)

print('\nInterpretasi singkat uji asumsi klasik:')
print('- Residual normal jika histogram/kde dan Q-Q plot mengikuti distribusi normal.')
print('- Residual homoskedastis jika scatter plot residual vs fitted menyebar acak (tidak berpola).')
print('- Tidak ada multikolinearitas jika semua VIF < 10.')
print('- Tidak ada autokorelasi jika Durbin-Watson mendekati 2.')

# ======== 5. Uji Model Regresi Secara Total dan Parsial (F dan t) ========
print('\n===== SUMMARY REGRESI (statsmodels) =====')
print(ols.summary())
print('\nInterpretasi:')
print('- Uji F (Prob (F-statistic) < 0.05): Model signifikan secara total.')
print('- Uji t (P>|t| < 0.05): Masing-masing variabel signifikan secara parsial.')

# ======== 6. Koefisien Determinasi (R²) ========
print('\nKoefisien Determinasi (R²) :', ols.rsquared)
print('R² menunjukkan persentase variasi nilai ujian matematika yang dapat dijelaskan oleh waktu belajar dan tidur.\n')

# ======== 7. Prediksi baru (opsional) ========
waktu_belajar = 2.5
tidur = 7.0
prediksi_input = pd.DataFrame([[waktu_belajar, tidur]], columns=['Waktu_Belajar_Jam', 'Tidur_Jam'])
prediksi = model.predict(prediksi_input)
print(f'Contoh prediksi: Jika belajar {waktu_belajar} jam dan tidur {tidur} jam, prediksi nilai ujian = {prediksi[0]:.2f}\n')
