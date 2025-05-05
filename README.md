# Eksperimen Klasifikasi - Data Mining

Notebook ini berisi eksperimen klasifikasi menggunakan dataset `el4233-2018-2019-02-klasifikasi-train.csv` dengan model Random Forest. Berikut penjelasan lengkap setiap bagian kode yang digunakan:



## 1. Import Library dan Load Dataset

python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('el4233-2018-2019-02-klasifikasi-train.csv')
df.head()


- Mengimpor pustaka penting untuk manipulasi data, visualisasi, dan machine learning.
- Membaca dataset dan menampilkan 5 baris pertama.



## 2. Informasi Struktur Dataset

python
df.info()


- Menampilkan struktur dataset: jumlah baris, kolom, tipe data, dan data yang hilang.



## 3. Statistik Deskriptif

python
df.describe()


- Menyediakan statistik dasar seperti mean, std, min, max untuk kolom numerik.



## 4. Distribusi Kelas Target

python
df['Y'].value_counts()


- Menghitung jumlah masing-masing label pada kolom target `Y`.



## 5. Visualisasi Distribusi Kelas

python
sns.countplot(x='Y', data=df)
plt.title('Distribusi Kelas')
plt.show()


- Membuat plot batang distribusi kelas target `Y`.



## 6. Pemisahan Fitur dan Target, Split Data

python
X = df.drop('Y', axis=1)
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

- Memisahkan fitur (X) dan target (y).
- Membagi dataset menjadi data latih (80%) dan data uji (20%).
  

## 7. Pelatihan dan Evaluasi Model Random Forest

python
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

- Melatih model Random Forest.
- Memprediksi data uji.
- Menampilkan metrik evaluasi: classification report dan confusion matrix.

## Catatan

- Random Forest dipilih karena kemampuannya menangani data dengan baik tanpa banyak preprocessing.
- Distribusi kelas perlu diperhatikan untuk menghindari bias model.# Analisis-Performa-Empat-Algoritma-Klasifikasi-pada-Dataset-CSV
