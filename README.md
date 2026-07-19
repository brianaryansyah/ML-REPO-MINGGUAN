# ML-REPO-MINGGUAN

Repository ini berisi tugas-tugas Pembelajaran Mesin (Machine Learning) mingguan. Pada minggu 11-12 ini, fokus pembelajaran adalah **Multiple Linear Regression**.

---

## Minggu 11-12: Multiple Linear Regression

### Overview
Proyek ini membangun model **Multiple Linear Regression** untuk memprediksi harga rumah di **King County, Washington, USA** berdasarkan beberapa fitur properti. Data diambil dari Kaggle dengan sedikit modifikasi.

### Dataset
File: `kc_house_data.csv`

| Fitur | Deskripsi |
|-------|-----------|
| `bedrooms` | Jumlah kamar tidur |
| `bathrooms` | Jumlah kamar mandi |
| `sqft_living` | Luas rumah dalam satuan sqft |
| `grade` | Grading system dari pemerintah King County |
| `yr_built` | Tahun pembangunan rumah |
| `price` | Harga rumah dalam US$ (target/dependent variable) |

Total data: **21.613 baris** x **6 kolom**

### Workflow
1. **Load Library** - pandas, matplotlib, numpy, seaborn, scikit-learn
2. **Load Dataset** - Membaca data dari `kc_house_data.csv`
3. **Data Preprocessing:**
   - Mengubah tipe data `bathrooms` dari float ke int
   - Mengatasi typo pada `bedrooms` (nilai 33 diganti menjadi 3)
   - Mengecek missing values (tidak ditemukan)
4. **Exploratory Data Analysis (EDA):**
   - Univariate analysis: distribusi dan boxplot untuk setiap fitur
   - Bivariate analysis: pairplot hubungan antar fitur dengan harga
   - Heatmap korelasi: `sqft_living` memiliki korelasi terkuat dengan `price` (0.70)
5. **Modelling:**
   - Split data training/testing dengan rasio 80:20 (`random_state=4`)
   - Membuat dan melatih model `LinearRegression`
   - Evaluasi model menggunakan R-squared score
6. **Prediction** - Memprediksi harga rumah berdasarkan input spesifik

### Hasil Model

**Persamaan Regresi:**
```
Y = -53061.75(bedrooms) + 64658.56(bathrooms) + 188.91(sqft_living) + 131290.89(grade) - 3969.56(yr_built) + 7031568
```

**Koefisien tiap fitur:**

| Fitur | Koefisien | Interpretasi |
|-------|-----------|-------------|
| bedrooms | -53,061.75 | Setiap tambah 1 kamar tidur, harga turun ~$53K (negatif karena multikolinearitas) |
| bathrooms | +64,658.56 | Setiap tambah 1 kamar mandi, harga naik ~$64K |
| sqft_living | +188.91 | Setiap tambah 1 sqft luas rumah, harga naik ~$189 |
| grade | +131,290.89 | Setiap naik 1 grade, harga naik ~$131K (fitur paling berpengaruh) |
| yr_built | -3,969.56 | Rumah baru cenderung sedikit lebih murah (korelasi hampir nol) |

**Akurasi Model:**
- R-squared Score: **0.6125 (61.25%)**

**Prediksi Harga Rumah Impian Joko:**
- Kriteria: 3 kamar tidur, 2 kamar mandi, 1800 sqft, grade 7, tahun 1990
- **Hasil Prediksi: ~$361,352**

### Key Findings
- `sqft_living` dan `grade` adalah fitur yang paling mempengaruhi harga rumah
- `yr_built` hampir tidak berpengaruh terhadap harga (korelasi ~0.05)
- Model memiliki akurasi 61.25%, yang menunjukkan masih ada faktor lain yang mempengaruhi harga rumah

### Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

### Cara Menjalankan
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Jalankan notebook
jupyter notebook "Multivariate-Linear-Regression/Multivariate Linear Regression.ipynb"
```

---

## Struktur Repository

```
ML-REPO-MINGGUAN/
├── README.md
└── Multivariate-Linear-Regression/
    ├── Multivariate Linear Regression.ipynb
    └── kc_house_data.csv
```

---

**Author:** Brian Aryansyah  
**Mata Kuliah:** Pembelajaran Mesin (Machine Learning)  
**Semester:** 4  
**Minggu ke:** 11-12
