# Pembelajaran Mesin - Mingguan

Repository tugas praktikum mata kuliah **Pembelajaran Mesin (Machine Learning)** - Semester 4, Universitas Dian Nuswantoro (UDINUS).

## Struktur Repository

```
.
├── acquisition_engine.py          # Pipeline akuisisi & profiling data
├── 03-data-acquisition/           # Modul akuisisi data
├── minggu 11-12/                  # Regresi Linear Berganda
│   └── Multivariate Linear Regression.ipynb
└── minggu 13-14/                  # Clustering (K-Means & Hierarchical)
    ├── k-means/
    │   └── k_means_clustering.ipynb
    └── hierarchical/
        └── hierarchical_clustering.ipynb
```

---

## Minggu 11-12: Regresi Linear Berganda (Multiple Linear Regression)

### Topik
Prediksi harga rumah menggunakan **Multiple Linear Regression** dengan 6 fitur: jumlah kamar tidur, kamar mandi, luas bangunan (sqft), grade, tahun dibangun, dan harga.

### Dataset
**King County House Sales** (Kaggle) - 21.613 data, 21 kolom (6 digunakan).

### Alur Kerja
1. Load data & seleksi fitur
2. Data cleaning (konversi tipe, perbaikan outlier)
3. EDA: univariate analysis (histogram, boxplot, KDE), bivariate analysis (pairplot), korelasi heatmap
4. Train-test split (80/20)
5. Training model `LinearRegression`
6. Evaluasi & prediksi

### Hasil Utama
| Metrik | Nilai |
|--------|-------|
| R-squared (Accuracy) | **61.25%** |
| Fitur paling berpengaruh | `grade` (+131.291), `bathrooms` (+64.659) |
| Korelasi terkuat dengan harga | `sqft_living` (0.70) |

### Libraries
`pandas` `numpy` `matplotlib` `seaborn` `scikit-learn`

---

## Minggu 13-14: Clustering (Unsupervised Learning)

Mengelompokkan pelanggan mall menjadi segmen-segmen berdasarkan **Annual Income** dan **Spending Score** menggunakan dua pendekatan clustering.

### Dataset
**Mall Customers** - 200 data, 5 kolom (`CustomerID`, `Genre`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`).

---

### 1. K-Means Clustering

**Algoritma:** `sklearn.cluster.KMeans` (k-means++ init, n_init=10)

**Alur Kerja:**
1. Load data & EDA (statistik deskriptif, distribusi, scatter plot)
2. **Feature Scaling** dengan `StandardScaler` (mean=0, std=1)
3. **Elbow Method + Silhouette Score** side-by-side (k=2 s/d 10) - optimal **k=5**
4. Training K-Means dengan 5 cluster & distribusi per cluster
5. **Cluster Profiling** - label otomatis berdasarkan threshold Income & Spending
6. Visualisasi: scatter plot + centroid, 3-panel (Income-Spending, Age-Income, Age-Spending)
7. **Model Evaluation:** Silhouette Score + Davies-Bouldin Score per cluster
8. **Silhouette Plot** per cluster
9. **Perbandingan Init Method:** k-means++ vs random
10. Kesimpulan & business insights

**Hasil:**
| Metrik | Nilai |
|--------|-------|
| Jumlah Cluster Optimal | **5** |
| Init Method Terbaik | **k-means++** |
| Silhouette Score | **~0.55** |
| Davies-Bouldin Score | **~0.57** |

**Profil Cluster:**

| Cluster | Profil | Keterangan |
|---------|--------|------------|
| High Income - High Spending | Target utama produk premium |
| High Income - Low Spending | Perlu strategi marketing untuk meningkatkan spending |
| Low Income - High Spending | Target untuk promosi dan diskon |
| Low Income - Low Spending | Pasar potensial dengan strategi yang tepat |
| Moderate | Segment middle-market, cocok untuk loyalty programs |

---

### 2. Hierarchical (Agglomerative) Clustering

**Algoritma:** `sklearn.cluster.AgglomerativeClustering` (Ward linkage)

**Alur Kerja:**
1. Load data & EDA (statistik deskriptif, distribusi, scatter plot)
2. **Feature Scaling** dengan `StandardScaler` (mean=0, std=1)
3. **Dendrogram Analysis** - Ward linkage, cutting distance ~7 menghasilkan 5 cluster
4. **Perbandingan Linkage Methods:** Ward, Complete, Average, Single
5. **Elbow + Silhouette Score** - k=5 dengan Silhouette Score tertinggi (0.5538)
6. Training model & cluster profiling
7. Evaluasi Silhouette Score per cluster
8. Business insights

**Hasil - Profil Cluster:**

| Cluster | Profil | Jumlah | Rata-rata Income | Rata-rata Spending |
|---------|--------|--------|-------------------|---------------------|
| 0 | High Income - Low Spending | 32 | $89.4k | 15.6 |
| 1 | High Income - High Spending | 39 | $86.5k | 82.1 |
| 2 | Moderate Income - Moderate Spending | 85 | $55.8k | 49.1 |
| 3 | Low Income - High Spending | 21 | $25.1k | 80.0 |
| 4 | Low Income - Low Spending | 23 | $26.3k | 20.9 |

**Ranking Linkage Methods (Silhouette Score):**
1. Ward: 0.5538
2. Complete: 0.5531
3. Average: 0.4794
4. Single: 0.2758

**Business Insights:**
- **High Income + High Spending** -> Target utama produk premium
- **High Income + Low Spending** -> Perlu strategi marketing untuk meningkatkan spending
- **Low Income + High Spending** -> Target promosi dan diskon
- **Low Income + Low Spending** -> Pasar potensial dengan strategi yang tepat
- **Moderate** -> Segmen middle-market, cocok untuk loyalty programs

### Libraries
`numpy` `pandas` `matplotlib` `seaborn` `scikit-learn` `scipy`

---

## Data Acquisition Engine

Pipeline untuk akuisisi data dari berbagai sumber dengan laporan profiling komprehensif.

### `MLAcquisitionEngine`
- **Manual CSV** - Muat dataset dari file lokal
- **API** - Tarik data via REST API
- **Database SQL** - Query ke database relasional (SQLAlchemy)

### `DataProfiler`
Laporan profiling 5 section:
1. **Dataset Overview** - Records, features, memory footprint, duplikat, missing values
2. **Schema & Feature Classification** - Tipe data dan skala pengukuran
3. **Numerical Features Summary** - Statistik deskriptif
4. **Categorical Features Distribution** - Distribusi kategori dengan visual
5. **Data Quality Assessment** - Skor kualitas 0-100 dengan rating

```bash
python acquisition_engine.py
```

---

## Teknik & Metodologi yang Digunakan

| Aspek | Teknik |
|-------|--------|
| **Preprocessing** | Missing value check, tipe konversi, outlier handling, StandardScaler normalization |
| **EDA** | Histogram, boxplot, KDE, pairplot, scatter plot, correlation heatmap |
| **Model Selection** | Elbow method (WCSS), Silhouette Score, Davies-Bouldin Score, Dendrogram |
| **Model Evaluation** | R-squared, Silhouette Score, Davies-Bouldin Score, per-cluster analysis |
| **Visualisasi** | matplotlib, seaborn, scipy dendrograms, cluster scatter plots |

## Instalasi

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy requests
```
