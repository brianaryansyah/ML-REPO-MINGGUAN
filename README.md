# ML Data Acquisition & Profiling Engine

Pipeline akuisisi data dan data profiling untuk proyek Machine Learning.

## Komponen Utama

### `MLAcquisitionEngine`
Engine untuk akuisisi data dari berbagai sumber:
- **Manual CSV** - Muat dataset dari file lokal
- **API** - Tarik data secara terprogram via REST API
- **Database SQL** - Query ke database relasional (SQLAlchemy)

### `DataProfiler`
Utility profiling data yang menghasilkan laporan komprehensif dengan 5 section:

1. **Dataset Overview** - Records, features, memory footprint, duplikat, missing values
2. **Schema & Feature Classification** - Tipe data dan skala pengukuran (Nominal/Ordinal/Ratio/Interval)
3. **Numerical Features Summary** - Statistik deskriptif (Mean, Std, Min, Median, Max)
4. **Categorical Features Distribution** - Distribusi kategori dengan visual bar chart
5. **Data Quality Assessment** - Skor kualitas dataset 0-100 dengan rating (EXCELLENT/GOOD/NEEDS ATTENTION)

## Instalasi

```bash
pip install pandas requests numpy
```

## Penggunaan

### Jalankan langsung
```bash
python acquisition_engine.py
```

### Gunakan sebagai modul
```python
from acquisition_engine import MLAcquisitionEngine, DataProfiler

# Load CSV
engine = MLAcquisitionEngine()
df = engine.load_manual_csv("data.csv")

# Generate report
DataProfiler.generate_report(df, dataset_name="Dataset Saya")
```

## Dataset Contoh

Script menggunakan data **Klasemen La Liga Spanyol 2026** dengan fitur:
`Pos`, `Tim`, `Main`, `Menang`, `Seri`, `Kalah`, `GM`, `GK`, `SG`, `Poin`
