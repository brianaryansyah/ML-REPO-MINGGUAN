import sys
import pandas as pd
import os
import requests
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

class MLAcquisitionEngine:
    """
    Engine untuk akuisisi data sebagai 'raw material' solusi AI.
    """
    
    def __init__(self, raw_data_path="../data/raw/"):
        self.raw_path = raw_data_path
        if not os.path.exists(self.raw_path):
            os.makedirs(self.raw_path)

    def load_manual_csv(self, filename):
        print(f"Memuat dataset manual dari repositori lokal...")
        try:
            return pd.read_csv(filename)
        except Exception as e:
            print(f"Error: {e}")
            return None

    def fetch_from_api(self, endpoint, api_key=None):
        print(f"Menarik data via API (Metode Akses Terprogram)...")
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        response = requests.get(endpoint, headers=headers)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        return None

class DataProfiler:
    """
    Professional data profiling utility for ML pipeline.
    Provides comprehensive dataset analysis including schema validation,
    type inference, quality metrics, and statistical summaries.
    """
    
    SEPARATOR = "=" * 60
    SUB_SEPARATOR = "-" * 60
    
    @staticmethod
    def _infer_measurement_scale(series):
        """Determine the measurement scale of a data column."""
        dtype_kind = series.dtype.kind
        
        if dtype_kind in ('O', 'U') or str(series.dtype) == 'str':
            return 'Categorical', 'Nominal/Ordinal'
        elif dtype_kind in ('i', 'u'):
            return 'Numerical', 'Ratio'
        elif dtype_kind == 'f':
            return 'Numerical', 'Interval/Ratio'
        elif dtype_kind == 'b':
            return 'Binary', 'Nominal'
        elif dtype_kind == 'M':
            return 'Temporal', 'Interval'
        else:
            return 'Unknown', 'Unknown'
    
    @classmethod
    def generate_report(cls, df, dataset_name="Untitled Dataset"):
        """Generate a comprehensive data profiling report."""
        
        print(f"\n{cls.SEPARATOR}")
        print(f"  DATA PROFILING REPORT")
        print(f"  Dataset: {dataset_name}")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{cls.SEPARATOR}")
        
        # Section 1: Dataset Overview
        print(f"\n[1] DATASET OVERVIEW")
        print(f"    Records (rows)   : {len(df):,}")
        print(f"    Features (cols)  : {len(df.columns)}")
        print(f"    Memory footprint : {df.memory_usage(deep=True).sum():,} bytes")
        print(f"    Duplicates       : {df.duplicated().sum()} rows")
        
        missing_total = df.isna().sum().sum()
        missing_pct = (missing_total / (len(df) * len(df.columns))) * 100
        print(f"    Missing values   : {missing_total} ({missing_pct:.1f}%)")
        
        # Section 2: Schema & Type Classification
        print(f"\n[2] SCHEMA & FEATURE CLASSIFICATION")
        print(f"    {'Feature':<15} {'Type':<12} {'Scale':<16} {'Unique':<8} {'Missing':<8} {'Sample'}")
        print(f"    {'-'*13} {'-'*10} {'-'*14} {'-'*6} {'-'*6} {'-'*12}")
        
        for col in df.columns:
            feature_type, scale = cls._infer_measurement_scale(df[col])
            unique_count = df[col].nunique()
            missing_count = df[col].isna().sum()
            
            sample_val = df[col].dropna().iloc[0] if not df[col].empty else 'N/A'
            if isinstance(sample_val, str) and len(sample_val) > 10:
                sample_val = sample_val[:10] + "..."
            
            print(f"    {col:<15} {feature_type:<12} {scale:<16} {unique_count:<8} {missing_count:<8} {sample_val}")
        
        # Section 3: Numerical Features Summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print(f"\n[3] NUMERICAL FEATURES SUMMARY")
            print(f"    {'Feature':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Median':>10} {'Max':>10}")
            print(f"    {'-'*13} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            
            for col in numeric_cols:
                stats = df[col].describe()
                print(f"    {col:<15} {stats['mean']:>10.2f} {stats['std']:>10.2f} {stats['min']:>10.2f} {stats['50%']:>10.2f} {stats['max']:>10.2f}")
        
        # Section 4: Categorical Features Summary
        categorical_cols = [col for col in df.columns if df[col].dtype.kind in ('O', 'U') or str(df[col].dtype) == 'str']
        if categorical_cols:
            print(f"\n[4] CATEGORICAL FEATURES DISTRIBUTION")
            for col in categorical_cols:
                print(f"\n    Feature: '{col}'")
                value_counts = df[col].value_counts()
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    bar = "█" * int(pct / 5)
                    print(f"      {str(val):<15} : {count:>3} ({pct:5.1f}%) {bar}")
        
        # Section 5: Data Quality Assessment
        print(f"\n[5] DATA QUALITY ASSESSMENT")
        quality_score = 100 - (missing_pct * 0.5) - (df.duplicated().sum() / len(df) * 0.5 * 100)
        quality_score = max(0, min(100, quality_score))
        
        status = "EXCELLENT" if quality_score >= 90 else "GOOD" if quality_score >= 70 else "NEEDS ATTENTION"
        print(f"    Quality Score : {quality_score:.1f}/100 [{status}]")
        print(f"{cls.SEPARATOR}\n")

if __name__ == "__main__":
    data = {
        'Pos': [1, 2, 3, 4, 5],
        'Tim': ['Real Madrid', 'FC Barcelona', 'Atletico Madrid', 'Girona FC', 'Athletic Bilbao'],
        'Main': [38, 38, 38, 38, 38],
        'Menang': [28, 26, 24, 22, 20],
        'Seri': [6, 8, 7, 9, 10],
        'Kalah': [4, 4, 7, 7, 8],
        'GM': [85, 78, 65, 70, 55],
        'GK': [30, 35, 32, 38, 40],
        'SG': [55, 43, 33, 32, 15],
        'Poin': [90, 86, 79, 75, 70]
    }
    
    df_sample = pd.DataFrame(data)
    DataProfiler.generate_report(df_sample, dataset_name="Klasemen La Liga Spanyol 2026")
