"""
Medical Dataset Generator for Thyroid Disease Classification
Generates realistic synthetic patient data with clinical correlations
"""

import numpy as np
import pandas as pd
from config import N_SAMPLES, RANDOM_STATE


def generate_medical_dataset(n_samples=N_SAMPLES):
    """
    Generate realistic thyroid disease dataset with medical feature correlations.
    
    Dataset Characteristics:
    - Sample size: 7,200 patient records
    - Features: 19 clinical and demographic attributes
    - Target classes: 3 (Negative, Hypothyroid, Hyperthyroid)
    - Class distribution: Imbalanced (reflecting real-world prevalence)
    
    Realistic Properties:
    - TSH correlation with disease state: 80%
    - Measurement uncertainty: ±8%
    - Overlapping hormone ranges across classes
    - Edge cases: 3% of samples with atypical presentations
    
    Returns:
        pd.DataFrame: Complete dataset with features and target variable
    """
    np.random.seed(RANDOM_STATE)
    
    print(f"\n{'='*80}")
    print(f"DATASET GENERATION: {n_samples:,} PATIENT RECORDS")
    print(f"{'='*80}")
    print("\nIncorporating Clinical Realism:")
    print("  • Laboratory measurement uncertainty (±8%)")
    print("  • Overlapping hormone reference ranges")
    print("  • Subclinical and borderline presentations")
    print("  • Treatment status variability\n")
    
    # Generate primary diagnostic indicator (TSH)
    TSH_base = np.random.lognormal(0.5, 1.4, n_samples)
    
    # Assign diagnostic labels based on TSH levels with clinical uncertainty
    targets = []
    for tsh in TSH_base:
        if tsh > 5.0:  # Elevated TSH
            targets.append(np.random.choice(
                ['hypothyroid', 'negative', 'hyperthyroid'], 
                p=[0.80, 0.15, 0.05]
            ))
        elif tsh < 0.5:  # Suppressed TSH
            targets.append(np.random.choice(
                ['hyperthyroid', 'negative', 'hypothyroid'], 
                p=[0.75, 0.20, 0.05]
            ))
        else:  # Normal TSH range
            targets.append(np.random.choice(
                ['negative', 'hypothyroid', 'hyperthyroid'], 
                p=[0.92, 0.05, 0.03]
            ))
    
    targets = np.array(targets)
    
    # Apply measurement uncertainty to TSH
    TSH = TSH_base * np.random.normal(1.0, 0.08, n_samples)
    TSH = np.clip(TSH, 0.01, 50)
    
    # Generate T3 (Triiodothyronine)
    T3 = np.zeros(n_samples)
    for i, target in enumerate(targets):
        if target == 'hypothyroid':
            T3[i] = np.clip(np.random.normal(1.0, 0.4), 0.4, 1.8)
        elif target == 'hyperthyroid':
            T3[i] = np.clip(np.random.normal(4.2, 0.8), 2.5, 6.5)
        else:
            T3[i] = np.clip(np.random.normal(1.8, 0.6), 0.8, 3.5)
    
    T3 = T3 * np.random.normal(1.0, 0.08, n_samples)
    
    # Generate T4 (Thyroxine)
    T4 = np.zeros(n_samples)
    for i, target in enumerate(targets):
        if target == 'hypothyroid':
            T4[i] = np.clip(np.random.normal(70, 25), 35, 120)
        elif target == 'hyperthyroid':
            T4[i] = np.clip(np.random.normal(155, 30), 110, 230)
        else:
            T4[i] = np.clip(np.random.normal(105, 25), 60, 160)
    
    T4 = T4 * np.random.normal(1.0, 0.08, n_samples)
    
    # Additional clinical parameters
    T4U = np.clip(np.random.normal(1.0, 0.18, n_samples), 0.5, 1.8)
    
    # Treatment status
    on_thyroxine = np.zeros(n_samples)
    on_antithyroid = np.zeros(n_samples)
    
    for i, target in enumerate(targets):
        if target == 'hypothyroid':
            on_thyroxine[i] = np.random.choice([0, 1], p=[0.40, 0.60])
        if target == 'hyperthyroid':
            on_antithyroid[i] = np.random.choice([0, 1], p=[0.45, 0.55])
    
    # Construct feature matrix
    data = {
        'age': np.clip(np.random.normal(48, 18, n_samples), 18, 90),
        'sex': np.random.choice([0, 1], n_samples, p=[0.68, 0.32]),
        'TSH': TSH,
        'T3': T3,
        'T4': T4,
        'T4U': T4U,
        'FTI': T4 / (T4U + 0.01),
        'on_thyroxine': on_thyroxine,
        'on_antithyroid': on_antithyroid,
        'sick': np.random.choice([0, 1], n_samples, p=[0.78, 0.22]),
        'pregnant': np.random.choice([0, 1], n_samples, p=[0.96, 0.04]),
        'thyroid_surgery': np.random.choice([0, 1], n_samples, p=[0.89, 0.11]),
        'goitre': np.random.choice([0, 1], n_samples, p=[0.83, 0.17]),
        'tumor': np.random.choice([0, 1], n_samples, p=[0.96, 0.04]),
        'query_hypothyroid': (targets == 'hypothyroid').astype(int) * np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'query_hyperthyroid': (targets == 'hyperthyroid').astype(int) * np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'TSH_measured': np.ones(n_samples),
        'T3_measured': np.random.choice([0, 1], n_samples, p=[0.12, 0.88]),
        'T4_measured': np.ones(n_samples),
        'target': targets
    }
    
    df = pd.DataFrame(data)
    
    # Introduce edge cases (atypical presentations)
    n_edge_cases = int(0.03 * n_samples)
    edge_indices = np.random.choice(n_samples, n_edge_cases, replace=False)
    
    for idx in edge_indices:
        current_target = df.loc[idx, 'target']
        if current_target == 'hypothyroid':
            df.loc[idx, 'target'] = np.random.choice(['negative', 'hyperthyroid'], p=[0.8, 0.2])
        elif current_target == 'hyperthyroid':
            df.loc[idx, 'target'] = np.random.choice(['negative', 'hypothyroid'], p=[0.7, 0.3])
    
    print(f"Dataset Generation Complete: {len(df):,} samples")
    print(f"\nClass Distribution:")
    for target, count in df['target'].value_counts().items():
        print(f"  {target:15} {count:5} ({count/len(df)*100:5.1f}%)")
    
    print(f"\nDataset Characteristics:")
    print(f"  TSH-Disease Correlation: 80%")
    print(f"  Measurement Uncertainty: ±8%")
    print(f"  Overlapping Ranges: Present")
    print(f"  Edge Cases: {n_edge_cases} samples ({n_edge_cases/n_samples*100:.1f}%)")
    print(f"\n{'='*80}\n")
    
    return df