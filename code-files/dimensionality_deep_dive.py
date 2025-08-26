# dimensionality_deep_dive_v3.py (with outlier filtering)

import pandas as pd
import re
from io import StringIO
from collections import defaultdict
import numpy as np
import sys

# --- Configuration ---
OBSERVATION_FILES = [
    'regression observations.txt',
    'regression observations pt2.txt',
    'regression observations pt3.txt',
    'regression observations pt4.txt'
]
DIMENSIONS_FILE = 'dataset_dimensions.csv'

MODEL_GROUPS = {
    'Tree-Based Ensembles': ['et', 'rf', 'gbr', 'lightgbm', 'xgboost', 'ada'],
    'Linear Models': ['lr', 'ridge', 'br', 'lasso', 'llar', 'en', 'omp', 'lar']
}

def parse_benchmark_files(files):
    # (This function remains the same)
    all_results = {}
    current_dataset_name = None
    current_dataset_content = ""
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.search(r'= = = = = (.*) = = = = =', line)
                    if match:
                        dataset_name = match.group(1).strip()
                        if current_dataset_name and current_dataset_content:
                            try:
                                df = pd.read_csv(StringIO(current_dataset_content))
                                df.columns = [col.strip() for col in df.columns]
                                all_results[current_dataset_name] = df
                            except Exception as e:
                                print(f"Warning: Could not parse data for '{current_dataset_name}'. Error: {e}")
                        current_dataset_name = dataset_name
                        current_dataset_content = ""
                    elif '"' in line:
                        current_dataset_content += line
        except FileNotFoundError:
            print(f"Error: File not found -> {file_path}.")
            return None
    if current_dataset_name and current_dataset_content:
        try:
            df = pd.read_csv(StringIO(current_dataset_content))
            df.columns = [col.strip() for col in df.columns]
            all_results[current_dataset_name] = df
        except Exception as e:
            print(f"Warning: Could not parse data for '{current_dataset_name}'. Error: {e}")
    return all_results

def calculate_group_performance(df):
    group_scores = defaultdict(list)
    for index, row in df.iterrows():
        model_abbr = row['INDEX']
        r2 = pd.to_numeric(row['R2'], errors='coerce')
        
        # --- NEW: Filter out catastrophic failures ---
        # Only include scores that are not NaN and are greater than -1 (a reasonable failure threshold)
        if pd.notna(r2) and r2 > -1:
            for group, models in MODEL_GROUPS.items():
                if model_abbr in models:
                    group_scores[group].append(r2)
                    
    # Return the mean R2 for each group that has valid scores
    return {group: np.mean(scores) for group, scores in group_scores.items() if scores}

def run_deep_dive_analysis(all_results, dimensions_df):
    analysis_data = []
    
    for name, df in all_results.items():
        cleaned_name = name.replace("=", "").strip()
        dim_row = dimensions_df[dimensions_df['Dataset'] == cleaned_name]
        if dim_row.empty: continue
        
        samples = dim_row.iloc[0]['Samples']
        features = dim_row.iloc[0]['Features']
        
        if samples > 0 and features > 0:
            dim_score = np.log(features / samples)
        else: continue
            
        group_perf = calculate_group_performance(df)
        
        if 'Tree-Based Ensembles' in group_perf and 'Linear Models' in group_perf:
            delta_ensemble_vs_linear = group_perf['Tree-Based Ensembles'] - group_perf['Linear Models']
            analysis_data.append({
                'Dataset': cleaned_name,
                'Dimensionality_Score': dim_score,
                'Delta_Ensemble_vs_Linear': delta_ensemble_vs_linear
            })

    analysis_df = pd.DataFrame(analysis_data)
    
    # --- Perform Binned Analysis ---
    print("\n--- Dimensionality Deep Dive: Binned Analysis (Corrected) ---")
    print("="*70)
    
    try:
        analysis_df['Dimensionality_Bin'] = pd.qcut(
            analysis_df['Dimensionality_Score'], 
            q=3, 
            labels=["Low (Tall Datasets)", "Medium (Balanced Datasets)", "High (Wide Datasets)"]
        )
    except ValueError:
        print("Warning: Not enough unique dimensionality scores to create 3 bins. Analysis may be limited.")
        return

    binned_summary = analysis_df.groupby('Dimensionality_Bin')['Delta_Ensemble_vs_Linear'].mean().reset_index()
    
    print("This table shows the average performance gap (R² Ensemble - R² Linear) for each dimensionality type.\n")
    print("A positive value means Ensembles performed better.")
    print("A negative value means Linear Models performed better.\n")
    
    print(binned_summary.to_string(index=False))
    
    print("\n--- Interpretation ---")
    
    low_bin_perf = binned_summary[binned_summary['Dimensionality_Bin'] == "Low (Tall Datasets)"]['Delta_Ensemble_vs_Linear'].iloc[0]
    high_bin_perf = binned_summary[binned_summary['Dimensionality_Bin'] == "High (Wide Datasets)"]['Delta_Ensemble_vs_Linear'].iloc[0]

    if high_bin_perf > low_bin_perf + 0.05: # Adding a small threshold for a meaningful trend
        print("✅ Finding CONFIRMED: As dimensionality increases, the performance advantage of Tree-Based Ensembles over Linear Models tends to grow.")
    elif low_bin_perf > high_bin_perf + 0.05:
        print("💡 New Finding: As dimensionality increases, the performance gap narrows, suggesting Linear Models become more competitive on wide datasets.")
    else:
        print("- Finding: There is no clear, monotonic trend. The performance gap between Ensembles and Linear Models does not consistently increase or decrease with dimensionality.")
        
    print("="*70)

if __name__ == '__main__':
    try:
        dims_df = pd.read_csv(DIMENSIONS_FILE)
        required_columns = {'Dataset', 'Samples', 'Features'}
        if not required_columns.issubset(dims_df.columns):
            print(f"ERROR: Your '{DIMENSIONS_FILE}' is missing required columns!")
            print(f"Please ensure the header contains EXACTLY these names: Dataset,Samples,Features")
            sys.exit(1)
            
        results = parse_benchmark_files(OBSERVATION_FILES)
        if results and not dims_df.empty:
            run_deep_dive_analysis(results, dims_df)
    except FileNotFoundError:
        print(f"Error: '{DIMENSIONS_FILE}' not found. Please create this file with your dataset dimensions.")