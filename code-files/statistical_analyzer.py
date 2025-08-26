# statistical_analyzer.py
#
# A Python script to parse the regression benchmark results and perform
# statistical significance testing (t-tests and ANOVA) to validate
# the key findings of the thesis.
#
# Required libraries: pandas, scipy, statsmodels
# You can install them using pip:
# pip install pandas scipy statsmodels

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import re
import io

# --- 1. CONFIGURATION: MODEL FAMILY MAPPING ---
# This dictionary maps the model abbreviations from the results file
# to their corresponding families, as defined in the thesis.
MODEL_FAMILY_MAP = {
    # Tree-Based Ensembles
    'lightgbm': 'Ensemble',
    'et': 'Ensemble',
    'xgboost': 'Ensemble',
    'gbr': 'Ensemble',
    'rf': 'Ensemble',
    'ada': 'Ensemble',
    # Linear Models
    'lar': 'Linear',
    'ridge': 'Linear',
    'lr': 'Linear',
    'en': 'Linear',
    'lasso': 'Linear',
    'llar': 'Linear',
    'br': 'Linear',
    'omp': 'Linear',
    # Proximity-Based Models
    'knn': 'Proximity',
    # Robust/Specialized Models
    'huber': 'Robust',
    'par': 'Robust',
    # Simple Tree Models
    'dt': 'Simple Tree',
    # Dummy model (will be excluded from analysis)
    'dummy': 'Dummy'
}

# --- 2. PARSING FUNCTION (CORRECTED) ---
# This function reads the custom format of the 'regression observations.txt' file
# using a robust line-by-line approach.

def parse_results_file(filepath):
    """
    Parses the semi-structured regression results text file.

    Args:
        filepath (str): The path to the 'regression observations.txt' file.

    Returns:
        pandas.DataFrame: A DataFrame containing all model results, with
                          columns for Cohort, Dataset, and model metrics.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    all_dataframes = []
    current_cohort = "Unknown"
    current_dataset = "Unknown"
    csv_lines = []
    is_in_csv_block = False

    lines = content.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Use regex to check if the line is a cohort header
        cohort_match = re.search(r'(Group \d:? .*? Cohort|Group \d The .*? Cohort)', line)
        
        if cohort_match:
            if is_in_csv_block and csv_lines:
                csv_string = "\n".join(csv_lines)
                df = pd.read_csv(io.StringIO(csv_string))
                df['Dataset'] = current_dataset
                df['Cohort'] = current_cohort
                all_dataframes.append(df)
                csv_lines = []
                is_in_csv_block = False
            
            current_cohort = cohort_match.group(1).strip()
            continue
        
        # Detect a new dataset header
        if line.startswith('= = = = =') and not ('Group' in line or 'Computational' in line or 'Parameter' in line or 'Data' in line or 'Control' in line or 'Benchmarks' in line):
            if is_in_csv_block and csv_lines:
                csv_string = "\n".join(csv_lines)
                df = pd.read_csv(io.StringIO(csv_string))
                df['Dataset'] = current_dataset
                df['Cohort'] = current_cohort
                all_dataframes.append(df)
            
            current_dataset = line.replace('=', '').strip()
            csv_lines = []
            is_in_csv_block = False
            continue

        if '"INDEX","MODEL"' in line:
            is_in_csv_block = True
        
        if is_in_csv_block:
            csv_lines.append(line)

    if is_in_csv_block and csv_lines:
        csv_string = "\n".join(csv_lines)
        df = pd.read_csv(io.StringIO(csv_string))
        df['Dataset'] = current_dataset
        df['Cohort'] = current_cohort
        all_dataframes.append(df)

    if not all_dataframes:
        print("Warning: No data was parsed. Check the file path and format.")
        return pd.DataFrame()

    full_df = pd.concat(all_dataframes, ignore_index=True)
    full_df.rename(columns={'INDEX': 'Model_ID', 'R2': 'R_Squared'}, inplace=True)
    full_df['Family'] = full_df['Model_ID'].map(MODEL_FAMILY_MAP)
    full_df = full_df[full_df['Family'] != 'Dummy']
    
    return full_df


# --- 3. STATISTICAL ANALYSIS FUNCTIONS ---

def perform_t_test_analysis(df):
    """
    Performs a paired t-test to compare Ensembles and Linear models
    in the 'High Row-to-Size' cohort.
    """
    print("="*60)
    print("🔬 T-TEST ANALYSIS: Law of Ensemble Dominance")
    print("="*60)
    print("Hypothesis: In the 'High Row-to-Size' cohort, the performance of")
    print("Tree-Based Ensembles is significantly better than Linear Models.\n")

    # Filter for the specific cohort
    # *** FIX: Corrected cohort name to match the file exactly ***
    cohort_df = df[df['Cohort'] == 'Group 1 High Row-to-Size Cohort'].copy()

    best_scores = cohort_df.loc[cohort_df.groupby(['Dataset', 'Family'])['R_Squared'].idxmax()]
    pivot_df = best_scores.pivot(index='Dataset', columns='Family', values='R_Squared')
    pivot_df.dropna(subset=['Ensemble', 'Linear'], inplace=True)

    if pivot_df.empty or 'Ensemble' not in pivot_df.columns or 'Linear' not in pivot_df.columns:
        print("Could not perform t-test: Missing 'Ensemble' or 'Linear' family data in the cohort.")
        return

    ensemble_scores = pivot_df['Ensemble']
    linear_scores = pivot_df['Linear']

    t_stat, p_value = stats.ttest_rel(ensemble_scores, linear_scores)

    print(f"Comparing Ensembles (Mean R² = {ensemble_scores.mean():.4f}) vs. Linear Models (Mean R² = {linear_scores.mean():.4f})")
    print(f"Paired t-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}\n")

    alpha = 0.05
    if p_value < alpha and t_stat > 0:
        print(f"Conclusion: The p-value ({p_value:.4f}) is less than {alpha}.")
        print("✅ We REJECT the null hypothesis. The observed performance difference is statistically significant.")
        print("This strongly supports the 'Law of Ensemble Dominance' for this cohort.\n")
    else:
        print(f"Conclusion: The p-value ({p_value:.4f}) is not less than {alpha} or the t-statistic is not positive.")
        print("❌ We FAIL to reject the null hypothesis. The performance difference is not statistically significant.\n")


def perform_anova_analysis(df):
    """
    Performs a one-way ANOVA and Tukey's HSD post-hoc test to compare
    the top 3 model families in the 'Wide Data' cohort.
    """
    print("="*60)
    print("🔬 ANOVA ANALYSIS: Law of Anomaly Supremacy (Wide Data)")
    print("="*60)
    print("Hypothesis: In the 'Wide Data' cohort, there is a significant")
    print("performance difference among Ensembles, Linear, and Proximity models.\n")

    cohort_df = df[df['Cohort'] == 'Group 2 The Wide Data Cohort'].copy()
    best_scores = cohort_df.loc[cohort_df.groupby(['Dataset', 'Family'])['R_Squared'].idxmax()]
    
    families_to_compare = ['Ensemble', 'Linear', 'Proximity']
    anova_df = best_scores[best_scores['Family'].isin(families_to_compare)]
    
    pivot_df = anova_df.pivot(index='Dataset', columns='Family', values='R_Squared')
    pivot_df.dropna(subset=families_to_compare, inplace=True)

    if len(pivot_df) < 2:
        print("Could not perform ANOVA: Not enough complete data for all three families.")
        return

    ensemble_scores = pivot_df['Ensemble']
    linear_scores = pivot_df['Linear']
    proximity_scores = pivot_df['Proximity']

    f_stat, p_value = stats.f_oneway(ensemble_scores, linear_scores, proximity_scores)

    print(f"Comparing Ensembles (Mean R² = {ensemble_scores.mean():.4f}), "
          f"Linear (Mean R² = {linear_scores.mean():.4f}), "
          f"and Proximity (Mean R² = {proximity_scores.mean():.4f})")
    print(f"One-way ANOVA results: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}\n")

    alpha = 0.05
    if p_value < alpha:
        print(f"Conclusion: The p-value ({p_value:.4f}) is less than {alpha}.")
        print("✅ A statistically significant difference exists somewhere among the groups.")
        print("This supports the 'Law of Anomaly Supremacy' by showing the playing field has leveled.\n")
        
        print("--- Running Tukey's HSD post-hoc test to find which pairs are different ---\n")
        
        tukey_data = pd.melt(pivot_df.reset_index(), id_vars=['Dataset'], value_vars=families_to_compare)
        tukey_result = pairwise_tukeyhsd(endog=tukey_data['value'], groups=tukey_data['Family'], alpha=alpha)
        
        print(tukey_result)
        print("\nInterpretation of Tukey's HSD:")
        print("The 'reject' column indicates if the difference between a pair is statistically significant (True) or not (False).")

    else:
        print(f"Conclusion: The p-value ({p_value:.4f}) is greater than {alpha}.")
        print("❌ We FAIL to reject the null hypothesis. There is no statistically significant difference among the groups.\n")


# --- 4. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    results_filepath = 'regression observations.txt'

    try:
        full_results_df = parse_results_file(results_filepath)

        if not full_results_df.empty:
            perform_t_test_analysis(full_results_df)
            perform_anova_analysis(full_results_df)
            
    except FileNotFoundError:
        print(f"ERROR: The file '{results_filepath}' was not found.")
        print("Please make sure the results file is in the same directory as this script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

