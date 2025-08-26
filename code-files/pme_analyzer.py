# pme_final_analyzer_all.py

import pandas as pd
import re
from io import StringIO
from collections import defaultdict
import numpy as np

# --- Configuration ---
OBSERVATION_FILES = [
    'regression observations.txt',
    'regression observations pt2.txt',
    'regression observations pt3.txt',
    'regression observations pt4.txt'
]

COHORT_DEFINITIONS = {
    "Group 1: High Row-to-Size (Computational Efficiency)": [
        "Concrete Compressive Strength", "Medical Cost Personal", "Student Grades Prediction",
        "Insurance Forecast", "Electric Vehicle Population Data", "Graduate Admission 2",
        "Superstore Sales", "Walmart Sales", "Credit Card Spending Habits In India",
        "Hotel Booking Demand", "Obesity Level Estimation", "Water Quality Prediction",
        "Weather Prediction For Weather Forecast", "Credit Card Customers",
        "Online Shoppers Purchasing Intention", "Rain in Australia", "Water Potability",
        "Bitcoin Historical Data", "Heart Failure Prediction", "Netflix Userbase",
        "Top 5000 Youtube Channels", "Unemployment in India", "Video Game Sales"
    ],
    "Group 2: Wide Data (Parameter Efficiency)": [
        "Car Details", "Heart Attack Analysis & Prediction", "Real Estate Price Prediction",
        "Employee Future Prediction", "Heart Disease UCI", "Mobile Price Classification",
        "Students Performance in Exams", "Bank Marketing Dataset", "Cervical Cancer Risk Factors",
        "Fertility Diagnosis", "Predict Students Dropout and Academic Success",
        "Predicting Churn for Bank Customers", "QSAR Biodegradation", "Statlog Heart Dataset",
        "Adult Census Income", "Student Alcohol Consumption (Mat)", "Student Alcohol Consumption (Por)",
        "Weather Dataset (Humidity)", "Weather Dataset (Pressure)", "Weather Dataset (Temperature)",
        "Weather Dataset (Direction)", "Weather Dataset (Wind Speed)",
        "Anuran Calls (MFCCs)", "Banknote Authentication", "Car Features and MSRP",
        "Census Income Dataset", "Online Retail", "Turkish Student Evaluation"
    ],
    "Group 3: Messy Data (Data Efficiency)": [
        "Fish Market", "IMDB Movie Data", "PIMA Indian Diabetes",
        "House Prices - Advanced Regression Techniques", "CarDekho Used Car Data",
        "Country Vaccinations by Manufacturer", "Credit Card Fraud Detection",
        "Health Insurance Cross Sell Prediction", "Human Resources Dataset",
        "Air Quality Data In India (City Day)", "Air Quality Data In India (City Hour)",
        "Air Quality Data In India (Station Day)", "Air Quality Data In India (Station Hour)",
        "Loan Prediction Problem Dataset", "Stroke Prediction Dataset", "Telco Customer Churn",
        "Used Car Price Prediction", "Default of Credit Card Clients Dataset", "World University Rankings",
        "Air Quality Dataset", "Credit Card Approval Prediction", "FIFA 22 Players Data",
        "Goodreads Books", "Medical Appointment No Shows", "New York City Airbnb Open Data",
        "US Accidents (2016-2021)"
    ],
    "Group 4: Baseline / Unknown (Control Group)": [
        "Boston Housing", "Car Purchase Amount Prediction", "Ecommerce Customers", "USA Housing",
        "50 Startups", "Advertising Dataset", "Car Price Prediction", "Fish Dataset",
        "Salary Data Data - Simple Linear Regression", "Crop Recommendation Dataset",
        "Customer Churn Prediction", "Diamonds Dataset", "Headbrains Dataset",
        "Sleep Health and Lifestyle Dataset", "Travel Insurance Prediction",
        "World Happiness Report", "Weather Dataset (City Attributes)",
        "Predicting House Prices in Bengaluru", "Body Fat Prediction", "Iris Species",
        "Mall Customer Segmentation", "Sales Prediction for Big Mart Sales"
    ]
}

MODEL_NAMES = {
    'et': 'Extra Trees Regressor', 'gbr': 'Gradient Boosting Regressor', 'rf': 'Random Forest Regressor',
    'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'ridge': 'Ridge Regression', 'br': 'Bayesian Ridge',
    'lr': 'Linear Regression', 'en': 'Elastic Net', 'lasso': 'Lasso Regression',
    'llar': 'Lasso Least Angle Regression', 'omp': 'Orthogonal Matching Pursuit',
    'huber': 'Huber Regressor', 'knn': 'K-Nearest Neighbors', 'dt': 'Decision Tree Regressor',
    'ada': 'AdaBoost Regressor', 'par': 'Passive Aggressive Regressor', 'dummy': 'Dummy Regressor',
    'lar': 'Least Angle Regression'
}

MODEL_GROUPS = {
    'Tree-Based Ensembles': ['et', 'rf', 'gbr', 'lightgbm', 'xgboost', 'ada'],
    'Linear Models': ['lr', 'ridge', 'br', 'lasso', 'llar', 'en', 'omp', 'lar'],
    'Robust/Specialized Models': ['huber', 'par'],
    'Proximity-Based Models': ['knn'],
    'Simple Tree Models': ['dt']
}

def parse_benchmark_files(files):
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

def analyze_performance(all_results, cohorts):
    cohort_analysis = defaultdict(lambda: {
        'top_placements': {1: defaultdict(int), 3: defaultdict(int), 5: defaultdict(int)},
        'bottom_placements': {1: defaultdict(int), 3: defaultdict(int), 5: defaultdict(int)},
        'individual_winners': defaultdict(int),
        'individual_losers': defaultdict(int),
        'win_margins_by_model': defaultdict(list),
        'loss_margins_by_model': defaultdict(list),
        'total_valid_datasets': 0
    })

    for name, df in all_results.items():
        cleaned_name = name.replace("=", "").strip()
        df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
        df = df.dropna(subset=['R2'])
        
        top_performers = df.sort_values(by='R2', ascending=False)
        bottom_performers = df.sort_values(by='R2', ascending=True)

        current_cohort = None
        for cohort, datasets in cohorts.items():
            if cleaned_name in datasets:
                current_cohort = cohort
                break
        if not current_cohort:
            continue
            
        is_valid_run = not top_performers.empty and top_performers.iloc[0]['R2'] > 0
        
        if is_valid_run:
            cohort_analysis[current_cohort]['total_valid_datasets'] += 1
            winner = top_performers.iloc[0]
            winner_model = winner['INDEX']
            cohort_analysis[current_cohort]['individual_winners'][winner_model] += 1
            
            if len(top_performers) > 1:
                second_place = top_performers.iloc[1]
                if second_place['R2'] > 0:
                    margin = ((winner['R2'] - second_place['R2']) / abs(second_place['R2'])) * 100
                    cohort_analysis[current_cohort]['win_margins_by_model'][winner_model].append(margin)

        if not bottom_performers.empty:
            loser = bottom_performers.iloc[0]
            loser_model = loser['INDEX']
            cohort_analysis[current_cohort]['individual_losers'][loser_model] += 1

            if len(bottom_performers) > 1:
                second_worst = bottom_performers.iloc[1]
                if abs(loser['R2']) > 1e-9:
                    margin = ((second_worst['R2'] - loser['R2']) / abs(loser['R2'])) * 100
                    cohort_analysis[current_cohort]['loss_margins_by_model'][loser_model].append(margin)

        for rank in [1, 3, 5]:
            models = top_performers.head(rank)['INDEX'].tolist()
            for model_abbr in models:
                for group_name, model_list in MODEL_GROUPS.items():
                    if model_abbr in model_list:
                        cohort_analysis[current_cohort]['top_placements'][rank][group_name] += 1
        
        for rank in [1, 3, 5]:
            models = bottom_performers.head(rank)['INDEX'].tolist()
            for model_abbr in models:
                for group_name, model_list in MODEL_GROUPS.items():
                    if model_abbr in model_list:
                        cohort_analysis[current_cohort]['bottom_placements'][rank][group_name] += 1
    
    return cohort_analysis

def print_final_report(analysis, cohorts):
    print("--- Final PME Analysis & Model Consistency Report (All Datasets) ---")
    print("="*70)
    
    for cohort_name, data in analysis.items():
        total_valid_datasets = data['total_valid_datasets']
        total_datasets_in_cohort = len(cohorts[cohort_name])
        if total_datasets_in_cohort == 0:
            continue
            
        print(f"\n🔬 COHORT: {cohort_name}\n")
        
        print("  --- ✅ Top Model Group Consistency ---")
        for rank in [1, 3, 5]:
            title = f"Appeared in Top {rank}"
            print(f"  - {title}:")
            placements = data['top_placements'][rank]
            if not placements: print("    No models with positive R-squared."); continue
            sorted_placements = sorted(placements.items(), key=lambda item: item[1], reverse=True)
            total_slots = total_valid_datasets * rank
            for group, count in sorted_placements:
                percentage = (count / total_slots) * 100 if total_slots > 0 else 0
                print(f"    - {group:<25} ({percentage:.1f}%)")
        print("  ---------------------------------------\n")

        print("  --- ❌ Bottom Model Group Underperformance ---")
        for rank in [1, 3, 5]:
            title = f"Appeared in Bottom {rank}"
            print(f"  - {title}:")
            placements = data['bottom_placements'][rank]
            if not placements: print("    No models to report."); continue
            sorted_placements = sorted(placements.items(), key=lambda item: item[1], reverse=True)
            total_slots = total_datasets_in_cohort * rank
            for group, count in sorted_placements:
                percentage = (count / total_slots) * 100 if total_slots > 0 else 0
                print(f"    - {group:<25} ({percentage:.1f}%)")
        print("  -------------------------------------------\n")

        print("  --- 🏆 Top 5 Individual Models (by # of Wins) ---")
        sorted_winners = sorted(data['individual_winners'].items(), key=lambda item: item[1], reverse=True)
        for model_abbr, count in sorted_winners[:5]:
            full_model_name = MODEL_NAMES.get(model_abbr, model_abbr)
            margins = data['win_margins_by_model'].get(model_abbr, [])
            avg_margin_str = f" (Avg. Win Margin: {np.mean(margins):.1f}%)" if margins else ""
            print(f"  - {full_model_name:<35} ({count} wins){avg_margin_str}")
        print("  -------------------------------------------------\n")

        print("  --- 📉 Worst 5 Individual Models (by # of Losses) ---")
        sorted_losers = sorted(data['individual_losers'].items(), key=lambda item: item[1], reverse=True)
        for model_abbr, count in sorted_losers[:5]:
            full_model_name = MODEL_NAMES.get(model_abbr, model_abbr)
            margins = data['loss_margins_by_model'].get(model_abbr, [])
            avg_margin_str = f" (Avg. Loss Margin: {np.mean(margins):.1f}%)" if margins else ""
            print(f"  - {full_model_name:<35} ({count} losses){avg_margin_str}")
        print("  ---------------------------------------------------\n")

    print("\n" + "="*70)
    print("Final analysis complete.")


if __name__ == '__main__':
    results = parse_benchmark_files(OBSERVATION_FILES)
    if results:
        final_stats = analyze_performance(results, COHORT_DEFINITIONS)
        print_final_report(final_stats, COHORT_DEFINITIONS)