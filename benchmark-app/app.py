# app.py

import os
import pandas as pd
import numpy as np
import gc
import re
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler
from pycaret.regression import setup as setup_r, compare_models as compare_models_r, pull as pull_r
from pycaret.classification import setup as setup_c, compare_models as compare_models_c, pull as pull_c

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dataframe(filepath):
    """Reads a CSV or XLSX file into a pandas DataFrame and sanitizes column names."""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            return None

        # --- NEW: Sanitize column names ---
        original_columns = df.columns
        new_columns = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in original_columns]
        df.columns = new_columns
        
        return df
    except Exception as e:
        flash(f"Error reading file: {e}", "error")
        return None

def optimize_df(df):
    """Downcasts numerical columns and converts low-cardinality objects to categories."""
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    return df

def get_data_quality(df):
    """Calculates missing values and potential outliers."""
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    quality_report = {}
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = missing_percent[missing_percent > 0].to_dict()
    quality_report['missing_values'] = {k: round(v, 2) for k, v in missing_data.items()}
    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if outlier_count > 0:
            outliers[col] = outlier_count
    quality_report['outliers'] = outliers
    return quality_report

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles file upload and redirects to the cleaning page."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Re-save with sanitized headers
            df = get_dataframe(filepath)
            if df is not None:
                df.to_csv(filepath, index=False)
            
            return redirect(url_for('clean_data', filename=filename))
    return render_template('index.html')

@app.route('/clean/<filename>')
def clean_data(filename):
    """Displays the data quality report and cleaning tools."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = get_dataframe(filepath)
    if df is None: return redirect(url_for('index'))

    df = optimize_df(df)
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    categorical_columns = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() < len(df):
            categorical_columns.append(col)

    data_quality = get_data_quality(df)

    del df; gc.collect()

    return render_template('clean.html', filename=filename, data_quality=data_quality, 
                           numerical_columns=numerical_columns, categorical_columns=categorical_columns)

@app.route('/impute/<filename>', methods=['POST'])
def impute(filename):
    """Handles imputation of missing values."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    column = request.form.get('column')
    strategy = request.form.get('strategy')
    df = get_dataframe(filepath)
    if df is None: return redirect(url_for('index'))

    if strategy in ['mean', 'median', 'most_frequent']:
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=strategy)
        df[column] = imputer.fit_transform(df[[column]]).ravel()
    elif strategy in ['ffill', 'bfill']:
        df[column].fillna(method=strategy, inplace=True)
    elif strategy == 'remove_rows':
        df.dropna(subset=[column], inplace=True)

    df.to_csv(filepath, index=False)
    flash(f"Successfully applied '{strategy}' to column '{column}'.", "success")
    return redirect(url_for('clean_data', filename=filename))

@app.route('/outliers/<filename>', methods=['POST'])
def handle_outliers(filename):
    """Handles clipping or removing outliers."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    column = request.form.get('column')
    action = request.form.get('action')
    iqr_multiplier = float(request.form.get('iqr_multiplier', 1.5))
    df = get_dataframe(filepath)
    if df is None: return redirect(url_for('index'))

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    if action == 'clip':
        df[column] = np.clip(df[column], lower_bound, upper_bound)
        flash(f"Clipped outliers in '{column}'.", "success")
    elif action == 'remove':
        original_rows = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        flash(f"Removed {original_rows - len(df)} outlier rows.", "success")

    df.to_csv(filepath, index=False)
    return redirect(url_for('clean_data', filename=filename))

@app.route('/benchmark/<filename>', methods=['POST'])
def benchmark(filename):
    """Runs regression models and displays a comparison table."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = get_dataframe(filepath)
    if df is None: return redirect(url_for('index'))

    target_column = request.form.get('target_column')
    if not target_column:
        flash("You must select a target column to benchmark.", "error")
        return redirect(url_for('clean_data', filename=filename))

    s = setup_r(data=df, target=target_column, session_id=123, verbose=False)
    compare_models_r()
    benchmark_results_df = pull_r().reset_index()

    if benchmark_results_df.empty:
        flash("Could not train any models for this regression task. Please check your data.", "error")
        return redirect(url_for('clean_data', filename=filename))

    all_metric_priorities = {'R2': 3, 'RMSE': -2, 'MAE': -1.5, 'MAPE': -1, 'RMSLE': -1}
    available_metrics = {k: v for k, v in all_metric_priorities.items() if k in benchmark_results_df.columns}
    
    if available_metrics:
        scaler = MinMaxScaler()
        metrics_df = benchmark_results_df[list(available_metrics.keys())]
        metrics_normalized = pd.DataFrame(scaler.fit_transform(metrics_df), columns=metrics_df.columns)
        priority_score = (metrics_normalized * pd.Series(available_metrics)).sum(axis=1)
        best_model_index = priority_score.idxmax()
        worst_model_index = priority_score.idxmin()
    else:
        numeric_metric_cols = benchmark_results_df.select_dtypes(include=np.number).columns
        if not numeric_metric_cols.empty:
            fallback_metric = numeric_metric_cols[0]
            best_model_index = benchmark_results_df[fallback_metric].idxmin()
            worst_model_index = benchmark_results_df[fallback_metric].idxmax()
        else:
            best_model_index, worst_model_index = 0, len(benchmark_results_df) - 1

    numeric_cols = benchmark_results_df.select_dtypes(include=np.number).columns
    metrics_extrema = {}
    for col in numeric_cols:
        if col in ['R2']:
            metrics_extrema[col] = {'best': benchmark_results_df[col].idxmax(), 'worst': benchmark_results_df[col].idxmin()}
        else:
            metrics_extrema[col] = {'best': benchmark_results_df[col].idxmin(), 'worst': benchmark_results_df[col].idxmax()}

    benchmark_results = benchmark_results_df.to_dict('records')
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].nunique() < len(df)]

    return render_template('benchmark.html', results=benchmark_results, problem_type="Regression", 
                           filename=filename, numerical_columns=numerical_columns, categorical_columns=categorical_columns,
                           best_model_index=best_model_index, worst_model_index=worst_model_index, metrics_extrema=metrics_extrema)

@app.route('/benchmark_classification/<filename>', methods=['POST'])
def benchmark_classification(filename):
    """Runs classification models and displays a comparison table."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = get_dataframe(filepath)
    if df is None: return redirect(url_for('index'))

    target_column = request.form.get('target_column')
    if not target_column:
        flash("You must select a target column to benchmark.", "error")
        return redirect(url_for('clean_data', filename=filename))

    s = setup_c(data=df, target=target_column, session_id=123, verbose=False)
    compare_models_c()
    benchmark_results_df = pull_c().reset_index()

    if benchmark_results_df.empty:
        flash("Could not train any models for this classification task. Please check your data.", "error")
        return redirect(url_for('clean_data', filename=filename))

    all_metric_priorities = {'Accuracy': 3, 'AUC': 2, 'F1': 1.5, 'Prec.': 1, 'Recall': 1, 'Kappa': 1}
    available_metrics = {k: v for k, v in all_metric_priorities.items() if k in benchmark_results_df.columns}
    
    if available_metrics:
        scaler = MinMaxScaler()
        metrics_df = benchmark_results_df[list(available_metrics.keys())]
        metrics_normalized = pd.DataFrame(scaler.fit_transform(metrics_df), columns=metrics_df.columns)
        priority_score = (metrics_normalized * pd.Series(available_metrics)).sum(axis=1)
        best_model_index = priority_score.idxmax()
        worst_model_index = priority_score.idxmin()
    else:
        numeric_metric_cols = benchmark_results_df.select_dtypes(include=np.number).columns
        if not numeric_metric_cols.empty:
            fallback_metric = numeric_metric_cols[0]
            best_model_index = benchmark_results_df[fallback_metric].idxmax()
            worst_model_index = benchmark_results_df[fallback_metric].idxmin()
        else:
            best_model_index, worst_model_index = 0, len(benchmark_results_df) - 1

    numeric_cols = benchmark_results_df.select_dtypes(include=np.number).columns
    metrics_to_minimize = [] 
    metrics_extrema = {}
    for col in numeric_cols:
        if col in metrics_to_minimize:
             metrics_extrema[col] = {'best': benchmark_results_df[col].idxmin(), 'worst': benchmark_results_df[col].idxmax()}
        else:
             metrics_extrema[col] = {'best': benchmark_results_df[col].idxmax(), 'worst': benchmark_results_df[col].idxmin()}
    
    benchmark_results = benchmark_results_df.to_dict('records')
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].nunique() < len(df)]

    return render_template('benchmark.html', results=benchmark_results, problem_type="Classification", 
                           filename=filename, numerical_columns=numerical_columns, categorical_columns=categorical_columns,
                           best_model_index=best_model_index, worst_model_index=worst_model_index, metrics_extrema=metrics_extrema)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)