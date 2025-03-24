import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

# Constants
BACKTEST_DIR = "backtests"
BACKTEST_LOGS_DIR = os.path.join(BACKTEST_DIR, "logs")
BACKTEST_GRAPHS_DIR = os.path.join(BACKTEST_DIR, "graphs")
SUMMARY_DIR = os.path.join(BACKTEST_DIR, "summary")

# Create summary directory
os.makedirs(SUMMARY_DIR, exist_ok=True)

def load_backtest_results():
    """Load all backtest results from log files"""
    log_files = glob.glob(os.path.join(BACKTEST_LOGS_DIR, "*.json"))
    
    if not log_files:
        print("No backtest logs found. Run backtest.py first.")
        return []
    
    results = []
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    print(f"Loaded {len(results)} backtest results")
    return results

def create_summary_dataframe(results):
    """Convert backtest results to a pandas DataFrame for analysis"""
    # Extract key metrics from results
    summary_data = []
    
    for result in results:
        if not result.get('metrics'):
            continue
            
        row = {
            'backtest_id': result['backtest_id'],
            'symbol': result['symbol'],
            'model_type': result['model_type'],
            'train_start': result['date_range']['train_start'],
            'train_end': result['date_range']['train_end'],
            'test_start': result['date_range']['test_start'],
            'test_end': result['date_range']['test_end'],
            'rmse': result['metrics'].get('rmse', np.nan),
            'mae': result['metrics'].get('mae', np.nan),
            'directional_accuracy': result['metrics'].get('directional_accuracy', np.nan),
            'next_day_error': result['metrics'].get('next_day_error', np.nan),
            'next_day_error_pct': result['metrics'].get('next_day_error_pct', np.nan),
            'prediction_days': result['prediction_days'],
            'feature_days': result['feature_days']
        }
        summary_data.append(row)
    
    if not summary_data:
        print("No valid results with metrics found.")
        return None
        
    df = pd.DataFrame(summary_data)
    
    # Convert date strings to datetime
    for date_col in ['train_start', 'train_end', 'test_start', 'test_end']:
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Add additional metrics
    df['test_period_days'] = (df['test_end'] - df['test_start']).dt.days
    
    return df

def plot_model_comparison(df):
    """Plot comparison of different model types"""
    if df is None or df.empty:
        return
        
    plt.figure(figsize=(16, 10))
    
    # Plot 1: RMSE by model type
    plt.subplot(2, 2, 1)
    model_rmse = df.groupby('model_type')['rmse'].mean().sort_values()
    model_rmse.plot(kind='bar', color='skyblue')
    plt.title('Average RMSE by Model Type')
    plt.ylabel('RMSE (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Directional accuracy by model type
    plt.subplot(2, 2, 2)
    model_dir = df.groupby('model_type')['directional_accuracy'].mean().sort_values(ascending=False)
    model_dir.plot(kind='bar', color='lightgreen')
    plt.title('Average Directional Accuracy by Model Type')
    plt.ylabel('Directional Accuracy (higher is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 3: Box plot of RMSE by model type
    plt.subplot(2, 2, 3)
    df.boxplot(column='rmse', by='model_type', grid=False)
    plt.title('RMSE Distribution by Model Type')
    plt.suptitle('')  # Remove auto-generated title
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 4: Box plot of directional accuracy by model type
    plt.subplot(2, 2, 4)
    df.boxplot(column='directional_accuracy', by='model_type', grid=False)
    plt.title('Directional Accuracy Distribution by Model Type')
    plt.suptitle('')  # Remove auto-generated title
    plt.ylabel('Directional Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, 'model_comparison.png'))
    plt.close()
    
def plot_stock_comparison(df):
    """Plot comparison of different stocks"""
    if df is None or df.empty or len(df['symbol'].unique()) <= 1:
        return
        
    plt.figure(figsize=(16, 10))
    
    # Plot 1: RMSE by stock
    plt.subplot(2, 2, 1)
    stock_rmse = df.groupby('symbol')['rmse'].mean().sort_values()
    stock_rmse.plot(kind='bar', color='coral')
    plt.title('Average RMSE by Stock')
    plt.ylabel('RMSE (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Directional accuracy by stock
    plt.subplot(2, 2, 2)
    stock_dir = df.groupby('symbol')['directional_accuracy'].mean().sort_values(ascending=False)
    stock_dir.plot(kind='bar', color='lightblue')
    plt.title('Average Directional Accuracy by Stock')
    plt.ylabel('Directional Accuracy (higher is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 3: Box plot of RMSE by stock (top 5 stocks by occurrence)
    plt.subplot(2, 2, 3)
    top_stocks = df['symbol'].value_counts().nlargest(5).index
    df_top = df[df['symbol'].isin(top_stocks)]
    if not df_top.empty:
        df_top.boxplot(column='rmse', by='symbol', grid=False)
        plt.title('RMSE Distribution (Top 5 Stocks)')
        plt.suptitle('')  # Remove auto-generated title
        plt.ylabel('RMSE')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 4: Heat map of best model type for each stock
    plt.subplot(2, 2, 4)
    pivot_data = []
    
    for symbol in df['symbol'].unique():
        for model in df['model_type'].unique():
            subset = df[(df['symbol'] == symbol) & (df['model_type'] == model)]
            if not subset.empty:
                avg_rmse = subset['rmse'].mean()
                pivot_data.append({'symbol': symbol, 'model_type': model, 'rmse': avg_rmse})
    
    if pivot_data:
        pivot_df = pd.DataFrame(pivot_data)
        pivot_table = pivot_df.pivot(index='symbol', columns='model_type', values='rmse')
        
        # Get the best model for each stock (lowest RMSE)
        best_models = pivot_table.idxmin(axis=1)
        
        # Create a text-based visualization
        plt.axis('off')
        plt.title('Best Model Type by Stock (Lowest RMSE)')
        
        y_pos = 0
        for symbol, model in best_models.items():
            rmse_value = pivot_table.loc[symbol, model]
            plt.text(0.1, 1 - y_pos*0.1, f"{symbol}: {model} (RMSE: {rmse_value:.2f})", fontsize=12)
            y_pos += 1
            if y_pos > 9:  # Limit to 10 stocks to avoid overcrowding
                break
    
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, 'stock_comparison.png'))
    plt.close()
    
def plot_next_day_prediction_analysis(df):
    """Create visualization of next-day prediction accuracy across models"""
    if df is None or df.empty:
        return
        
    # Check if next_day_error metrics are available
    if 'next_day_error_pct' not in df.columns or df['next_day_error_pct'].isna().all():
        print("No next-day prediction data available for analysis")
        return
    
    plt.figure(figsize=(18, 15))
    
    # Plot 1: Average next-day error percentage by model type
    plt.subplot(3, 2, 1)
    model_error = df.groupby('model_type')['next_day_error_pct'].mean().sort_values()
    bars = model_error.plot(kind='bar', color='salmon')
    plt.title('Average Next-Day Prediction Error by Model (%)', fontsize=14)
    plt.ylabel('Error Percentage (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(model_error):
        plt.text(i, v + 0.1, f'{v:.2f}%', ha='center')
    
    # Plot 2: Box plot of next-day error percentages by model type
    plt.subplot(3, 2, 2)
    df.boxplot(column='next_day_error_pct', by='model_type', grid=False, showfliers=False)
    plt.title('Next-Day Error Distribution by Model Type', fontsize=14)
    plt.suptitle('')  # Remove auto-generated title
    plt.ylabel('Error Percentage')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 3: Next-day error percentage by stock
    plt.subplot(3, 2, 3)
    stock_error = df.groupby('symbol')['next_day_error_pct'].mean().sort_values()
    stock_error[:10].plot(kind='bar', color='lightblue')  # Top 10 stocks only
    plt.title('Average Next-Day Prediction Error by Stock (%)', fontsize=14)
    plt.ylabel('Error Percentage (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 4: Correlation between directional accuracy and error percentage
    plt.subplot(3, 2, 4)
    plt.scatter(df['directional_accuracy'], df['next_day_error_pct'], alpha=0.6)
    plt.title('Directional Accuracy vs. Next-Day Error', fontsize=14)
    plt.xlabel('Directional Accuracy')
    plt.ylabel('Next-Day Error Percentage')
    
    # Add a linear regression line
    if len(df) > 1:
        z = np.polyfit(df['directional_accuracy'], df['next_day_error_pct'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['directional_accuracy'].min(), df['directional_accuracy'].max(), 100)
        plt.plot(x_range, p(x_range), 'r--', alpha=0.8)
        
        # Add correlation coefficient
        corr = df['directional_accuracy'].corr(df['next_day_error_pct'])
        plt.annotate(f'Correlation: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Histogram of next-day prediction errors
    plt.subplot(3, 2, 5)
    plt.hist(df['next_day_error_pct'], bins=20, color='lightgreen', alpha=0.7)
    plt.title('Distribution of Next-Day Prediction Errors', fontsize=14)
    plt.xlabel('Error Percentage')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add mean and median lines
    mean_error = df['next_day_error_pct'].mean()
    median_error = df['next_day_error_pct'].median()
    plt.axvline(mean_error, color='r', linestyle='--', alpha=0.8, label=f'Mean: {mean_error:.2f}%')
    plt.axvline(median_error, color='b', linestyle='--', alpha=0.8, label=f'Median: {median_error:.2f}%')
    plt.legend()
    
    # Plot 6: Time series of next-day errors (if dates are available)
    plt.subplot(3, 2, 6)
    
    # Create a heatmap of model performance by stock
    pivot_data = []
    for symbol in df['symbol'].unique():
        for model in df['model_type'].unique():
            subset = df[(df['symbol'] == symbol) & (df['model_type'] == model)]
            if not subset.empty:
                avg_error = subset['next_day_error_pct'].mean()
                pivot_data.append({'symbol': symbol, 'model_type': model, 'error': avg_error})
    
    if pivot_data:
        pivot_df = pd.DataFrame(pivot_data)
        # Filter to top 10 stocks by frequency
        top_stocks = df['symbol'].value_counts().nlargest(10).index
        top_models = df['model_type'].value_counts().index
        
        filtered_pivot = pivot_df[pivot_df['symbol'].isin(top_stocks)]
        
        if not filtered_pivot.empty:
            pivot_table = filtered_pivot.pivot(index='symbol', columns='model_type', values='error')
            
            # Create a text-based visualization of best model by stock
            plt.axis('off')
            plt.title('Best Model for Next-Day Prediction by Stock', fontsize=14)
            
            # Find best model for each stock
            best_models = pivot_table.idxmin(axis=1)
            best_errors = pivot_table.min(axis=1)
            
            for i, (symbol, model) in enumerate(best_models.items()):
                error = best_errors[symbol]
                plt.text(0.1, 0.9 - (i * 0.08), 
                       f"{symbol}: {model} (Error: {error:.2f}%)",
                       fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, 'next_day_prediction_analysis.png'))
    plt.close()
    
    # Create a ranking table of model performance specifically for next-day predictions
    if not df.empty and 'next_day_error_pct' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Model ranking by next-day error
        model_ranking = df.groupby('model_type').agg({
            'next_day_error_pct': ['mean', 'median', 'std', 'count']
        }).sort_values(('next_day_error_pct', 'mean'))
        
        model_count = len(model_ranking)
        cell_text = []
        for idx, row in model_ranking.iterrows():
            cell_text.append([
                idx,
                f"{row[('next_day_error_pct', 'mean')]:.2f}%", 
                f"{row[('next_day_error_pct', 'median')]:.2f}%",
                f"{row[('next_day_error_pct', 'std')]:.2f}%",
                f"{int(row[('next_day_error_pct', 'count')])}"
            ])
        
        # Create the table
        plt.axis('off')
        plt.title('Model Ranking for Next-Day Prediction Accuracy', fontsize=16)
        
        the_table = plt.table(cellText=cell_text, 
                           colLabels=['Model Type', 'Mean Error %', 'Median Error %', 'StdDev %', 'Count'],
                           loc='center',
                           cellLoc='center')
        
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)
        the_table.scale(1.2, 1.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(SUMMARY_DIR, 'next_day_model_ranking.png'))
        plt.close()

def generate_summary_report(df):
    """Generate a text summary report"""
    if df is None or df.empty:
        return
        
    # Basic stats
    total_backtests = len(df)
    unique_stocks = df['symbol'].nunique()
    unique_models = df['model_type'].nunique()
    
    # Performance summary
    avg_rmse = df['rmse'].mean()
    avg_dir_acc = df['directional_accuracy'].mean()
    
    # Next-day prediction analysis
    has_next_day_data = 'next_day_error_pct' in df.columns and not df['next_day_error_pct'].isna().all()
    if has_next_day_data:
        avg_next_day_error = df['next_day_error_pct'].mean()
        best_next_day_idx = df['next_day_error_pct'].idxmin()
        best_next_day_config = df.loc[best_next_day_idx]
        
        # Calculate models with best next-day prediction accuracy
        next_day_by_model = df.groupby('model_type')['next_day_error_pct'].agg(['mean', 'std', 'count'])
        next_day_by_model = next_day_by_model.sort_values('mean')
        
        # Get stocks with best next-day predictions
        next_day_by_stock = df.groupby('symbol')['next_day_error_pct'].agg(['mean', 'std', 'count'])
        next_day_by_stock = next_day_by_stock.sort_values('mean')
    
    # Best configurations
    best_rmse_idx = df['rmse'].idxmin()
    best_rmse_config = df.loc[best_rmse_idx]
    
    best_dir_idx = df['directional_accuracy'].idxmax()
    best_dir_config = df.loc[best_dir_idx]
    
    # Average performance by model type
    model_performance = df.groupby('model_type').agg({
        'rmse': 'mean',
        'directional_accuracy': 'mean'
    }).sort_values('rmse')
    
    # Add next-day error to model performance if available
    if has_next_day_data:
        model_next_day = df.groupby('model_type')['next_day_error_pct'].mean()
        model_performance = model_performance.join(model_next_day)
    
    # Generate report
    report = [
        "========== STOCK PREDICTION MODEL BACKTEST SUMMARY ==========",
        f"Generated on: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total backtests: {total_backtests}",
        f"Unique stocks tested: {unique_stocks}",
        f"Model types evaluated: {unique_models}",
        "",
        "OVERALL PERFORMANCE",
        f"Average RMSE: {avg_rmse:.4f}",
        f"Average Directional Accuracy: {avg_dir_acc:.4f}"
    ]
    
    if has_next_day_data:
        report.append(f"Average Next-Day Prediction Error: {avg_next_day_error:.2f}%")
    
    report.extend([
        "",
        "BEST CONFIGURATIONS",
        f"Best RMSE: {best_rmse_config['rmse']:.4f}",
        f"  Stock: {best_rmse_config['symbol']}",
        f"  Model: {best_rmse_config['model_type']}",
        f"  Time period: {best_rmse_config['test_start'].strftime('%Y-%m-%d')} to {best_rmse_config['test_end'].strftime('%Y-%m-%d')}",
        "",
        f"Best Directional Accuracy: {best_dir_config['directional_accuracy']:.4f}",
        f"  Stock: {best_dir_config['symbol']}",
        f"  Model: {best_dir_config['model_type']}",
        f"  Time period: {best_dir_config['test_start'].strftime('%Y-%m-%d')} to {best_dir_config['test_end'].strftime('%Y-%m-%d')}"
    ])
    
    if has_next_day_data:
        report.extend([
            "",
            "NEXT-DAY PREDICTION PERFORMANCE",
            f"Average Next-Day Prediction Error: {avg_next_day_error:.2f}%",
            f"Best Next-Day Prediction: {best_next_day_config['next_day_error_pct']:.2f}% error",
            f"  Stock: {best_next_day_config['symbol']}",
            f"  Model: {best_next_day_config['model_type']}",
            f"  Time period: {best_next_day_config['test_start'].strftime('%Y-%m-%d')} to {best_next_day_config['test_end'].strftime('%Y-%m-%d')}",
            "",
            "NEXT-DAY PREDICTION BY MODEL TYPE",
        ])
        
        # Add model-specific next-day metrics
        for idx, row in next_day_by_model.iterrows():
            report.append(f"  {idx}: {row['mean']:.2f}% error (±{row['std']:.2f}%, n={int(row['count'])})")
            
        report.append("")
        report.append("NEXT-DAY PREDICTION CHAMPIONS")
        # Top 5 stocks by next-day error
        for i, (stock, row) in enumerate(next_day_by_stock.head(5).iterrows()):
            report.append(f"  {i+1}. {stock}: {row['mean']:.2f}% error (±{row['std']:.2f}%, n={int(row['count'])})")
    
    report.extend([
        "",
        "MODEL TYPE COMPARISON"
    ])
    
    for model, row in model_performance.iterrows():
        model_report = f"  {model}: RMSE={row['rmse']:.4f}, Dir. Acc={row['directional_accuracy']:.4f}"
        if has_next_day_data and 'next_day_error_pct' in row:
            model_report += f", Next-Day Error={row['next_day_error_pct']:.2f}%"
        report.append(model_report)
    
    report.append("")
    report.append("STOCK PERFORMANCE")
    
    # Top 5 stocks by RMSE
    stock_rmse = df.groupby('symbol')['rmse'].mean().sort_values()
    for i, (stock, rmse) in enumerate(stock_rmse.head(5).items()):
        report.append(f"  {i+1}. {stock}: RMSE={rmse:.4f}")
    
    report.append("")
    report.append("==========================================================")
    
    # Save report
    report_path = os.path.join(SUMMARY_DIR, 'backtest_summary.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to {report_path}")
    
    # Also save the dataframe as CSV
    csv_path = os.path.join(SUMMARY_DIR, 'backtest_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to {csv_path}")

def find_best_model_overall(df):
    """Find the best model type overall based on average metrics"""
    if df is None or df.empty:
        return None
        
    model_metrics = df.groupby('model_type').agg({
        'rmse': 'mean',
        'mae': 'mean',
        'directional_accuracy': 'mean'
    })
    
    # Normalize metrics (lower is better for RMSE/MAE, higher is better for directional accuracy)
    normalized = pd.DataFrame()
    normalized['rmse'] = (model_metrics['rmse'] - model_metrics['rmse'].min()) / (model_metrics['rmse'].max() - model_metrics['rmse'].min())
    normalized['mae'] = (model_metrics['mae'] - model_metrics['mae'].min()) / (model_metrics['mae'].max() - model_metrics['mae'].min())
    normalized['directional_accuracy'] = 1 - ((model_metrics['directional_accuracy'] - model_metrics['directional_accuracy'].min()) / 
                                           (model_metrics['directional_accuracy'].max() - model_metrics['directional_accuracy'].min()))
    
    # Calculate overall score (lower is better)
    normalized['overall_score'] = normalized['rmse'] * 0.4 + normalized['mae'] * 0.2 + normalized['directional_accuracy'] * 0.4
    
    # Find best model
    best_model = normalized['overall_score'].idxmin()
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Bar chart of overall scores
    overall_scores = normalized['overall_score'].sort_values()
    overall_scores.plot(kind='bar', color='lightblue')
    plt.title('Model Comparison - Overall Score (lower is better)')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, 'best_model_overall.png'))
    plt.close()
    
    return best_model
    
def plot_prediction_horizons(df):
    """Plot how model performance varies with prediction horizon"""
    if df is None or df.empty:
        return
        
    # Check if we have different prediction_days values
    if df['prediction_days'].nunique() <= 1:
        return
        
    plt.figure(figsize=(14, 6))
    
    # Plot 1: RMSE by prediction horizon
    plt.subplot(1, 2, 1)
    horizon_rmse = df.groupby('prediction_days')['rmse'].mean()
    horizon_rmse.plot(marker='o', linestyle='-', color='blue')
    plt.title('RMSE by Prediction Horizon')
    plt.xlabel('Forecast Days')
    plt.ylabel('RMSE (lower is better)')
    plt.grid(linestyle='--', alpha=0.7)
    
    # Plot 2: Directional accuracy by prediction horizon
    plt.subplot(1, 2, 2)
    horizon_dir = df.groupby('prediction_days')['directional_accuracy'].mean()
    horizon_dir.plot(marker='o', linestyle='-', color='green')
    plt.title('Directional Accuracy by Prediction Horizon')
    plt.xlabel('Forecast Days')
    plt.ylabel('Directional Accuracy (higher is better)')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, 'prediction_horizons.png'))
    plt.close()

def main():
    """Generate a comprehensive summary of backtest results"""
    print("Generating backtest summary...")
    
    # Load results
    results = load_backtest_results()
    
    if not results:
        print("No results to analyze.")
        return
    
    # Create summary dataframe
    df = create_summary_dataframe(results)
    
    if df is None:
        print("Could not create summary dataframe.")
        return
    
    print(f"Analyzing {len(df)} backtest results...")
    
    # Generate visualizations
    plot_model_comparison(df)
    plot_stock_comparison(df)
    plot_prediction_horizons(df)
    plot_next_day_prediction_analysis(df)
    
    # Find best model
    best_model = find_best_model_overall(df)
    if best_model:
        print(f"Best overall model: {best_model}")
    
    # Generate summary report
    generate_summary_report(df)
    
    print(f"Summary completed. Results saved to {SUMMARY_DIR}")
    print(f"Next-day prediction analysis saved to {os.path.join(SUMMARY_DIR, 'next_day_prediction_analysis.png')}")
    print(f"Model ranking for next-day predictions saved to {os.path.join(SUMMARY_DIR, 'next_day_model_ranking.png')}")

if __name__ == "__main__":
    main() 