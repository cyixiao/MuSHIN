import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gapfill_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Define product to reaction mapping
PRODUCT_MAPPING = {
    'acetic acid': 'EX_ac_e',
    'butyric acid': 'EX_but_e',
    'ethanol': 'EX_etoh_e',
    'formic acid': 'EX_for_e',
    'H2': 'EX_h2_e',
    'DL-lactic acid': ['EX_lac__D_e', 'EX_lac__L_e'],
    'n-butanol': 'EX_btoh_e',
    'propionic acid': 'EX_ppa_e',
    'succinic acid': 'EX_succ_e',
    'acetone': 'EX_acetone_e'
}

def load_experimental_data(result_dir='.'):
    """Load experimental reference data"""
    exp_file = os.path.join(result_dir, 'ferm_prod_exp.csv')
    if not os.path.exists(exp_file):
        logger.error(f"Experimental data file not found: {exp_file}")
        return None
    
    try:
        df_exp = pd.read_csv(exp_file)
        # Check if 'id' column exists as index
        if 'id' in df_exp.columns:
            df_exp = df_exp.set_index('id')
        elif 'genome' in df_exp.columns:
            df_exp = df_exp.set_index('genome')
        logger.info(f"Successfully loaded experimental data with {len(df_exp)} genomes")
        return df_exp
    except Exception as e:
        logger.error(f"Error reading experimental data: {e}")
        return None

def find_strategy_files(result_dir='.'):
    """Find all strategy files"""
    file_pattern = os.path.join(result_dir, 'ferm_prod_*_*.csv')
    strategy_files = glob.glob(file_pattern)
    
    if not strategy_files:
        logger.error(f"No matching strategy files found: {file_pattern}")
        return []
    
    logger.info(f"Found {len(strategy_files)} strategy files")
    return strategy_files

def extract_strategy_info(file_path):
    """Extract strategy and nselect value from filename"""
    file_name = os.path.basename(file_path)
    parts = file_name.replace('ferm_prod_', '').split('_')
    
    if len(parts) >= 2:
        strategy = parts[0]
        nselect = parts[1].replace('.csv', '')
        
        # Try to convert nselect to integer
        try:
            nselect = int(nselect)
        except ValueError:
            pass
            
        return strategy, nselect, file_name
    else:
        logger.warning(f"Unable to extract strategy info from {file_name}")
        return None, None, file_name

def process_file(file_path, exp_data):
    """Process a single strategy file and calculate evaluation metrics"""
    logger.info(f"Processing file: {file_path}")

    # Extract strategy info from filename
    strategy, nselect, file_name = extract_strategy_info(file_path)
    if strategy is None:
        return None
    
    # Read prediction results
    try:
        df_pred = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None
        
    if df_pred.empty:
        logger.warning(f"Warning: {file_path} file is empty")
        return None
    
    # Calculate predicted values
    try:
        df_pred['predicted'] = ((df_pred['minimum'] + df_pred['maximum']) / 2 > 0).astype(int)
    except KeyError as e:
        logger.error(f"Error calculating predicted values: missing required column in {file_path} - {e}")
        return None
    
    # Convert to wide format
    try:
        # Use genome or id as index
        index_col = 'genome' if 'genome' in df_pred.columns else 'id'
        df_pivot = df_pred.pivot_table(index=index_col, columns='reaction', values='predicted', aggfunc='max')
    except KeyError as e:
        logger.error(f"Error converting to wide format: missing required column in {file_path} - {e}")
        return None
    
    # Calculate product predictions
    df_pred_products = pd.DataFrame(index=df_pivot.index)
    for product, reactions in PRODUCT_MAPPING.items():
        if isinstance(reactions, list):
            # Ensure all reaction columns exist
            available_reactions = [r for r in reactions if r in df_pivot.columns]
            if available_reactions:
                df_pred_products[product] = df_pivot[available_reactions].max(axis=1)
            else:
                df_pred_products[product] = 0
        else:
            if reactions in df_pivot.columns:
                df_pred_products[product] = df_pivot[reactions]
            else:
                df_pred_products[product] = 0
    
    # Merge predictions and true data
    df_combined = exp_data.join(df_pred_products, how='inner', lsuffix='_exp', rsuffix='_pred')
    if df_combined.empty:
        logger.warning(f"Warning: {file_path} resulted in empty dataset after merging with experimental data")
        return None
    
    # Get product list
    products = list(PRODUCT_MAPPING.keys())
    
    # Prepare metrics dictionary
    metrics = {
        'strategy': strategy,
        'nselect': nselect,
        'file': file_name,
        'genomes': len(df_combined)
    }
    
    # For overall metrics calculation
    all_gt = []
    all_pred = []
    
    # For each product and genome
    for genome in df_combined.index:
        # Extract true and predicted values
        gt_list = []
        pred_list = []
        
        for product in products:
            # Get true and predicted values for each product
            gt_val = df_combined.loc[genome, f"{product}_exp"]
            pred_val = df_combined.loc[genome, f"{product}_pred"]
            
            # Ensure boolean values then convert to integers
            gt_val = bool(gt_val) * 1
            pred_val = bool(pred_val) * 1
            
            gt_list.append(gt_val)
            pred_list.append(pred_val)
        
        # Convert to numpy arrays
        gt = np.array(gt_list)
        pred = np.array(pred_list)
        
        # Record overall data
        all_gt.extend(gt)
        all_pred.extend(pred)
    
    # Calculate overall metrics
    metrics['overall_precision'] = precision_score(all_gt, all_pred, zero_division=0)
    metrics['overall_recall'] = recall_score(all_gt, all_pred, zero_division=0)
    metrics['overall_f1'] = f1_score(all_gt, all_pred, zero_division=0)
    metrics['overall_auprc'] = average_precision_score(all_gt, all_pred) if sum(all_pred) > 0 else 0
    
    logger.info(f"Strategy: {strategy}_{nselect}, F1 Score: {metrics['overall_f1']:.4f}, Precision: {metrics['overall_precision']:.4f}, Recall: {metrics['overall_recall']:.4f}, AUPRC: {metrics['overall_auprc']:.4f}")
    
    return metrics

def analyze_strategies(result_dir='.'):
    """Analyze all strategy files and find the best algorithm"""
    logger.info("Starting gap fill strategy analysis...")
    
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)
    
    # Find all strategy files
    strategy_files = find_strategy_files(result_dir)
    if not strategy_files:
        return
    
    # Read experimental reference data
    df_exp = load_experimental_data(result_dir)
    if df_exp is None:
        return
    
    # Process each strategy file
    results = []
    for file_path in strategy_files:
        metrics = process_file(file_path, df_exp)
        if metrics:
            results.append(metrics)
    
    if not results:
        logger.error("No successfully processed strategy files")
        return
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Save basic results
    results_file = os.path.join(result_dir, 'strategy_metrics.csv')
    results_df.to_csv(results_file, index=False)
    logger.info(f"Metrics saved to {results_file}")
    
    # Find and print the best strategy
    best_idx = results_df['overall_f1'].idxmax()
    best_strategy = results_df.loc[best_idx]
    
    logger.info("\n======== BEST ALGORITHM ========")
    logger.info(f"Strategy: {best_strategy['strategy']}_{best_strategy['nselect']}")
    logger.info(f"F1 Score: {best_strategy['overall_f1']:.4f}")
    logger.info(f"Precision: {best_strategy['overall_precision']:.4f}")
    logger.info(f"Recall: {best_strategy['overall_recall']:.4f}")
    logger.info(f"AUPRC: {best_strategy['overall_auprc']:.4f}")
    logger.info("================================")
    
    return results_df

if __name__ == "__main__":
    print("Starting gap fill strategy analysis...")
    results_df = analyze_strategies()
    if results_df is not None:
        print("Analysis complete.")