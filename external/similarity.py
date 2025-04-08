import os
import pandas as pd
import numpy as np
import cobra
from cobra.util.array import create_stoichiometric_matrix
import scipy.spatial.distance as distance
from tqdm import tqdm
import contextlib
import io
import logging
from utils import get_data
import warnings
from joblib import Parallel, delayed
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("cobra").setLevel(logging.ERROR)


def preprocess_stoichiometric_matrix(matrix):
    """
    Preprocess the stoichiometric matrix for better similarity calculation.
    
    Args:
        matrix: Stoichiometric matrix to preprocess
        
    Returns:
        Normalized matrix
    """
    # Remove all-zero rows
    nonzero_rows = np.sum(np.abs(matrix), axis=1) > 0
    matrix = matrix[nonzero_rows, :]
    
    # Normalize rows so each metabolite has equal weight
    row_norms = np.sqrt(np.sum(matrix**2, axis=1)).reshape(-1, 1)
    # Avoid division by zero
    row_norms[row_norms == 0] = 1.0
    matrix_normalized = matrix / row_norms
    
    return matrix_normalized


def calculate_multiple_similarities(candidate_matrix, model_matrix, feature=None):
    """
    Calculate multiple similarity metrics between candidate reactions and model reactions.
    
    Args:
        candidate_matrix: Matrix of candidate reactions
        model_matrix: Matrix of model reactions
        feature: Optional pre-computed feature matrix
        
    Returns:
        Dictionary of similarity measures
    """
    # If no preprocessed feature matrix is provided, create and preprocess one
    if feature is None:
        feature = np.concatenate((candidate_matrix, model_matrix), axis=1)
        feature = feature[np.sum(np.abs(feature), axis=1) > 0, :]
    
    # Normalize feature matrix
    feature_normalized = preprocess_stoichiometric_matrix(feature)
    
    # 1. Cosine similarity (better for sparse matrices)
    cosine_matrix = 1 - distance.cdist(feature_normalized.T, feature_normalized.T, 'cosine')
    cosine_max = cosine_matrix[:candidate_matrix.shape[1], candidate_matrix.shape[1]:].max(axis=1)
    cosine_mean = cosine_matrix[:candidate_matrix.shape[1], candidate_matrix.shape[1]:].mean(axis=1)
    
    # 2. Jaccard similarity (compares reaction pattern overlap)
    jaccard_sim = np.zeros(candidate_matrix.shape[1])
    for i in range(candidate_matrix.shape[1]):
        c_pattern = (candidate_matrix[:, i] != 0).astype(int)
        max_jaccard = 0
        for j in range(model_matrix.shape[1]):
            m_pattern = (model_matrix[:, j] != 0).astype(int)
            intersection = np.sum(c_pattern & m_pattern)
            union = np.sum(c_pattern | m_pattern)
            if union > 0:
                jaccard = intersection / union
                max_jaccard = max(max_jaccard, jaccard)
        jaccard_sim[i] = max_jaccard
    
    # 3. Correlation distance (maintain original method, but don't take absolute value, 
    # so negative correlations aren't treated as high similarity)
    corr_matrix = 1 - distance.cdist(feature_normalized.T, feature_normalized.T, 'correlation')
    # Only keep positive correlations
    corr_matrix[corr_matrix < 0] = 0
    corr_max = corr_matrix[:candidate_matrix.shape[1], candidate_matrix.shape[1]:].max(axis=1)
    
    # 4. Composite similarity (default weights can be adjusted based on performance)
    composite_sim = 0.5 * cosine_max + 0.3 * jaccard_sim + 0.2 * corr_max
    
    return {
        'cosine_max': cosine_max,
        'cosine_mean': cosine_mean,
        'jaccard': jaccard_sim,
        'correlation': corr_max,
        'composite': composite_sim
    }


def process_single_sample(sample, input_path, output_path, model_dir, model_pool, top_N):
    """
    Process a single sample file for parallel computation.
    
    Args:
        sample: Sample filename
        input_path: Path to input files
        output_path: Path to output files
        model_dir: Directory containing models
        model_pool: Combined model pool
        top_N: Number of top reactions to consider
        
    Returns:
        Sample name if successful, None otherwise
    """
    if not sample.endswith('.csv'):
        return None
    
    try:
        model_pool_copy = model_pool.copy()
        # Read prediction scores
        scores = pd.read_csv(os.path.join(input_path, sample), index_col=0)
        
        # If multiple columns, calculate mean
        if scores.shape[1] > 1:
            scores = scores.mean(axis=1)
        else:
            scores = scores.iloc[:, 0]
            
        scores = scores.sort_values(ascending=False)
        
        # Load model
        model_filename = sample[:-4] + '.xml'
        # Modified to use the returned model directly
        model = get_data(model_dir, model_filename)
        
        if model is None:
            print(f"Warning: Model {model_filename} loading failed, skipping this file.")
            return None
        
        rxns = [rxn.id for rxn in model.reactions]
        
        # Merge models and build stoichiometric matrix
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            logger = logging.getLogger("cobra")
            old_level = logger.getEffectiveLevel()
            logger.setLevel(logging.ERROR)
            try:
                model_pool_copy.merge(model)
            finally:
                logger.setLevel(old_level)
        
        model_pool_df = create_stoichiometric_matrix(model_pool_copy, array_type='DataFrame')
        
        # Select candidate reactions
        candidate_rxns = scores.index.tolist()[:top_N]
        candidate_rxns = [rxn for rxn in candidate_rxns if rxn in model_pool_df.columns]
        
        if len(candidate_rxns) == 0:
            print(f"Warning: No candidate reactions for {sample} appear in the model pool, skipping this file.")
            return None
        
        # Extract matrices
        candidate_matrix = model_pool_df[candidate_rxns].values
        model_matrix = model_pool_df[rxns].values
        
        # Calculate multiple similarity scores
        similarities = calculate_multiple_similarities(candidate_matrix, model_matrix)
        
        # Build result dataframe
        result_df = pd.DataFrame({
            'predicted_scores': scores.loc[candidate_rxns].values,
            'cosine_similarity_max': similarities['cosine_max'],
            'cosine_similarity_mean': similarities['cosine_mean'],
            'jaccard_similarity': similarities['jaccard'],
            'correlation_similarity': similarities['correlation'],
            'composite_similarity': similarities['composite']
        }, index=candidate_rxns)
        
        # Save results
        result_df.to_csv(os.path.join(output_path, sample))
        return sample
        
    except Exception as e:
        print(f"Error processing sample {sample}: {e}")
        return None


def get_similarity_score(top_N, input_path=None, output_path=None, model_dir=None, n_jobs=None):
    """
    Calculate reaction similarity scores, supports multiprocessing parallel computation.
    
    Args:
        top_N: Number of top reactions to consider
        input_path: Path to input files
        output_path: Path to output files
        model_dir: Directory containing models
        n_jobs: Number of parallel jobs
        
    Returns:
        None
    """
    # Default path settings
    if input_path is None:
        input_path = 'results/fba_result'
    if output_path is None:
        output_path = 'results/similarity_scores'
    if model_dir is None:
        model_dir = '../data/gems/draft_gems'  # Modified default path
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load global model pool
    try:
        model_pool = cobra.io.read_sbml_model('../data/pool/universe_with_smiles.xml')  # Modified path
    except Exception as e:
        print(f"Error loading universe model: {e}")
        return
    
    # Get all prediction score files
    all_files = sorted([f for f in os.listdir(input_path) if f.endswith('.csv')])
    
    # Determine CPU cores
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # Process samples in parallel
    print(f"Processing {len(all_files)} sample files using {n_jobs} CPU cores...")
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_sample)(
            sample, input_path, output_path, model_dir, model_pool, top_N
        ) for sample in all_files
    )
    
    # Count successfully processed samples
    processed = [r for r in results if r is not None]
    print(f"Successfully processed {len(processed)}/{len(all_files)} sample files")


def select_diverse_candidates(similarity_file, threshold=0.3, max_select=200):
    """
    Select diverse candidate reactions from similarity file, avoiding functional redundancy.
    
    Args:
        similarity_file: Path to similarity score file
        threshold: Similarity threshold for diversity
        max_select: Maximum number of reactions to select
        
    Returns:
        List of selected reactions
    """
    try:
        df = pd.read_csv(similarity_file, index_col=0)
        
        # Prioritize prediction scores
        if 'predicted_scores' in df.columns:
            df = df[df['predicted_scores'] >= 0.9995]
        
        # Use composite similarity for sorting
        if 'composite_similarity' in df.columns:
            sort_column = 'composite_similarity'
        elif 'similarity_scores' in df.columns:
            sort_column = 'similarity_scores'
        else:
            sort_column = df.columns[0]
        
        # Sort by prediction scores (descending), then by similarity (ascending)
        df = df.sort_values(['predicted_scores', sort_column], ascending=[False, True])
        
        selected = []
        similarities = []
        
        # Greedy selection of diverse reactions
        for idx, row in df.iterrows():
            if len(selected) >= max_select:
                break
                
            # Check similarity with already selected reactions
            is_diverse = True
            for sim in similarities:
                # Calculate similarity between reactions
                # Use values from the pre-calculated similarity matrix instead of recalculating
                if sim > threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(idx)
                similarities.append(row[sort_column])
        
        return selected
        
    except Exception as e:
        print(f"Error selecting diverse candidates: {e}")
        return []


if __name__ == '__main__':
    # Example usage
    # get_similarity_score(top_N=2000)
    # Multiprocessing example
    get_similarity_score(top_N=2000, n_jobs=8)
   
    # Diverse candidate reaction selection example
    # selected_rxns = select_diverse_candidates('output_path/sample.csv', threshold=0.3, max_select=200)
    # print(f"Selected {len(selected_rxns)} diverse candidate reactions")