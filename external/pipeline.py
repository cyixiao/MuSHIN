import os
import time
import logging
import traceback
import pandas as pd
import cobra
import warnings
import gc
import sys
from contextlib import contextmanager
from utils import create_pool
from predict import predict
from similarity import get_similarity_score
from gapfill import parallel_process_genomes, get_fermentation_product_improved

# Setup global error handler and warning suppression
@contextmanager
def suppress_stderr_and_warnings():
    """Temporarily suppress stderr output and warnings."""
    original_stderr = sys.stderr
    original_warn_filters = warnings.filters.copy()
    null_file = open(os.devnull, 'w')
    sys.stderr = null_file
    warnings.filterwarnings('ignore')
    try:
        yield
    finally:
        sys.stderr.close() 
        sys.stderr = original_stderr
        warnings.filters = original_warn_filters

def run_metabolic_gapfill_pipeline(directory="Zimmermann2021GenBiol", pipeline="carveme", method="GPR_POOL",
                                  n_jobs=-1, nselect_values=None, strategy="advanced"):
    """
    Run the complete metabolic gap-filling pipeline.
    
    This function integrates all steps of the metabolic gap-filling process:
    1. Creating and merging reaction pools from genome-scale metabolic models
    2. Predicting scores for candidate reactions using neural networks
    3. Calculating similarity scores between reactions
    4. Performing metabolic gap-filling to improve models
    
    Args:
        directory: Base directory name for data
        pipeline: Modeling pipeline name ('carveme' or 'modelseed')
        method: Method for reaction selection ('GPR_POOL')
        n_jobs: Number of CPU cores to use (-1 for all available)
        nselect_values: List of numbers of reactions to select for gap-filling [default: [100, 200]]
        strategy: Gap-filling strategy ('advanced' or 'balanced')
        
    Returns:
        Dictionary with results for each nselect value
    """
    # Set up base directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    SIMILARITY_SCORES_DIR = os.path.join(RESULTS_DIR, 'similarity_scores')
    FBA_RESULT_DIR = os.path.join(RESULTS_DIR, 'fba_result')
    
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
    METADATA_DIR = os.path.join(DATA_DIR, 'metadata', 'Zimmermann2021GenBiol_metadata')
    MODELS_DIR = os.path.join(DATA_DIR, 'Models_EGC_removed', 'Zimmermann2021GenBiol', 'carveme')
    POOLS_DIR = os.path.join(DATA_DIR, 'pools')
    
    # Set default nselect values if not provided
    if nselect_values is None:
        nselect_values = [100, 200]
    
    # Create directory structure
    directory_pipeline = directory + "_" + pipeline
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results', 'fba_result'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results', 'similarity_scores'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, directory_pipeline), exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(BASE_DIR, f"{directory_pipeline}/pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger("metabolic_pipeline")
    warnings.filterwarnings("ignore")
    
    # Start pipeline
    logger.info("=== Starting Metabolic Network Reconstruction Pipeline ===")
    logger.info(f"Pipeline: {pipeline}, Directory: {directory}")
    
    try:
        # Step 1: Create and merge reaction pools
        logger.info("=== Step 1: Creating and merging reaction pools ===")
        create_pool()
        
        # Step 2: Predict scores for candidate reactions
        logger.info("=== Step 2: Predicting reaction scores with MuSHIN ===")
        predict()
        
        # Step 3: Calculate similarity scores
        logger.info("=== Step 3: Calculating reaction similarities ===")
        get_similarity_score(top_N=2000, n_jobs=n_jobs)
        
        # Step 4: Perform gap-filling
        logger.info("=== Step 4: Performing metabolic gap-filling ===")
        
        # Read metadata
        df_genome = pd.read_csv(os.path.join(METADATA_DIR, 'organisms2.csv'), sep='\t', index_col=0)
        df_genome = df_genome[df_genome.tax=='Bacteria']
        available_genomes = []
        
        # Process similarity scores directory
        similarity_dir = os.path.join(SIMILARITY_SCORES_DIR)
        
        if not os.path.exists(similarity_dir):
            logger.info(f"Original directory does not exist: {similarity_dir}")
            potential_dirs = [
                os.path.join(BASE_DIR, 'results', 'similarity_scores'),
                os.path.join(BASE_DIR, 'similarity_scores'),
                os.path.join(BASE_DIR, 'results')
            ]
            for alt_dir in potential_dirs:
                if os.path.exists(alt_dir):
                    similarity_dir = alt_dir
                    logger.info(f"Using alternative similarity scores directory: {similarity_dir}")
                    break
        
        if os.path.exists(similarity_dir):
            for g in os.listdir(similarity_dir):
                if g.endswith(".csv"):
                    available_genomes.append(g.rstrip('.csv'))
            df_genome = df_genome.loc[[g for g in available_genomes if g in df_genome.index]]
            logger.info(f"Found {len(df_genome)} available genomes")
        else:
            logger.error(f"Similarity scores directory does not exist: {similarity_dir}")
            return None
        
        # Load universal reaction library
        namespace = "bigg" if pipeline == "carveme" else "modelseed"
        universe_file = os.path.join(POOLS_DIR, f"{namespace}_universe.xml")
        
        if not os.path.exists(universe_file):
            logger.error(f"Universal reaction library file does not exist: {universe_file}")
            potential_paths = [
                os.path.join(DATA_DIR, 'pools', f'{namespace}_universe.xml'),
                os.path.join(BASE_DIR, f'{namespace}_universe.xml')
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    universe_file = path
                    logger.info(f"Found alternative universal reaction library: {universe_file}")
                    break
            else:
                logger.error("Unable to find universal reaction library file")
                return None
        
        logger.info(f"Loading universal reaction library: {universe_file}")
        with suppress_stderr_and_warnings():
            universe = cobra.io.read_sbml_model(universe_file)
            # Resolve objective function issues
            if len(universe.reactions) > 0:
                default_obj_reaction = universe.reactions[0]
                default_obj_reaction.objective_coefficient = 1.0
                universe.objective = default_obj_reaction
        logger.info(f"Successfully loaded universal reaction library with {len(universe.reactions)} reactions")
        
        # Run gap-filling for each specified nselect value
        results = {}
        for nselect in nselect_values:
            logger.info(f"=== Processing with nselect = {nselect} ===")
            result_file = os.path.join(BASE_DIR, f"{directory_pipeline}/ferm_prod_{strategy}_{nselect}.csv")
            
            if os.path.exists(result_file):
                logger.info(f"Loading existing results from {result_file}")
                results[nselect] = pd.read_csv(result_file)
                continue
                
            logger.info(f"Processing configuration: Method {method}, Selection {nselect}")
            
            # Use parallel processing if enabled
            retLst = parallel_process_genomes(
                directory, pipeline, method, nselect, universe, 
                list(df_genome.index), n_jobs=n_jobs, strategy=strategy
            )
            
            if retLst is not None:
                retLst.to_csv(result_file, index=False)
                logger.info(f"Processing completed successfully, results saved to {result_file}")
                results[nselect] = retLst
            else:
                logger.error(f"Gap-filling failed for nselect={nselect}")
                
            # Force garbage collection after processing
            gc.collect()
        
        logger.info("=== Pipeline completed successfully! ===")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution error: {str(e)}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Example usage
    results = run_metabolic_gapfill_pipeline(
        directory="Zimmermann2021GenBiol",
        pipeline="carveme",
        method="GPR_POOL",
        n_jobs=-1,
        nselect_values=[100, 200],
        strategy="advanced"
    )
