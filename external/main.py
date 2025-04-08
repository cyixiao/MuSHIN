import argparse
import os
import logging
import time
from utils import create_pool
from predict import predict
from similarity import get_similarity_score
from pipeline import run_metabolic_gapfill_pipeline
import warnings

def parse_arguments():
    """Parse command line arguments for pipeline configuration."""
    parser = argparse.ArgumentParser(description="Metabolic Network Gap-filling Pipeline")
    
    # Basic configuration
    parser.add_argument("--directory", default="Zimmermann2021GenBiol", 
                        help="Base directory name for data")
    parser.add_argument("--pipeline", default="carveme", choices=["carveme", "modelseed"],
                        help="Modeling pipeline name")
    parser.add_argument("--method", default="GPR_POOL", choices=["GPR_POOL"],
                        help="Method for reaction selection")
    
    # Advanced configuration
    parser.add_argument("--n_jobs", type=int, default=-1, 
                        help="Number of CPU cores to use (-1 for all available)")
    parser.add_argument("--top_n", type=int, default=2000, 
                        help="Number of top reactions to consider in similarity calculation")
    parser.add_argument("--nselect", type=int, nargs="+", default=[100, 200], 
                        help="Numbers of reactions to select for gap-filling")
    parser.add_argument("--strategy", default="advanced", choices=["advanced", "balanced"],
                        help="Gap-filling strategy")
    
    # Execution control
    parser.add_argument("--skip_pool", action="store_true", 
                        help="Skip reaction pool creation step")
    parser.add_argument("--skip_predict", action="store_true", 
                        help="Skip reaction score prediction step")
    parser.add_argument("--skip_similarity", action="store_true", 
                        help="Skip similarity calculation step")
    parser.add_argument("--skip_gapfill", action="store_true", 
                        help="Skip gap-filling step")
    
    return parser.parse_args()

def setup_logging():
    """Configure logging for the pipeline."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    return logging.getLogger("metabolic_pipeline")

def main():
    """Main function to run the complete metabolic gap-filling pipeline."""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("=== Starting Metabolic Gap-filling Pipeline ===")
    logger.info(f"Configuration: {vars(args)}")
    
    # Create results directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/fba_result", exist_ok=True)
    os.makedirs("results/similarity_scores", exist_ok=True)
    os.makedirs("results/universe", exist_ok=True)
    
    try:
        # Step 1: Create and merge reaction pools
        if not args.skip_pool:
            logger.info("=== Step 1: Creating and merging reaction pools ===")
            create_pool()
        else:
            logger.info("Skipping reaction pool creation step")
        
        # Step 2: Predict scores for candidate reactions
        if not args.skip_predict:
            logger.info("=== Step 2: Predicting reaction scores with MuSHIN ===")
            predict()
        else:
            logger.info("Skipping reaction score prediction step")
        
        # Step 3: Calculate similarity scores
        if not args.skip_similarity:
            logger.info("=== Step 3: Calculating reaction similarities ===")
            get_similarity_score(top_N=args.top_n, n_jobs=args.n_jobs)
        else:
            logger.info("Skipping similarity calculation step")
        
        # Step 4: Perform gap-filling
        if not args.skip_gapfill:
            logger.info("=== Step 4: Performing metabolic gap-filling ===")
            results = run_metabolic_gapfill_pipeline(
                directory=args.directory,
                pipeline=args.pipeline,
                method=args.method,
                n_jobs=args.n_jobs,
                nselect_values=args.nselect,
                strategy=args.strategy
            )
            
            if results is not None:
                logger.info("Gap-filling completed successfully")
            else:
                logger.error("Gap-filling failed")
        else:
            logger.info("Skipping gap-filling step")
        
        logger.info("=== Pipeline completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Pipeline execution error: {str(e)}")
        logger.exception("Exception details:")
        return 1
    
    return 0

if __name__ == "__main__":
    main()