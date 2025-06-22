import pandas as pd
import cobra
from cobra import Model, Reaction, Metabolite
import numpy as np
from copy import deepcopy
import networkx as nx
from joblib import Parallel, delayed
import multiprocessing
import time
import logging
import contextlib
import io
import gc
import warnings
from collections import defaultdict, Counter
import sys
import traceback
from scipy.sparse import lil_matrix
from itertools import combinations

import os


# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SIMILARITY_SCORES_DIR = os.path.join(RESULTS_DIR, 'similarity_scores')
FBA_RESULT_DIR = os.path.join(RESULTS_DIR, 'fba_result')

DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata', 'Zimmermann2021GenBiol_metadata')
MODELS_DIR = os.path.join(DATA_DIR, 'Models_EGC_removed', 'Zimmermann2021GenBiol', 'carveme')
POOLS_DIR = os.path.join(DATA_DIR, 'pools')

# Set global error handler, ignore specific warnings
def exception_handler(exctype, value, tb):
    if exctype == UserWarning and "No objective coefficients" in str(value):
        pass
    else:
        sys.__excepthook__(exctype, value, tb)

sys.excepthook = exception_handler

@contextlib.contextmanager
def suppress_stderr_and_warnings():
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

#---------------------------------------------
# Conversion from common name to metabolite id
#---------------------------------------------
name_convertor_modelseed = {
    'acetic acid':['cpd00029'],
    'butyric acid':['cpd00211'],
    'ethanol':['cpd00363'],
    'formic acid':['cpd00047'],
    'H2':['cpd11640'],
    'DL-lactic acid':['cpd00159','cpd00221'],
    'n-butanol':['cpd03662'],
    'propionic acid':['cpd00141'],
    'succinic acid':['cpd00036'],
    'acetone':['cpd00178']
}
target_exchange_reactions_modelseed = [
    'EX_cpd00029_e0','EX_cpd03662_e0','EX_cpd00211_e0',
    'EX_cpd00363_e0','EX_cpd00047_e0','EX_cpd11640_e0',
    'EX_cpd00221_e0','EX_cpd00159_e0','EX_cpd00141_e0',
    'EX_cpd00036_e0','EX_cpd00178_e0'
]
energy_couples_modelseed = {
    'cpd00002_c0': 'cpd00008_c0',
    'cpd00052_c0': 'cpd00096_c0',
    'cpd00038_c0': 'cpd00031_c0',
    'cpd00062_c0': 'cpd00014_c0',
    'cpd00068_c0': 'cpd00090_c0',
    'cpd00005_c0': 'cpd00006_c0',
    'cpd00004_c0': 'cpd00003_c0',
    'cpd00982_c0': 'cpd00015_c0',
    'cpd01270_c0': 'cpd00050_c0',
    'cpd15561_c0': 'cpd15560_c0',
    'cpd15499_c0': 'cpd15500_c0',
    'cpd15994_c0': 'cpd15995_c0',
    'cpd23255_c0': 'cpd11606_c0',
    'cpd15353_c0': 'cpd15352_c0',
    'cpd00022_c0': 'cpd00010_c0',
    'cpd00023_c0': 'cpd00024_c0',
    'cpd00067_p0': 'cpd00067_c0'
}

name_convertor_bigg = {
    'acetic acid':['ac'],
    'butyric acid':['but'],
    'ethanol':['etoh'],
    'formic acid':['for'],
    'H2':['h2'],
    'DL-lactic acid':['lac__L','lac__D'],
    'n-butanol':['btoh'],
    'propionic acid':['ppa'],
    'succinic acid':['succ'],
    'acetone':['acetone']
}
target_exchange_reactions_bigg = [
    'EX_ac_e','EX_btoh_e','EX_but_e',
    'EX_etoh_e','EX_for_e','EX_h2_e',
    'EX_lac__D_e','EX_lac__L_e','EX_ppa_e',
    'EX_succ_e','EX_acetone_e'
]
energy_couples_bigg = {
    'atp_c': 'adp_c',
    'ctp_c': 'cdp_c',
    'gtp_c': 'gdp_c',
    'utp_c': 'udp_c',
    'itp_c': 'idp_c',
    'nadph_c': 'nadp_c',
    'nadh_c': 'nad_c',
    'fadh2_c': 'fad_c',
    'fmnh2_c': 'fmn_c',
    'q8h2_c': 'q8_c',
    'mql8_c': 'mqn8_c',
    'mql6_c': 'mqn6_c',
    'mql7_c': 'mqn7_c',
    '2dmmql8_c': '2dmmq8_c',
    'accoa_c': 'coa_c',
    'glu__L_c': 'akg_c',
    'h_p': 'h_c'
}

# Common metabolite categories for pathway analysis
central_carbon_metabolites = {
    'bigg': ['glc__D_c', 'g6p_c', 'f6p_c', 'fdp_c', 'dhap_c', 'g3p_c', '13dpg_c', '3pg_c', '2pg_c', 
             'pep_c', 'pyr_c', 'accoa_c', 'cit_c', 'icit_c', 'akg_c', 'succoa_c', 'succ_c', 'fum_c', 
             'mal__L_c', 'oaa_c', '6pgc_c', 'ru5p__D_c', 'r5p_c', 'xu5p__D_c', 's7p_c', 'e4p_c'],
    'modelseed': ['cpd00027_c0', 'cpd00079_c0', 'cpd00071_c0', 'cpd00290_c0', 'cpd00363_c0', 'cpd00068_c0', 
                  'cpd00169_c0', 'cpd00482_c0', 'cpd00345_c0', 'cpd00002_c0', 'cpd00022_c0', 'cpd00010_c0', 
                  'cpd00137_c0', 'cpd00198_c0', 'cpd00024_c0', 'cpd00036_c0', 'cpd00130_c0', 'cpd00074_c0']
}

amino_acid_metabolites = {
    'bigg': ['ala__L_c', 'arg__L_c', 'asn__L_c', 'asp__L_c', 'cys__L_c', 'gln__L_c', 'glu__L_c', 'gly_c', 
             'his__L_c', 'ile__L_c', 'leu__L_c', 'lys__L_c', 'met__L_c', 'phe__L_c', 'pro__L_c', 'ser__L_c', 
             'thr__L_c', 'trp__L_c', 'tyr__L_c', 'val__L_c'],
    'modelseed': ['cpd00035_c0', 'cpd00051_c0', 'cpd00132_c0', 'cpd00041_c0', 'cpd00084_c0', 'cpd00053_c0', 
                  'cpd00023_c0', 'cpd00033_c0', 'cpd00119_c0', 'cpd00322_c0', 'cpd00107_c0', 'cpd00039_c0', 
                  'cpd00060_c0', 'cpd00066_c0', 'cpd00129_c0', 'cpd00054_c0', 'cpd00161_c0', 'cpd00065_c0', 
                  'cpd00069_c0', 'cpd00156_c0']
}

# Metabolite functional groups for structural similarity
functional_groups = {
    'bigg': {
        'phosphate': ['_p_', 'ph_', 'phos'],
        'carboxyl': ['coa', 'ac_', 'succ', 'mal'],
        'amino': ['glu', 'asp', 'ala', 'gly', 'ser'],
        'sugar': ['glc', 'gal', 'man', 'xyl', 'rib'],
        'nucleotide': ['atp', 'gtp', 'ctp', 'utp', 'amp', 'gmp', 'cmp', 'ump'],
    },
    'modelseed': {
        'phosphate': ['ph', 'phos'],
        'carboxyl': ['coa', 'ac', 'succ', 'mal'],
        'amino': ['glu', 'asp', 'ala', 'gly', 'ser'],
        'sugar': ['glc', 'gal', 'man', 'xyl', 'rib'],
        'nucleotide': ['atp', 'gtp', 'ctp', 'utp', 'amp', 'gmp', 'cmp', 'ump'],
    }
}

# Global configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("model_reconstruction.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

MAX_WORKERS = -1      # Default: use all available CPUs
BATCH_SIZE = 20       # Batch processing size
MEMORY_THRESHOLD = 0.8  # Memory usage threshold (80%)
MAX_GROWTH_THRESHOLD = 2.81  # Maximum growth threshold
MIN_PREDICTED_SCORE = 0.999  # Minimum prediction score threshold

egc_cache = {}
evaluation_metrics = defaultdict(list)
network_cache = {}
metabolite_pathway_cache = {}

def memory_usage_percent():
    """Get current memory usage as a percentage."""
    try:
        import psutil
        return psutil.virtual_memory().percent / 100.0
    except ImportError:
        return 0.0

def timed_function(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def add_EX_reactions(model, universe, namespace):
    """
    Add exchange reactions to the model.
    
    Args:
        model: COBRA model to add exchange reactions to
        universe: Universal reaction model containing all possible reactions
        namespace: Namespace for reaction IDs ('bigg' or 'modelseed')
        
    Returns:
        Modified COBRA model with added exchange reactions
    """
    try:
        target_exchange_reactions = target_exchange_reactions_bigg if namespace == "bigg" else target_exchange_reactions_modelseed
        for rid in target_exchange_reactions:
            if rid not in model.reactions:
                reaction = Reaction(rid)
                reaction.name = 'R_' + rid
                reaction.lower_bound = -1000.0
                reaction.upper_bound = 1000.0
                met_id = rid.split('EX_')[1]
                if met_id in model.metabolites:
                    met = model.metabolites.get_by_id(met_id)
                else:
                    try:
                        met = universe.metabolites.get_by_id(met_id)
                    except:
                        logger.warning(f"Metabolite {met_id} for exchange reaction {rid} does not exist in universal model")
                        continue
                reaction.add_metabolites({met: -1.0})
                model.add_reactions([reaction])
                logger.debug(f"Added exchange reaction to model: {rid}")
        return model
    except Exception as e:
        logger.error(f"Failed to add exchange reactions: {str(e)}")
        logger.error(traceback.format_exc())
        return model

def constrain_media(directory, genome, model, namespace):
    """
    Set media constraints for the model based on experimental conditions.
    
    Args:
        directory: Directory containing metadata
        genome: Genome ID
        model: COBRA model to constrain
        namespace: Namespace for reaction IDs ('bigg' or 'modelseed')
        
    Returns:
        COBRA model with media constraints
    """
    try:
        media_file = os.path.join(METADATA_DIR, 'media.tsv')
        ferm_prod_file = os.path.join(METADATA_DIR, 'ferm_prod_exp.csv')
        if os.path.exists(media_file) and os.path.exists(ferm_prod_file):
            df_ft = pd.read_csv(media_file, sep='\t')
            df_ferm_prod = pd.read_csv(ferm_prod_file, index_col=0)
            to_exclude = []
            if genome in df_ferm_prod.index:
                for prod in list(df_ferm_prod.loc[genome][df_ferm_prod.loc[genome] > 0].index):
                    if namespace == "bigg":
                        to_exclude.extend(name_convertor_bigg[prod])
                    else:
                        to_exclude.extend(name_convertor_modelseed[prod])
            media_constraints_applied = False
            for ex in model.exchanges:
                if namespace == "bigg":
                    ex_cpd = ex.id.split('EX_')[1].split('_e')[0]
                else:
                    ex_cpd = ex.id.split('EX_')[1].split('_e0')[0]
                if namespace in df_ft.columns and ex_cpd in list(df_ft[namespace]) and ex_cpd not in to_exclude:
                    flux_value = df_ft.loc[df_ft[namespace]==ex_cpd, 'maxFlux'].values[0]
                    ex.lower_bound = -flux_value
                    ex.upper_bound = 1000.0
                    media_constraints_applied = True
                else:
                    ex.lower_bound = 0.0
                    ex.upper_bound = 1000.0
            logger.info(f"Media conditions set from file, applied {media_constraints_applied} constraints")
            if not media_constraints_applied:
                logger.warning("Could not apply media constraints from file, using backup settings")
                apply_backup_media(model, namespace)
            return model
        else:
            logger.warning(f"Media files do not exist: {media_file} or {ferm_prod_file}")
            apply_backup_media(model, namespace)
            return model
    except Exception as e:
        logger.error(f"Failed to set media constraints: {str(e)}")
        logger.error(traceback.format_exc())
        apply_backup_media(model, namespace)
        return model

def apply_backup_media(model, namespace):
    """
    Apply backup media constraints when media files are not available.
    
    Args:
        model: COBRA model to constrain
        namespace: Namespace for reaction IDs ('bigg' or 'modelseed')
        
    Returns:
        None (modifies model in-place)
    """
    for ex in model.exchanges:
        ex.bounds = (0, 0)
    basic_nutrients = {
        "bigg": {
            "EX_glc__D_e": -10.0,
            "EX_nh4_e": -1000.0,
            "EX_pi_e": -1000.0,
            "EX_k_e": -1000.0,
            "EX_na1_e": -1000.0,
            "EX_mg2_e": -1000.0,
            "EX_ca2_e": -1000.0,
            "EX_h2o_e": -1000.0,
            "EX_h_e": -1000.0
        },
        "modelseed": {
            "EX_cpd00027_e0": -10.0,
            "EX_cpd00013_e0": -1000.0,
            "EX_cpd00009_e0": -1000.0,
            "EX_cpd00205_e0": -1000.0,
            "EX_cpd00971_e0": -1000.0,
            "EX_cpd00254_e0": -1000.0,
            "EX_cpd00063_e0": -1000.0,
            "EX_cpd00001_e0": -1000.0,
            "EX_cpd00067_e0": -1000.0
        }
    }
    media_constraints_applied = False
    for ex_id, lb in basic_nutrients[namespace].items():
        if ex_id in model.reactions:
            model.reactions.get_by_id(ex_id).lower_bound = lb
            media_constraints_applied = True
    for ex in model.exchanges:
        if ex.upper_bound == 0:
            ex.upper_bound = 1000.0
    logger.info(f"Applied backup media, set {media_constraints_applied} constraints")

def detec_EGC(model, metabolite_id, rid, namespace):
    """
    Detect Energy Generating Cycles (EGC) in the metabolic model.
    
    Args:
        model: COBRA model to analyze
        metabolite_id: ID of the energy metabolite to check
        rid: Reaction ID to focus on, or None for whole model check
        namespace: Namespace for reaction IDs ('bigg' or 'modelseed')
        
    Returns:
        DataFrame of reactions involved in EGC, or None if no EGC is detected
    """
    try:
        met = model.metabolites.get_by_id(metabolite_id)
        if namespace == "bigg":
            energy_couples = energy_couples_bigg
            if energy_couples[met.id] in model.metabolites:
                dissipation_product = model.metabolites.get_by_id(energy_couples[met.id])
            else:
                return None
        else:
            energy_couples = energy_couples_modelseed
            if energy_couples[met.id] in model.metabolites:
                dissipation_product = model.metabolites.get_by_id(energy_couples[met.id])
            else:
                return None
        dissipation_rxn = Reaction('Dissipation')
        for boundary in model.boundary:
            boundary.bounds = (0, 0)
        model.add_reactions([dissipation_rxn])
        if namespace == "bigg":
            if met.id in ['atp_c', 'ctp_c', 'gtp_c', 'utp_c', 'itp_c']:
                dissipation_rxn.reaction = "h2o_c --> h_c + pi_c"
            elif met.id in ['nadph_c', 'nadh_c']:
                dissipation_rxn.reaction = "--> h_c"
            elif met.id in ['fadh2_c', 'fmnh2_c', 'q8h2_c', 'mql8_c',
                            'mql6_c', 'mql7_c', 'dmmql8_c']:
                dissipation_rxn.reaction = "--> 2 h_c"
            elif met.id == 'accoa_c':
                dissipation_rxn.reaction = "h2o_c --> h_c + ac_c"
            elif met.id == 'glu__L_c':
                dissipation_rxn.reaction = "h2o_c --> 2 h_c + nh3_c"
            elif met.id == 'h_p':
                pass
        else:
            if met.id in ['cpd00002_c0', 'cpd00052_c0', 'cpd00038_c0', 'cpd00062_c0', 'cpd00068_c0']:
                dissipation_rxn.reaction = "cpd00001_c0 --> cpd00067_c0 + cpd00009_c0"
            elif met.id in ['cpd00005_c0', 'cpd00004_c0']:
                dissipation_rxn.reaction = "--> cpd00067_c0"
            elif met.id in ['cpd00982_c0', 'cpd01270_c0', 'cpd15561_c0', 'cpd15499_c0',
                            'cpd15994_c0', 'cpd23255_c0', 'cpd15353_c0']:
                dissipation_rxn.reaction = "--> 2 cpd00067_c0"
            elif met.id == 'cpd00022_c0':
                dissipation_rxn.reaction = "cpd00001_c0 --> cpd00067_c0 +  cpd00029_c0"
            elif met.id == 'cpd00023_c0':
                dissipation_rxn.reaction = "cpd00001_c0 --> 2 cpd00067_c0 +  cpd00013_c0"
            elif met.id == 'cpd00067_p0':
                pass
        dissipation_rxn.add_metabolites({met: -1, dissipation_product: 1})
        model.objective = dissipation_rxn
        if rid is None:
            solution = model.optimize()
        else:
            bounds = model.reactions.get_by_id(rid).bounds
            positive_solution_found = False
            if bounds[1] >= 0.01:
                try:
                    model.reactions.get_by_id(rid).lower_bound = 0.01
                    model.reactions.get_by_id(rid).upper_bound = 1000.0
                    solution = model.optimize()
                    if solution.status == "optimal":
                        positive_solution_found = True
                    else:
                        model.reactions.get_by_id(rid).bounds = bounds
                except:
                    model.reactions.get_by_id(rid).bounds = bounds
            negative_solution_found = False
            if not positive_solution_found and bounds[0] <= -0.01:
                try:
                    model.reactions.get_by_id(rid).lower_bound = -1000.0
                    model.reactions.get_by_id(rid).upper_bound = -0.01
                    solution = model.optimize()
                    if solution.status == 'optimal':
                        negative_solution_found = True
                    else:
                        model.reactions.get_by_id(rid).bounds = bounds
                except:
                    model.reactions.get_by_id(rid).bounds = bounds
            if not positive_solution_found and not negative_solution_found:
                model.reactions.get_by_id(rid).bounds = bounds
                solution = model.optimize()
        if solution.status == "infeasible":
            return None
        elif solution.objective_value > 0.0:
            return solution.fluxes[solution.fluxes.abs() > 0.0].to_frame().drop(["Dissipation"])
        else:
            return None
    except Exception as e:
        logger.warning(f"EGC detection error: {str(e)}")
        return None

def resolve_EGC(model, rid, namespace):
    """
    Resolve Energy Generating Cycles in the model.
    
    Args:
        model: COBRA model to fix
        rid: Reaction ID to focus on
        namespace: Namespace for reaction IDs ('bigg' or 'modelseed')
        
    Returns:
        Boolean indicating whether EGC was resolved
    """
    cache_key = (model.id, rid, namespace)
    if cache_key in egc_cache:
        return egc_cache[cache_key]
    energy_couples = energy_couples_bigg if namespace == "bigg" else energy_couples_modelseed
    for key_met in energy_couples.keys():
        if key_met in model.metabolites:
            df_res = detec_EGC(deepcopy(model), key_met, rid, namespace=namespace)
            if df_res is not None and rid in list(df_res.index):
                rxn = model.reactions.get_by_id(rid)
                if rxn.lower_bound == 0.0 or rxn.upper_bound == 0.0:
                    egc_cache[cache_key] = False
                    return False
                rid_flux = df_res.loc[rid, 'fluxes']
                if rid_flux > 0:
                    rxn.upper_bound = 0.0
                else:
                    rxn.lower_bound = 0.0
                df_res = detec_EGC(deepcopy(model), key_met, rid, namespace=namespace)
                if df_res is not None:
                    egc_cache[cache_key] = False
                    return False
    egc_cache[cache_key] = True
    return True

def identify_dead_end_metabolites(model):
    """
    Identify dead-end metabolites in the model - metabolites that are only consumed or only produced.
    
    Args:
        model: COBRA model to analyze
        
    Returns:
        Set of dead-end metabolite IDs
    """
    dead_ends = set()
    for met in model.metabolites:
        # Skip boundary metabolites
        if met.compartment in ['e', 'e0']:
            continue
        
        # Check if metabolite is a dead end
        producing_rxns = 0
        consuming_rxns = 0
        
        for rxn in met.reactions:
            if rxn.id.startswith('EX_') or rxn.id.startswith('DM_') or rxn.id.startswith('sink_'):
                continue
                
            stoich = rxn.metabolites[met]
            # Note: if stoich < 0, metabolite is consumed; if stoich > 0, metabolite is produced
            if stoich > 0 and rxn.upper_bound > 0:  # Metabolite is produced
                producing_rxns += 1
            elif stoich < 0 and rxn.lower_bound < 0:  # Metabolite is consumed
                consuming_rxns += 1
        
        # If metabolite has only producing or only consuming reactions, it's a dead end
        if (producing_rxns > 0 and consuming_rxns == 0) or (consuming_rxns > 0 and producing_rxns == 0):
            dead_ends.add(met.id)
            
    return dead_ends

def build_metabolite_reaction_graph(model):
    """
    Build a bipartite graph of metabolites and reactions for connectivity analysis.
    
    Args:
        model: COBRA model to analyze
        
    Returns:
        NetworkX directed graph object
    """
    G = nx.DiGraph()
    
    # Add metabolite nodes
    for met in model.metabolites:
        G.add_node(met.id, type='metabolite')
    
    # Add reaction nodes
    for rxn in model.reactions:
        if not rxn.id.startswith('EX_') and not rxn.id.startswith('DM_') and not rxn.id.startswith('sink_'):
            G.add_node(rxn.id, type='reaction')
            
            # Add edges (metabolite -> reaction: metabolite is a substrate)
            # Add edges (reaction -> metabolite: metabolite is a product)
            for met, coef in rxn.metabolites.items():
                if coef < 0:  # Metabolite is a substrate
                    G.add_edge(met.id, rxn.id, weight=abs(coef))
                else:  # Metabolite is a product
                    G.add_edge(rxn.id, met.id, weight=abs(coef))
    
    return G

def calculate_pathway_distance(metabolite1, metabolite2, graph):
    """
    Calculate the distance (path length) between two metabolites in the metabolic network.
    
    Args:
        metabolite1: First metabolite ID
        metabolite2: Second metabolite ID
        graph: NetworkX graph of the metabolic network
        
    Returns:
        Path length (integer) or infinity if no path exists
    """
    if metabolite1 not in graph.nodes or metabolite2 not in graph.nodes:
        return float('inf')
    
    try:
        # Calculate shortest path length
        path_length = nx.shortest_path_length(graph, source=metabolite1, target=metabolite2)
        return path_length
    except nx.NetworkXNoPath:
        return float('inf')

def calculate_structural_similarity(met_id1, met_id2, namespace):
    """
    Estimate structural similarity based on metabolite ID functional groups.
    
    Args:
        met_id1: First metabolite ID
        met_id2: Second metabolite ID
        namespace: Namespace for metabolite IDs ('bigg' or 'modelseed')
        
    Returns:
        Similarity score between 0 and 1
    """
    if not met_id1 or not met_id2:
        return 0
    
    # Check if metabolites belong to the same family
    if (met_id1 in central_carbon_metabolites[namespace] and met_id2 in central_carbon_metabolites[namespace]) or \
       (met_id1 in amino_acid_metabolites[namespace] and met_id2 in amino_acid_metabolites[namespace]):
        return 0.7  # Belong to the same major metabolite category
    
    # Calculate similarity based on functional groups
    groups1 = set()
    groups2 = set()
    
    for group, patterns in functional_groups[namespace].items():
        for pattern in patterns:
            if pattern in met_id1:
                groups1.add(group)
            if pattern in met_id2:
                groups2.add(group)
    
    if not groups1 or not groups2:
        return 0.1  # Basic similarity
    
    # Calculate Jaccard similarity of functional groups
    similarity = len(groups1.intersection(groups2)) / len(groups1.union(groups2))
    return similarity

def get_metabolite_pathway_enrichment(metabolite_ids, namespace):
    """
    Get pathway enrichment information for a set of metabolites.
    
    Args:
        metabolite_ids: List of metabolite IDs
        namespace: Namespace for metabolite IDs ('bigg' or 'modelseed')
        
    Returns:
        Dictionary with pathway distribution
    """
    # Use cache to improve performance
    cache_key = frozenset(metabolite_ids)
    if cache_key in metabolite_pathway_cache:
        return metabolite_pathway_cache[cache_key]
    
    pathway_counts = Counter()
    
    # Find the main pathways for metabolites
    for met_id in metabolite_ids:
        if met_id in central_carbon_metabolites[namespace]:
            pathway_counts['central_carbon'] += 1
        if met_id in amino_acid_metabolites[namespace]:
            pathway_counts['amino_acid'] += 1
        
        # Estimate pathway association based on functional groups
        for group, patterns in functional_groups[namespace].items():
            for pattern in patterns:
                if pattern in met_id:
                    pathway_counts[group] += 1
                    break
    
    # Calculate pathway distribution
    total = len(metabolite_ids)
    pathway_distribution = {pathway: count/total for pathway, count in pathway_counts.items()}
    
    # Cache the result
    metabolite_pathway_cache[cache_key] = pathway_distribution
    return pathway_distribution

def calculate_pathway_overlap(rxn_mets, model_mets, namespace):
    """
    Calculate pathway overlap between reaction metabolites and model metabolites.
    
    Args:
        rxn_mets: Set of reaction metabolite IDs
        model_mets: Set of model metabolite IDs
        namespace: Namespace for metabolite IDs ('bigg' or 'modelseed')
        
    Returns:
        Overlap score between 0 and 1
    """
    # Get pathway enrichment for reaction and model metabolites
    rxn_pathway = get_metabolite_pathway_enrichment(rxn_mets, namespace)
    model_pathway = get_metabolite_pathway_enrichment(model_mets, namespace)
    
    # Calculate pathway distribution similarity
    overlap_score = 0
    all_pathways = set(rxn_pathway.keys()).union(model_pathway.keys())
    
    if not all_pathways:
        return 0.0
    
    for pathway in all_pathways:
        rxn_value = rxn_pathway.get(pathway, 0)
        model_value = model_pathway.get(pathway, 0)
        # Use minimum value as overlap measure
        overlap_score += min(rxn_value, model_value)
    
    # Normalize score
    return overlap_score / len(all_pathways)

def calculate_reaction_network_features(rxn, model, universe, namespace, dead_ends=None, network_graph=None):
    """
    Calculate network features for a reaction, considering substrate/product structural similarity and pathway overlap.
    
    Args:
        rxn: Reaction to evaluate
        model: Current model
        universe: Universal model
        namespace: Namespace ('bigg' or 'modelseed')
        dead_ends: Set of dead-end metabolites in the current model
        network_graph: Metabolic network graph of the model
        
    Returns:
        Dictionary of features
    """
    if rxn.id not in universe.reactions:
        return {
            'connectivity': 0,
            'substrate_connectivity': 0,
            'product_connectivity': 0,
            'novel_mets': 0,
            'dead_end_resolving': 0,
            'hub_mets': 0,
            'pathway_coherence': 0,
            'structural_similarity': 0,
            'gap_filling': 0
        }
    
    rxn_obj = universe.reactions.get_by_id(rxn.id)
    rxn_mets = {m.id for m in rxn_obj.metabolites}
    model_mets = {m.id for m in model.metabolites}
    
    # Extract reaction substrates and products
    substrates = {m.id for m in rxn_obj.metabolites if rxn_obj.metabolites[m] < 0}
    products = {m.id for m in rxn_obj.metabolites if rxn_obj.metabolites[m] > 0}
    
    # 1. Calculate basic connectivity metrics
    connectivity = len(rxn_mets.intersection(model_mets)) / len(rxn_mets) if rxn_mets else 0
    substrate_connectivity = len(substrates.intersection(model_mets)) / len(substrates) if substrates else 0
    product_connectivity = len(products.intersection(model_mets)) / len(products) if products else 0
    
    # 2. Calculate number of new metabolites
    novel_mets = len(rxn_mets - model_mets)
    
    # 3. Calculate hub metabolite score
    # Improved hub metabolite calculation, considering reaction importance and connectivity
    hub_mets_score = 0
    for m_id in rxn_mets:
        # Get the number of reactions for each metabolite
        if m_id in universe.metabolites:
            met_reaction_count = len(universe.metabolites.get_by_id(m_id).reactions)
            # Weighted calculation: hub metabolite score increases with reaction count
            if met_reaction_count > 5:  # Increase hub metabolite threshold
                # For important hub metabolites, value grows logarithmically with reaction count
                hub_mets_score += min(10, np.log2(met_reaction_count))
    
    # 4. If dead-end metabolites are provided, calculate dead-end resolving ability
    dead_end_resolving = 0
    if dead_ends is not None:
        # Calculate if the reaction can resolve existing dead-end metabolites
        for met_id in rxn_mets:
            if met_id in dead_ends:
                # Differentiate between substrate and product resolving capabilities
                if met_id in substrates and any(prod in model_mets for prod in products):
                    dead_end_resolving += 1  # Dead-end metabolite consumed as substrate
                elif met_id in products and any(sub in model_mets for sub in substrates):
                    dead_end_resolving += 1  # Dead-end metabolite produced as product
    
    # 5. Calculate pathway coherence/overlap
    pathway_coherence = calculate_pathway_overlap(rxn_mets, model_mets, namespace)
    
    # 6. Calculate metabolite structural similarity
    structural_similarity = 0
    if rxn_mets:
        # Calculate structural similarity between reaction metabolites and model metabolites
        sim_scores = []
        for rxn_met in rxn_mets:
            # Find the most similar model metabolite for each reaction metabolite
            max_sim = 0
            for model_met in model_mets:
                sim = calculate_structural_similarity(rxn_met, model_met, namespace)
                max_sim = max(max_sim, sim)
            sim_scores.append(max_sim)
        # Take average structural similarity
        structural_similarity = sum(sim_scores) / len(sim_scores) if sim_scores else 0
    
    # 7. Calculate "gap filling" ability
    gap_filling = 0
    if network_graph is not None:
        # Check if reaction connects disconnected parts of the network
        for sub in substrates:
            for prod in products:
                if sub in model_mets and prod in model_mets:
                    # Calculate distance between two metabolites in the model
                    try:
                        # If two metabolites are far apart or unreachable in the graph, but the reaction connects them directly, it fills a gap
                        distance = calculate_pathway_distance(sub, prod, network_graph)
                        if distance > 3:  # Distance threshold, can be adjusted as needed
                            gap_filling += 1
                    except:
                        gap_filling += 0.5  # Default value if distance can't be calculated
    
    # Return complete feature set
    return {
        'connectivity': connectivity,
        'substrate_connectivity': substrate_connectivity,
        'product_connectivity': product_connectivity,
        'novel_mets': novel_mets,
        'dead_end_resolving': dead_end_resolving,
        'hub_mets': hub_mets_score,
        'pathway_coherence': pathway_coherence,
        'structural_similarity': structural_similarity, 
        'gap_filling': gap_filling
    }

@timed_function
def select_reactions_by_strategy(df_hp, universe, model, max_select, namespace, strategy="advanced"):
    """
    Select reactions according to the specified strategy.
    
    Args:
        df_hp: DataFrame with predicted reaction scores
        universe: Universal model containing all possible reactions
        model: Current model
        max_select: Maximum number of reactions to select
        namespace: Namespace for reaction IDs ('bigg' or 'modelseed')
        strategy: Selection strategy ('advanced' or 'balanced')
        
    Returns:
        List of selected reaction IDs
    """
    logger.info(f"Using {strategy} strategy to select reactions, max selection: {max_select}")
    # Normalize prediction score column name
    if 'predicted_scores' in df_hp.columns:
        pred_score_col = 'predicted_scores'
    elif 'predicted scores' in df_hp.columns:
        pred_score_col = 'predicted scores'
    elif 'mean_score' in df_hp.columns:
        pred_score_col = 'mean_score'
    else:
        potential_score_cols = ['max_score', 'scores', 'value']
        for col in potential_score_cols:
            if col in df_hp.columns:
                pred_score_col = col
                break
        else:
            pred_score_col = df_hp.columns[0]
    if 'similarity_scores' in df_hp.columns:
        sim_col = 'similarity_scores'
    elif 'similarity max' in df_hp.columns:
        sim_col = 'similarity max'
    else:
        sim_col = None
        
    if pred_score_col in df_hp.columns:
        df_filtered = df_hp[df_hp[pred_score_col] >= MIN_PREDICTED_SCORE].copy()
        logger.info(f"After prediction score filtering, remaining candidate reactions: {len(df_filtered)}/{len(df_hp)}")
    else:
        df_filtered = df_hp.copy()
        
    if len(df_filtered) < max_select:
        logger.warning("Not enough reactions after filtering, relaxing filter criteria")
        df_filtered = df_hp.copy()
        
    # Preprocess model for analysis, improving performance
    dead_ends = None
    network_graph = None
    
    if strategy == "advanced":
        # Calculate dead-end metabolites to evaluate reaction "gap filling" capability
        dead_ends = identify_dead_end_metabolites(model)
        logger.info(f"Identified {len(dead_ends)} dead-end metabolites")
        
        # Build model metabolic network graph for connectivity analysis
        model_key = model.id if hasattr(model, 'id') else id(model)
        if model_key not in network_cache:
            network_graph = build_metabolite_reaction_graph(model)
            network_cache[model_key] = network_graph
        else:
            network_graph = network_cache[model_key]
    
    # Calculate features for each candidate reaction
    candidate_features = {}
    model_mets = {m.id for m in model.metabolites}
    
    for rid in df_filtered.index:
        if rid not in universe.reactions:
            continue
            
        rxn = universe.reactions.get_by_id(rid)
        rxn_mets = {m.id for m in rxn.metabolites}
        
        # Exclude oxygen-containing reactions (based on namespace)
        if namespace == "bigg" and 'o2_c' in rxn_mets:
            continue
        if namespace == "modelseed" and 'cpd00007_c0' in rxn_mets:
            continue
        
        # Calculate advanced network features
        if strategy == "advanced":
            net_features = calculate_reaction_network_features(
                rxn, model, universe, namespace, 
                dead_ends=dead_ends, 
                network_graph=network_graph
            )
        else:
            # Use original network feature calculation method
            net_features = {
                'connectivity': len(rxn_mets.intersection(model_mets)) / len(rxn_mets) if rxn_mets else 0,
                'novel_mets': len(rxn_mets - model_mets),
                'hub_mets': sum(1 for m_id in rxn_mets if 
                               sum(1 for r in universe.metabolites.get_by_id(m_id).reactions if r.id != rxn.id) > 5)
            }
        
        # Extract similarity features
        sim_features = {}
        for feature, possible_names in {
            'cosine_max': ['cosine_similarity_max', 'cosine_max'],
            'cosine_mean': ['cosine_similarity_mean', 'cosine_mean'],
            'jaccard': ['jaccard_similarity', 'jaccard'],
            'correlation': ['correlation_similarity', 'correlation']
        }.items():
            for col_name in possible_names:
                if col_name in df_filtered.columns:
                    sim_features[feature] = df_filtered.loc[rid, col_name]
                    break
            else:
                sim_features[feature] = 0.0
                
        if pred_score_col in df_filtered.columns:
            sim_features['pred_score'] = df_filtered.loc[rid, pred_score_col]
        else:
            sim_features['pred_score'] = 0.5
            
        if sim_col and sim_col in df_filtered.columns:
            sim_features['main_sim'] = df_filtered.loc[rid, sim_col]
        else:
            sim_features['main_sim'] = 0.5
            
        candidate_features[rid] = {**sim_features, **net_features}
    
    if not candidate_features:
        logger.warning("No candidate reactions found meeting criteria, falling back to original strategy")
        return df_filtered.index.tolist()[:max_select]
    
    # Create features DataFrame and normalize
    features_df = pd.DataFrame.from_dict(candidate_features, orient='index')
    
    # Normalize features
    for col in features_df.columns:
        if col in ['connectivity', 'substrate_connectivity', 'product_connectivity', 
                   'pred_score', 'hub_mets', 'dead_end_resolving', 'pathway_coherence', 
                   'structural_similarity', 'gap_filling']:
            col_min = features_df[col].min()
            col_max = features_df[col].max()
            if col_max > col_min:
                features_df[col+'_norm'] = (features_df[col] - col_min) / (col_max - col_min)
            else:
                features_df[col+'_norm'] = 0.5
    
    # Special treatment for main similarity score (higher is better)
    if 'main_sim' in features_df.columns:
        col_min = features_df['main_sim'].min()
        col_max = features_df['main_sim'].max()
        if col_max > col_min:
            features_df['main_sim_norm'] = (features_df['main_sim'] - col_min) / (col_max - col_min)
        else:
            features_df['main_sim_norm'] = 0.5
    
    # Calculate comprehensive score (based on strategy)
    if strategy == "advanced":
        # Advanced strategy: balance connectivity, prediction score, dead-end resolving, pathway overlap, and structural similarity
        features_df['advanced_score'] = (
            0.25 * features_df.get('connectivity_norm', 0) + 
            0.15 * features_df.get('substrate_connectivity_norm', 0) + 
            0.15 * features_df.get('product_connectivity_norm', 0) + 
            0.15 * features_df.get('pred_score_norm', 0) + 
            0.10 * features_df.get('dead_end_resolving_norm', 0) + 
            0.10 * features_df.get('hub_mets_norm', 0) +
            0.05 * features_df.get('pathway_coherence_norm', 0) +
            0.05 * features_df.get('structural_similarity_norm', 0) +
            0.05 * features_df.get('gap_filling_norm', 0)
        )
        features_df = features_df.sort_values('advanced_score', ascending=False)
    else:
        # Balanced strategy (original)
        features_df['balanced_score'] = (
            0.3 * features_df.get('connectivity_norm', 0) + 
            0.3 * features_df.get('pred_score_norm', 0) + 
            0.2 * features_df.get('main_sim_norm', 0) + 
            0.2 * features_df.get('hub_mets_norm', 0)
        )
        features_df = features_df.sort_values('balanced_score', ascending=False)
    
    # Select highest scoring reactions
    selected_rxns = features_df.index[:max_select].tolist()
    logger.info(f"Selected {len(selected_rxns)} reactions using {strategy} strategy")
    
    return selected_rxns

@timed_function
def process_reactions_in_order(model, candidate_rxns, universe, namespace):
    """
    Process reactions in order, adding them to the model if they improve it.
    
    Args:
        model: COBRA model to modify
        candidate_rxns: List of candidate reaction IDs
        universe: Universal model containing all possible reactions
        namespace: Namespace for reaction IDs ('bigg' or 'modelseed')
        
    Returns:
        Tuple of (modified model, list of added reaction IDs)
    """
    model_copy = deepcopy(model)
    max_growth = model_copy.slim_optimize()
    added_rxns = []
    counter = 0
    
    # Track resolved dead-end metabolites
    initial_dead_ends = identify_dead_end_metabolites(model_copy)
    resolved_dead_ends = set()
    
    for rid in candidate_rxns:
        if counter >= len(candidate_rxns):
            break
            
        if rid not in universe.reactions:
            continue
            
        assert not rid.startswith('EX_')
        rxn = universe.reactions.get_by_id(rid)
        
        # Exclude oxygen-containing reactions
        if namespace == "bigg":
            if 'o2_c' in [met.id for met in rxn.reactants] or 'o2_c' in [met.id for met in rxn.products]:
                continue
        else:
            if 'cpd00007_c0' in [met.id for met in rxn.reactants] or 'cpd00007_c0' in [met.id for met in rxn.products]:
                continue
        
        # Create temporary model copy for testing
        model_test = deepcopy(model_copy)
        model_test.add_reactions([rxn])
        assert rid in model_test.reactions
        
        # Check growth rate change
        new_growth = model_test.slim_optimize()
        resolved = True
        
        if np.abs(new_growth - max_growth) > 1e-6:
            resolved = resolve_EGC(model_test, rid, namespace)
        
        if resolved == True and model_test.slim_optimize() < MAX_GROWTH_THRESHOLD:
            # Check if adding this reaction resolves dead-end metabolites
            if initial_dead_ends:
                rxn_mets = {m.id for m in rxn.metabolites}
                new_resolved = initial_dead_ends.intersection(rxn_mets)
                resolved_dead_ends.update(new_resolved)
                if new_resolved:
                    logger.debug(f"Reaction {rid} resolved {len(new_resolved)} dead-end metabolites")
            
            # Accept this reaction
            model_copy = deepcopy(model_test)
            counter += 1
            added_rxns.append(rid)
            max_growth = model_copy.slim_optimize()
            logger.debug(f"Added reaction {rid}, current growth rate = {max_growth:.6f}")
    
    if initial_dead_ends:
        logger.info(f"Resolved {len(resolved_dead_ends)}/{len(initial_dead_ends)} dead-end metabolites in total")
    
    logger.info(f"Successfully added {len(added_rxns)}/{len(candidate_rxns)} reactions")
    return model_copy, added_rxns

@timed_function
def optimize_fba_analysis(model, target_ex_rxns):
    """
    Optimize flux balance analysis for the model.
    
    Args:
        model: COBRA model to analyze
        target_ex_rxns: List of target exchange reaction IDs
        
    Returns:
        Dictionary with FBA, pFBA, and FVA results
    """
    try:
        original_objective_reactions = []
        original_objective_coefficients = []
        if hasattr(model, 'objective') and hasattr(model.objective, 'expression'):
            for variable, coefficient in model.objective.expression.as_coefficients_dict().items():
                if hasattr(variable, 'reaction') and variable.reaction is not None:
                    original_objective_reactions.append(variable.reaction.id)
                    original_objective_coefficients.append(coefficient)
                elif hasattr(variable, 'name'):
                    for rxn in model.reactions:
                        if rxn.id in variable.name:
                            original_objective_reactions.append(rxn.id)
                            original_objective_coefficients.append(coefficient)
                            break
        elif hasattr(model, 'objective') and hasattr(model.objective, 'get_linear_coefficients'):
            for rxn, coef in model.objective.get_linear_coefficients(model.reactions).items():
                if coef != 0:
                    original_objective_reactions.append(rxn.id)
                    original_objective_coefficients.append(coef)
        else:
            for rxn in model.reactions:
                if rxn.objective_coefficient != 0:
                    original_objective_reactions.append(rxn.id)
                    original_objective_coefficients.append(rxn.objective_coefficient)
        logger.info(f"Saved {len(original_objective_reactions)} objective function reactions")
    except Exception as e:
        logger.warning(f"Error saving objective function: {str(e)}, using backup method")
        for rxn in model.reactions:
            if hasattr(rxn, "objective_coefficient") and rxn.objective_coefficient > 0:
                original_objective_reactions = [rxn.id]
                original_objective_coefficients = [rxn.objective_coefficient]
                break
    fba_solution = model.optimize()
    if model.solver.status != "optimal":
        is_optimal = False
        for lp_method in ["primal", "dual", "network", "barrier", "sifting", "concurrent"]:
            try:
                model.solver.configuration.lp_method = lp_method
                fba_solution = model.optimize()
                if model.solver.status == "optimal":
                    is_optimal = True
                    break
            except Exception as e:
                logger.warning(f"FBA optimization failed using method {lp_method}: {str(e)}")
                continue
        if not is_optimal:
            logger.error("Could not find a feasible FBA solution")
            return None
    if fba_solution.objective_value <= 0:
        logger.error("FBA solution growth rate is zero or negative")
        return None
    try:
        pfba_solution = cobra.flux_analysis.pfba(model)
    except Exception as e:
        logger.warning(f"pFBA calculation failed: {str(e)}")
        pfba_solution = None
        for lp_method in ["primal", "dual", "network", "barrier", "sifting", "concurrent"]:
            try:
                model.solver.configuration.lp_method = lp_method
                pfba_solution = cobra.flux_analysis.pfba(model)
                break
            except Exception as e:
                continue
        if pfba_solution is None:
            logger.warning("pFBA failed, using regular FBA results")
            pfba_solution = fba_solution
    for ex in model.exchanges:
        if pfba_solution.fluxes[ex.id] >= 0:
            ex.lower_bound = 0
        else:
            ex.lower_bound = pfba_solution.fluxes[ex.id]
    try:
        fva_results = cobra.flux_analysis.flux_variability_analysis(
            model, 
            reaction_list=target_ex_rxns,
            fraction_of_optimum=0.999999,
            loopless=True
        )
    except Exception as e:
        logger.warning(f"FVA with loopless failed: {str(e)}, trying without loopless")
        try:
            fva_results = cobra.flux_analysis.flux_variability_analysis(
                model, 
                reaction_list=target_ex_rxns,
                fraction_of_optimum=0.999999,
                loopless=False
            )
        except Exception as e:
            logger.warning(f"FVA still failed: {str(e)}, reducing fraction_of_optimum")
            try:
                fva_results = cobra.flux_analysis.flux_variability_analysis(
                    model, 
                    reaction_list=target_ex_rxns,
                    fraction_of_optimum=0.95,
                    loopless=False
                )
            except Exception as e:
                logger.error(f"All FVA attempts failed: {str(e)}")
                return None
    try:
        if original_objective_reactions:
            if len(original_objective_reactions) == 1:
                rxn_id = original_objective_reactions[0]
                coef = original_objective_coefficients[0]
                if rxn_id in model.reactions:
                    model.objective = model.reactions.get_by_id(rxn_id)
            else:
                objective_dict = {}
                for i, rxn_id in enumerate(original_objective_reactions):
                    if rxn_id in model.reactions:
                        objective_dict[model.reactions.get_by_id(rxn_id)] = original_objective_coefficients[i]
                if objective_dict:
                    try:
                        model.objective = objective_dict
                    except Exception as e:
                        logger.warning(f"Setting objective function with dictionary failed: {str(e)}, trying one by one")
                        for rxn, coef in objective_dict.items():
                            model.objective = rxn
                            break
                        for rxn, coef in objective_dict.items():
                            if hasattr(rxn, 'objective_coefficient'):
                                rxn.objective_coefficient = coef
        else:
            biomass_reactions = [rxn for rxn in model.reactions if "biomass" in rxn.id.lower()]
            if biomass_reactions:
                model.objective = biomass_reactions[0]
            else:
                model.objective = model.reactions[0]
    except Exception as e:
        logger.warning(f"Error restoring objective function: {str(e)}, continuing execution")
        try:
            for rxn in model.reactions:
                if "biomass" in rxn.id.lower():
                    model.objective = rxn
                    logger.info(f"Set objective function using backup method: {rxn.id}")
                    break
        except:
            pass
    return {
        'fba_solution': fba_solution,
        'pfba_solution': pfba_solution,
        'fva_results': fva_results
    }

def get_fermentation_product_improved(directory, pipeline, method, genome, nselect, universe, strategy="advanced"):
    """
    Improved fermentation product analysis, supporting advanced network evaluation strategies.
    
    Args:
        directory: Base directory name
        pipeline: Modeling pipeline name ('carveme' or 'modelseed')
        method: Method for reaction selection ('GPR_POOL')
        genome: Genome ID to process
        nselect: Number of reactions to select
        universe: Universal model containing all possible reactions
        strategy: Selection strategy ('advanced' or 'balanced')
        
    Returns:
        DataFrame with gap-filling results or None if failed
    """
    try:
        if method != "GPR_POOL":
            raise ValueError(f"Only GPR_POOL method is supported, received: {method}")
        namespace = "bigg" if pipeline == "carveme" else "modelseed"
        model_path = os.path.join(MODELS_DIR, f'{genome}.xml')
        logger.info(f"Loading model: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist: {model_path}")
            return None
        model = cobra.io.read_sbml_model(model_path)
        model.solver = 'cplex'
        model = add_EX_reactions(model, universe, namespace)
        model = constrain_media(directory, genome, model, namespace)
        max_growth = model.slim_optimize()
        if max_growth <= 0:
            logger.error(f"Model {genome} cannot grow under current media conditions")
            return None
        logger.info(f"Model {genome} initial maximum growth rate: {max_growth:.6f}")
        directory_pipeline = directory + '_' + pipeline
        sim_file = os.path.join(SIMILARITY_SCORES_DIR, f'{genome}.csv')
        if not os.path.exists(sim_file):
            alt_paths = [
                f"{directory_pipeline}/{method}/similarity_scores/{genome}.csv",
                f"{directory_pipeline}/{method}/predicted scores/{genome}.csv",
                f"{directory_pipeline}/{method}/predicted_scores/{genome}.csv",
                f"{directory_pipeline}/{method}/scores/{genome}.csv"
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    sim_file = path
                    logger.info(f"Using alternative similarity file: {sim_file}")
                    break
            else:
                logger.error(f"Cannot find similarity scores file: {sim_file}")
                return None
        df_hp = pd.read_csv(sim_file, index_col=0)
        num_rxns_before = len(model.reactions)
        rxn_ids_changed = []
        
        # Calculate initial model evaluation metrics
        initial_dead_ends = identify_dead_end_metabolites(model)
        logger.info(f"Initial model contains {len(initial_dead_ends)} dead-end metabolites")
        
        if method == "GPR_POOL":
            df_hp = df_hp.loc[[idx for idx in df_hp.index if idx not in model.reactions and idx in universe.reactions]]
            candidate_rxns = select_reactions_by_strategy(df_hp, universe, model, nselect, namespace, strategy)
            logger.info(f"Selected {len(candidate_rxns)} candidate reactions using {strategy} strategy")
            model, rxn_ids_changed = process_reactions_in_order(model, candidate_rxns, universe, namespace)
        
        # Calculate modified model metrics
        final_dead_ends = identify_dead_end_metabolites(model)
        dead_ends_resolved = len(initial_dead_ends) - len(final_dead_ends)
        logger.info(f"Model optimization resolved {dead_ends_resolved} dead-end metabolites")
        
        num_rxns_after = len(model.reactions)
        num_rxns_changed = num_rxns_after - num_rxns_before
        logger.info(f"Added {abs(num_rxns_changed)} reactions to model {genome}")
        
        target_exchange_reactions = target_exchange_reactions_bigg if namespace == "bigg" else target_exchange_reactions_modelseed
        analysis_results = optimize_fba_analysis(model, target_exchange_reactions)
        
        if analysis_results is None:
            logger.error(f"FBA/FVA analysis failed for model {genome}")
            return None
            
        fva_results = analysis_results['fva_results']
        fba_solution = analysis_results['fba_solution']
        fva_results.index.name = 'reaction'
        fv = fva_results.reset_index()
        fv['genome'] = genome
        fv['method'] = method
        fv['nselect'] = nselect
        fv['num_rxns_changed'] = num_rxns_changed
        fv['rxn_ids_changed'] = ';'.join(rxn_ids_changed)
        fv['biomass'] = fba_solution.objective_value
        fv['selection_strategy'] = strategy
        fv['dead_ends_resolved'] = dead_ends_resolved
        
        return fv
    except Exception as e:
        logger.error(f"Error processing model {genome}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def parallel_process_genomes(directory, pipeline, method, nselect, universe, genomes, n_jobs=None, batch_size=None, strategy="advanced"):
    """
    Process genomes in parallel.
    
    Args:
        directory: Base directory name
        pipeline: Modeling pipeline name ('carveme' or 'modelseed')
        method: Method for reaction selection ('GPR_POOL')
        nselect: Number of reactions to select
        universe: Universal model containing all possible reactions
        genomes: List of genome IDs to process
        n_jobs: Number of parallel jobs (-1 for all available CPUs)
        batch_size: Size of batches for processing
        strategy: Selection strategy ('advanced' or 'balanced')
        
    Returns:
        DataFrame with combined gap-filling results
    """
    if n_jobs is None:
        n_jobs = min(MAX_WORKERS, len(genomes))
    if batch_size is None:
        batch_size = min(BATCH_SIZE, len(genomes))
    logger.info(f"Starting parallel processing of {len(genomes)} genomes using {n_jobs} worker processes")
    checkpoint_file = f"{directory}_{pipeline}/ferm_prod_{strategy}_{nselect}.csv"
    results = []
    processed_genomes = set()
    if os.path.exists(checkpoint_file):
        try:
            checkpoint_df = pd.read_csv(checkpoint_file)
            processed_genomes = set(checkpoint_df['genome'].unique())
            results.append(checkpoint_df)
            logger.info(f"Restored from checkpoint, {len(processed_genomes)} genomes already processed")
        except Exception as e:
            logger.warning(f"Cannot restore from checkpoint: {str(e)}")
    remaining_genomes = [g for g in genomes if g not in processed_genomes]
    processed_count = 0
    for i in range(0, len(remaining_genomes), batch_size):
        current_batch_size = batch_size if memory_usage_percent() <= MEMORY_THRESHOLD else max(1, batch_size // 2)
        batch_genomes = remaining_genomes[i:i+current_batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}, containing {len(batch_genomes)} genomes")
        batch_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(get_fermentation_product_improved)(
                directory, pipeline, method, genome, nselect, universe, strategy
            ) 
            for genome in batch_genomes
        )
        valid_results = [r for r in batch_results if r is not None]
        processed_count += len(valid_results)
        if valid_results:
            batch_df = pd.concat(valid_results)
            results.append(batch_df)
            all_results_df = pd.concat(results)
            all_results_df.to_csv(checkpoint_file, index=False)
            logger.info(f"Created checkpoint, currently successfully processed {processed_count}/{len(genomes)} genomes")
        gc.collect()
    if results:
        final_results = pd.concat(results)
        return final_results
    else:
        logger.error("Failed to process any genomes successfully")
        return None

def main_improved():
    """
    Main function for improved metabolic network reconstruction and fermentation product analysis.
    """
    parallel = True
    pipeline = "carveme"
    directory = "Zimmermann2021GenBiol"
    directory_pipeline = directory + "_" + pipeline
    n_jobs = -1
    os.makedirs(os.path.join(BASE_DIR, directory_pipeline), exist_ok=True)
    log_file = os.path.join(BASE_DIR, f"{directory_pipeline}/reconstruction_{time.strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("=== Starting metabolic network reconstruction and fermentation product analysis ===")
    logger.info(f"Pipeline: {pipeline}, Directory: {directory}")
    try:
        # Read metadata using relative paths
        df_genome = pd.read_csv(os.path.join(METADATA_DIR, 'organisms2.csv'), sep='\t', index_col=0)
        df_genome = df_genome[df_genome.tax=='Bacteria']
        available_genomes = []
        
        # Process similarity scores directory with relative path
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
            return
        
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
                return
        
        logger.info(f"Loading universal reaction library: {universe_file}")
        with suppress_stderr_and_warnings():
            universe = cobra.io.read_sbml_model(universe_file)
            # Resolve objective function issues
            if len(universe.reactions) > 0:
                # Randomly select a reaction as the objective
                default_obj_reaction = universe.reactions[0]
                default_obj_reaction.objective_coefficient = 1.0
                universe.objective = default_obj_reaction
        logger.info(f"Successfully loaded universal reaction library with {len(universe.reactions)} reactions")
        
        # Use new advanced strategy to improve metabolic network reconstruction quality
        nselect_values = [100, 200]
        for nselect in nselect_values:
            for method in ['GPR_POOL']:
                result_file = os.path.join(BASE_DIR, f"{directory_pipeline}/ferm_prod_advanced_{nselect}.csv")
                if os.path.exists(result_file):
                    logger.info(f"Skipping already processed configuration: Method {method}, Selection {nselect}")
                    continue
                logger.info(f"Processing configuration: Method {method}, Selection {nselect}")
                if parallel:
                    retLst = parallel_process_genomes(
                        directory, pipeline, method, nselect, universe, list(df_genome.index), n_jobs=n_jobs, strategy="advanced"
                    )
                    if retLst is not None:
                        retLst.to_csv(result_file, index=False)
                        logger.info(f"Processing completed successfully, results saved to {result_file}")
                else:
                    results = []
                    for genome in df_genome.index:
                        logger.info(f"Processing genome: {genome}")
                        ret = get_fermentation_product_improved(directory, pipeline, method, genome, nselect, universe, strategy="advanced")
                        if ret is not None:
                            results.append(ret)
                        if results:
                            pd.concat(results).to_csv(result_file, index=False)
                    if results:
                        pd.concat(results).to_csv(result_file, index=False)
                        logger.info(f"Processing completed successfully, results saved to {result_file}")
                    else:
                        logger.error("Processing completed, but no valid results")
    except Exception as e:
        logger.error(f"Main program execution error: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main_improved()