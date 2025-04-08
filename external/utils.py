import os
import torch
import pandas as pd
import cobra
import math
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients
import numpy as np

############### Remove reactions without SMILES ################
def remove_rxn(model, name_list):
    """
    Remove reactions that don't have SMILES representations.
    
    Args:
        model: COBRA model
        name_list: List of metabolite names to keep
        
    Returns:
        None, modifies model in-place
    """
    remove_list = []
    for i in range(len(model.metabolites)):
        meta = model.metabolites[i]
        if meta.name in name_list:
            continue
        remove_list.append(meta)
    model.remove_metabolites(remove_list, destructive=True)
    print(f'remove_rxn:{len(remove_list)}')

def get_filenames(path):
    """
    Get sorted list of filenames in a directory.
    
    Args:
        path: Directory path
        
    Returns:
        Sorted list of filenames
    """
    return sorted(os.listdir(path))

def get_data(path, sample):
    """
    Load and preprocess a metabolic model.
    
    Args:
        path: Directory path
        sample: Sample filename
        
    Returns:
        Preprocessed COBRA model
    """
    model = cobra.io.read_sbml_model(path + '/' + sample)
    
    biomass = linear_reaction_coefficients(model)
    model.remove_reactions(biomass, remove_orphans=True)
    stoichiometric_matrix = create_stoichiometric_matrix(model)
    incidence_matrix = np.abs(stoichiometric_matrix) > 0
    remove_rxn_index = np.sum(incidence_matrix, axis=0) <= 1
    indices_to_remove = np.where(remove_rxn_index)[0]
    model.remove_reactions([model.reactions[i] for i in indices_to_remove], remove_orphans=True)
    return model

def create_pool():
    """
    Merge genome-scale metabolic models (GEMs) with reaction pool.
    
    Creates a combined universe of reactions from all available models
    and a reference reaction pool.
    """
    print('-------------------------------------------------------')
    print('Merging GEMs with reaction pool...')
    path = '../data/gems/draft_gems'
    # path = './test'
    namelist = get_filenames(path)
    model_pool = cobra.io.read_sbml_model('../data/pool/universe_with_smiles.xml')
    for sample in namelist:
        if sample.endswith('xml'):
            model = get_data(path, sample)
            model_pool.merge(model)
    cobra.io.write_sbml_model(model_pool, './results/universe/comb_universe.xml')
    print('Done with merging reaction pool!')


def get_data_from_pool(path, sample, model_pool_df):
    """
    Extract model data and potential reaction pool for a specific sample.
    
    Args:
        path: Path to model files
        sample: Sample filename
        model_pool_df: Combined pool dataframe
        
    Returns:
        Two dataframes: model reactions and candidate reactions
    """
    if os.path.exists(path + '/reactions_w_gene_reaction_rule.csv'):
        rxns_df = pd.read_csv(path + '/reactions_w_gene_reaction_rule.csv')
        rxns = rxns_df.reaction[rxns_df.id == sample[:-4]].to_numpy()
    else:
        model = get_data(path, sample)
        rxns = np.array([rxn.id for rxn in model.reactions])
    model_df = model_pool_df[rxns]
    cols2use = model_pool_df.columns.difference(model_df.columns)
    return model_df, model_pool_df[cols2use]


def create_neg_incidence_matrix(incidence_matrix):
    """
    Create negative samples for the incidence matrix.
    
    Args:
        incidence_matrix: Positive incidence matrix
        
    Returns:
        Negative incidence matrix
    """
    incidence_matrix_neg = torch.zeros(incidence_matrix.shape)
    for i, edge in enumerate(incidence_matrix.T):
        nodes = torch.where(edge)[0]
        nodes_comp = torch.tensor(list(set(range(len(incidence_matrix))) - set(nodes.tolist())))
        edge_neg_l = torch.tensor(np.random.choice(nodes, math.floor(len(nodes) * 0.5), replace=False))
        edge_neg_r = torch.tensor(np.random.choice(nodes_comp, len(nodes) - math.floor(len(nodes) * 0.5), replace=False))
        edge_neg = torch.cat((edge_neg_l, edge_neg_r))
        incidence_matrix_neg[edge_neg, i] = 1
    return incidence_matrix_neg


def hyperlink_score_loss(y_pred, y):
    """
    Calculate hyperlink score loss for training.
    
    Args:
        y_pred: Predicted scores
        y: True labels
        
    Returns:
        Loss value
    """
    negative_score = torch.mean(y_pred[y == 0])
    logistic_loss = torch.log(1 + torch.exp(negative_score - y_pred[y == 1]))
    loss = torch.mean(logistic_loss)
    return loss


def create_label(incidence_matrix_pos, incidence_matrix_neg):
    """
    Create labels for positive and negative incidence matrices.
    
    Args:
        incidence_matrix_pos: Positive incidence matrix
        incidence_matrix_neg: Negative incidence matrix
        
    Returns:
        Combined labels tensor
    """
    y_pos = torch.ones(len(incidence_matrix_pos.T))
    y_neg = torch.zeros(len(incidence_matrix_neg.T))
    return torch.cat((y_pos, y_neg))