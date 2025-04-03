import os
import sys
import torch as t
import pandas as pd
import cobra
import math
import random
from cobra.util.array import create_stoichiometric_matrix
import numpy as np

from rdkit import Chem
import deepchem as dc

from process_data import get_coefficient_and_reactant, create_neg_rxn
import torch
from rdkit import Chem

import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"


# Suppress C++ warnings temporarily
def suppress_cpp_warnings():
    stderr_fileno = sys.stderr.fileno()
    original_stderr = os.dup(stderr_fileno)

    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fileno)

    return original_stderr, devnull


# Restore stderr after suppressing warnings
def restore_stderr(original_stderr, devnull):
    os.dup2(original_stderr, sys.stderr.fileno())
    os.close(devnull)
    os.close(original_stderr)


# Create negative reactions based on positive reactions and metadata
def create_neg_rxns(args):
    with open(f"./data/{args.train}/{args.train}_rxn_name_list.txt", "r") as f:
        pos_rxn = [i.strip().replace("\n", "") for i in f.readlines()]
    pos_index, pos_metas, pos_nums, rxn_directions = get_coefficient_and_reactant(
        pos_rxn
    )
    pos_metas_smiles = pd.read_csv(f"./data/{args.train}/{args.train}_meta_count.csv")
    chebi_meta_filter = pd.read_csv("./data/pool/cleaned_chebi.csv")
    name_to_smiles = pd.concat(
        [
            chebi_meta_filter.loc[:, ["name", "smiles"]],
            pos_metas_smiles.loc[:, ["name", "smiles"]],
        ]
    )

    # Generate negative reactions
    neg_rxn = create_neg_rxn(
        pos_rxn,
        pos_metas_smiles,
        chebi_meta_filter,
        args.balanced,
        args.negative_ratio,
        args.atom_ratio,
    )
    neg_index, neg_metas, neg_nums, rxn_directions = get_coefficient_and_reactant(
        neg_rxn
    )
    all_metas = list(set(sum(pos_metas, []) + sum(neg_metas, [])))
    all_metas.sort()

    # Create stoichiometric matrices for positive and negative reactions
    pos_matrix = np.zeros((len(all_metas), len(pos_rxn)))
    rxn_df = pd.DataFrame(
        pos_matrix,
        index=all_metas,
        columns=["p_" + str(i) for i in range(len(pos_rxn))],
    )
    reaction_smiles = []
    for i in range(len(pos_index)):
        reactants = []
        products = []
        for j in range(len(pos_metas[i])):
            rxn_df.loc[pos_metas[i][j], "p_" + str(i)] = float(pos_index[i][j])
            if j < pos_nums[i]:
                reactants.append(
                    name_to_smiles[
                        pos_metas[i][j] == name_to_smiles["name"]
                    ].smiles.values[0]
                )
            else:
                products.append(
                    name_to_smiles[
                        pos_metas[i][j] == name_to_smiles["name"]
                    ].smiles.values[0]
                )
        direction = rxn_directions[i]
        smiles = "+".join(reactants) + direction + "+".join(products)
        reaction_smiles.append(smiles)

    neg_matrix = np.zeros((len(all_metas), len(neg_rxn)))
    neg_df = pd.DataFrame(
        neg_matrix,
        index=all_metas,
        columns=["n_" + str(i) for i in range(len(neg_rxn))],
    )
    for i in range(len(neg_index)):
        for j in range(len(neg_metas[i])):
            neg_df.loc[neg_metas[i][j], "n_" + str(i)] = float(neg_index[i][j])
    label2rxn_df = pd.DataFrame(
        {
            "label": rxn_df.columns.to_list() + neg_df.columns.to_list(),
            "rxn": pos_rxn + neg_rxn,
        }
    )

    return rxn_df, neg_df, name_to_smiles, label2rxn_df, reaction_smiles


# Set random seed for reproducibility
def set_random_seed(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError(
            "Seed must be a non-negative integer or omitted, not {}".format(seed)
        )
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    return seed


# Get sorted filenames from a directory
def get_filenames(path):
    return sorted(os.listdir(path))


# Reduce and fold data into a target dimension
def fold_and_reduce(data, target_dim=64):
    results = []
    for row in data:
        result = reduce_item(row, target_dim)
        results.append(result)
    return np.array(results)


# Calculate graph features from adjacency list and feature matrix
def calculate_graph_feat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype="float32")
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index, adj_mat]


# Extract molecular features from input data
def molecular_feature_extract(mol_data):
    mol_data = pd.DataFrame(mol_data).T
    mol_feat = [[] for _ in range(len(mol_data))]
    for i in range(len(mol_feat)):
        feat_mat, adj_list = mol_data.iloc[i]
        mol_feat[i] = calculate_graph_feat(feat_mat, adj_list)
    return mol_feat


# Reduce a row to a fixed target dimension using XOR
def reduce_item(row, target_dim=64):
    current_length = len(row)
    needed_length = ((current_length - 1) // target_dim + 1) * target_dim
    if current_length < needed_length:
        row = np.pad(row, (0, needed_length - current_length), mode="constant")
    row = row.astype(np.int64)
    reshaped_row = row.reshape(-1, target_dim)
    result = np.bitwise_xor.reduce(reshaped_row, axis=0)
    return result


# Featurize SMILES strings into molecular graph features
def featurize_smiles(smiles):
    featurizer = dc.feat.ConvMolFeaturizer()
    features = pd.DataFrame()
    original_stderr, devnull = suppress_cpp_warnings()
    try:
        for i, smile_item in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile_item)
            mol_f = featurizer.featurize(mol)
            if mol_f:
                atom_feat = mol_f[0].get_atom_features()
                adjacency = mol_f[0].get_adjacency_list()
                features[str(i)] = [atom_feat, adjacency]
    finally:
        restore_stderr(original_stderr, devnull)
    return features


# Remove reactions from a metabolic model based on a name list
def remove_rxn(model, name_list):
    remove_list = []
    for i in range(len(model.metabolites)):
        meta = model.metabolites[i]
        if meta.name in name_list:
            continue
        remove_list.append(meta)
    model.remove_metabolites(remove_list, destructive=True)
    print(f"remove_rxn:{len(remove_list)}")


# Load a metabolic model from a file
def get_data(path, sample):
    model = cobra.io.read_sbml_model(path + "/" + sample)
    return model


# Create a pool of metabolic models by merging multiple models
def create_pool():
    path = "data/gems/xml-file"
    namelist = get_filenames(path)
    model_pool = cobra.io.read_sbml_model("./data/pool/universe.xml")
    pool_df = create_stoichiometric_matrix(model_pool, array_type="DataFrame")
    for sample in namelist:
        if sample.endswith("xml"):
            model = get_data(path, sample)
            model_pool.merge(model)
    cobra.io.write_sbml_model(model_pool, "./results/bigg/comb_universe-fliter.xml")
    print(
        f"create pool done! The total number of metabolites is {len(model_pool.metabolites)}"
    )


# Extract data from a pool of metabolic models
def get_data_from_pool(path, sample, model_pool_df):
    if os.path.exists(path + "/reactions_w_gene_reaction_rule.csv"):
        rxns_df = pd.read_csv(path + "/reactions_w_gene_reaction_rule.csv")
        rxns = rxns_df.reaction[rxns_df.id == sample[:-4]].to_numpy()
    else:
        model = get_data(path, sample)
        rxns = np.array([rxn.id for rxn in model.reactions])
    model_df = model_pool_df[rxns]
    cols2use = model_pool_df.columns.difference(model_df.columns)
    return model_df, model_pool_df[cols2use]


# Create a negative incidence matrix for graph-based models
def create_neg_incidence_matrix(incidence_matrix):
    incidence_matrix_neg = t.zeros(incidence_matrix.shape)
    for i, edge in enumerate(incidence_matrix.T):
        nodes = t.where(edge)[0]
        nodes_comp = t.tensor(
            list(set(range(len(incidence_matrix))) - set(nodes.tolist()))
        )
        edge_neg_l = t.tensor(
            np.random.choice(nodes, math.floor(len(nodes) * 0.5), replace=False)
        )
        edge_neg_r = t.tensor(
            np.random.choice(
                nodes_comp, len(nodes) - math.floor(len(nodes) * 0.5), replace=False
            )
        )
        edge_neg = t.cat((edge_neg_l, edge_neg_r))
        incidence_matrix_neg[edge_neg, i] = 1
    return incidence_matrix_neg


# Compute loss for hyperlink scoring
def hyperlink_score_loss(y_pred, y):
    negative_score = t.mean(y_pred[y == 0])
    logistic_loss = t.log(1 + t.exp(negative_score - y_pred[y == 1]))
    loss = t.mean(logistic_loss)
    return loss


# Create labels for positive and negative incidence matrices
def create_label(incidence_matrix_pos, incidence_matrix_neg):
    y_pos = t.ones(len(incidence_matrix_pos.T))
    y_neg = t.zeros(len(incidence_matrix_neg.T))
    return t.cat((y_pos, y_neg))


# Compute Gaussian Interaction Profile (GIP) kernel
def getGipKernel(y, trans, gamma):
    if trans:
        y = y.T
    krnl = t.mm(y, y.T)
    krnl = krnl / t.mean(t.diag(krnl))
    krnl = t.exp(-kernelToDistance(krnl) * gamma)
    return krnl


# Convert kernel matrix to distance matrix
def kernelToDistance(k):
    di = t.diag(k).T
    d = (
        di.repeat(len(k)).reshape(len(k), len(k)).T
        + di.repeat(len(k)).reshape(len(k), len(k))
        - 2 * k
    )
    return d


# Compute cosine similarity kernel
def cosine_kernel(tensor_1, tensor_2):
    return t.DoubleTensor(
        [
            t.cosine_similarity(tensor_1[i], tensor_2, dim=-1).tolist()
            for i in range(tensor_1.shape[0])
        ]
    )


# Normalize tensor values to [0, 1] range using CUDA
def min_max_normalize_cuda(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


# Offset tensor values to make all values non-negative
def min_max_offset_cuda(tensor):
    min_val = torch.min(tensor)
    return tensor + min_val


# Monitor gradients during training to detect vanishing gradients
def monitor_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm()
            if grad_norm < 1e-6:
                print(f"Warning: Gradient vanishing detected in {name} layer")


import re

pattern = re.compile(r"(?<!\d)\s*s_\d+\[\w+\]")


# Extract reactants, products, and direction from a reaction string
def extract_chemicals(reaction):
    direction = "=>"
    if "<=>" in reaction:
        direction = "<=>"
    reactants, products = reaction.split(direction)
    reactant_names = pattern.findall(reactants)
    product_names = pattern.findall(products)
    reactant_names = [re.sub(r"\[\w+\]", "", name.strip()) for name in reactant_names]
    product_names = [re.sub(r"\[\w+\]", "", name.strip()) for name in product_names]
    return reactant_names, product_names, direction


import requests
from xml.etree import ElementTree as ET


# Convert MOL data to SMILES format
def convert_mol_to_smiles(mol_data):
    try:
        mol = Chem.MolFromMolBlock(mol_data)
        if mol:
            smiles = Chem.MolToSmiles(mol)
            return smiles
    except Exception as e:
        print(f"Error converting mol to SMILES: {e}")
    return None


# Remove XML namespaces for easier parsing
def remove_namespace(tree):
    for elem in tree.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
    return tree


# Fetch SMILES string from ChEBI database using ChEBI ID
def get_smiles_from_chebi(chebi_id):
    url = f"https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity?chebiId={chebi_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            tree = ET.ElementTree(ET.fromstring(response.content))
            tree = remove_namespace(tree)
            root = tree.getroot()
            smiles = None
            smiles_element = root.find(".//smiles")
            if smiles_element is not None:
                smiles = smiles_element.text
            if not smiles:
                mol_data = None
                for structure in root.findall(".//ChemicalStructures"):
                    structure_type = structure.find("type").text
                    default_structure = structure.find("defaultStructure").text
                    if structure_type == "mol" and default_structure == "true":
                        mol_data = structure.find("structure").text
                        break
                if mol_data:
                    smiles = convert_mol_to_smiles(mol_data)
            return smiles
    except ET.ParseError as parse_err:
        print(f"Error parsing XML for ChEBI ID {chebi_id}: {parse_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error while fetching data for ChEBI ID {chebi_id}: {req_err}")
    return None


# Fetch SMILES string from KEGG database using KEGG ID
def get_smiles_from_kegg(kegg_id):
    url = f"http://rest.kegg.jp/get/{kegg_id}/mol"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            molfile = response.text
            from rdkit import Chem

            mol = Chem.MolFromMolBlock(molfile)
            if mol:
                smiles = Chem.MolToSmiles(mol)
                return smiles
    except Exception as e:
        print(f"Error fetching SMILES from KEGG for ID {kegg_id}: {e}")
    return None
