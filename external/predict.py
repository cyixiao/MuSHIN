from utils import *
import torch
import pandas as pd
import cobra
import sys
import argparse

sys.path.append('../')

from MuSHIN import MuSHIN  # Import MuSHIN network as replacement for CLOSEgaps
from algorithms.smiles2vec import smiles_to_chemberta_vector, smiles_to_chemberta_vector_gpu

device = 'cuda' if torch.cuda.is_available() else "cpu"


def train(feature, y, incidence_matrix, model, optimizer, loss_fun):
    """
    Train the MuSHIN network model.
    
    Args:
        feature: Input features
        y: Target labels
        incidence_matrix: Incidence matrix for the reactions
        model: The neural network model
        optimizer: Optimizer for model training
        loss_fun: Loss function
    """
    model.train()
    optimizer.zero_grad()
    y_pred = model(feature, incidence_matrix)
    loss = loss_fun(y_pred, y)
    print(loss.item())
    loss.backward()
    optimizer.step()


def test(feature, incidence_matrix, model):
    """
    Test the model on the given data.
    
    Args:
        feature: Input features
        incidence_matrix: Incidence matrix for the reactions
        model: The trained neural network model
        
    Returns:
        Predicted scores for the reactions
    """
    model.eval()
    epoch_size = incidence_matrix.shape[1] // 10
    iters = 10 if epoch_size * 10 == incidence_matrix.shape[1] else 11
    y_pred_list = []
    with torch.no_grad():
        for itern in range(iters):
            y_pred = model.predict(feature,
                                   incidence_matrix[:, itern * epoch_size:(itern + 1) * epoch_size])
            y_pred_list.append(y_pred)
    y_pred_list = torch.cat(y_pred_list)
    return torch.squeeze(y_pred_list)


def parse():
    """Parse command line arguments for the neural network parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--conv_dim', type=int, default=128)
    parser.add_argument('--L', type=int, default=2)
    parser.add_argument('--head', type=int, default=6)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--g_lambda', type=float, default=1)
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    return parser.parse_args()

def generate_reaction_expression(reaction, smiles_dict):
    """
    Generate reaction expression based on reaction and SMILES dictionary.
    
    Args:
        reaction: cobra.Reaction object
        smiles_dict: Dictionary containing metabolite ID and SMILES
        
    Returns:
        SMILES expression string for the reaction
    """
    substrates = []
    products = []

    for metabolite, coeff in reaction.metabolites.items():
        smiles = smiles_dict.get(metabolite.id, None)
        if smiles:
            if coeff < 0:  # Substrate
                substrates.append(smiles)
            elif coeff > 0:  # Product
                products.append(smiles)
    
    # Build reaction expression
    substrates_expression = " + ".join(substrates)
    products_expression = " + ".join(products)
    reaction_expression = f"{substrates_expression} -> {products_expression}"
    return reaction_expression

def predict():
    """
    Main function to predict reaction scores using the MuSHIN network.
    
    This function loads genome-scale metabolic models, extracts features,
    trains the MuSHIN network, and predicts reaction scores for potential
    reactions to add to the model.
    """
    print('-------------------------------------------------------')
    path = '../data/gems/draft_gems'
    namelist = get_filenames(path)
    args = parse()
    # Read reaction pool
    bigg_pool = cobra.io.read_sbml_model('../data/pool/universe_with_smiles.xml')
    model_pool = cobra.io.read_sbml_model('./results/universe/comb_universe.xml')
    # Extract SMILES information
    bigg_smiles_dict = {meta.id: meta.notes['smiles'] for meta in bigg_pool.metabolites}

    # Change xml file to dataframe
    model_pool_df = create_stoichiometric_matrix(model_pool, array_type='DataFrame')
    for sample in namelist:
        if sample.endswith('.xml'):
            print('Training MuSHIN and predicting reaction scores: ' + sample[:-4] + '...')
            # Read the model and reaction pool
            rxn_df, rxn_pool_df = get_data_from_pool(path, sample, model_pool_df)
            incidence_matrix_pos = np.abs(rxn_df.to_numpy()) > 0
            incidence_matrix_pos = torch.tensor(incidence_matrix_pos, dtype=torch.float)
            incidence_matrix_pos = torch.unique(incidence_matrix_pos, dim=1)
            incidence_matrix_cand = np.abs(rxn_pool_df.to_numpy()) > 0
            incidence_matrix_cand = torch.tensor(incidence_matrix_cand, dtype=torch.float).to(device)

            # Generate reaction SMILES expressions for feature extraction
            reaction_smiles_list = []
            for reaction in model_pool.reactions:
                reaction_smiles = generate_reaction_expression(reaction, bigg_smiles_dict)
                reaction_smiles_list.append({
                    "reaction_id": reaction.id,
                    "smiles_expression": reaction_smiles
                })
            reaction_feature_t = smiles_to_chemberta_vector_gpu(smiles=reaction_smiles, batch_size=32).to(device)

            # Extract metabolite IDs and corresponding SMILES
            metabolite_ids = rxn_df.index
            X_smiles = [bigg_smiles_dict.get(id, None) for id in metabolite_ids]
            X_smiles = [smiles for smiles in X_smiles if smiles is not None]  # Remove None values
            extra_feature_t = smiles_to_chemberta_vector_gpu(smiles=X_smiles, batch_size=32).to(device)

            score = torch.empty((incidence_matrix_cand.shape[1], args.num_iter))
            for i in range(args.num_iter):
                # Create negative reactions
                incidence_matrix_neg = create_neg_incidence_matrix(incidence_matrix_pos)
                incidence_matrix_neg = torch.unique(incidence_matrix_neg, dim=1)
                incidence_matrix = torch.cat((incidence_matrix_pos, incidence_matrix_neg), dim=1).to(device)
                y = create_label(incidence_matrix_pos, incidence_matrix_neg)
                y = torch.tensor(y, dtype=torch.long, device=device)
                incidence_matrix_pos = incidence_matrix_pos.to(device)
                node_num, hyper_num = incidence_matrix.shape
                model = MuSHIN(input_num=node_num, 
                               input_feature_num=incidence_matrix_pos.shape[1],
                               extra_feature=extra_feature_t,
                               reaction_feature=reaction_feature_t,
                               emb_dim=64, conv_dim=64,
                               head=6, p=0.1, L=2,
                               use_attention=True).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                crossentropyloss = torch.nn.CrossEntropyLoss()
                print(' --------- start training --------------------')
                for _ in range(args.max_epoch):
                    # Training
                    model.train()
                    epoch_size = incidence_matrix.shape[1] // 10
                    l = 0
                    for itern in range(10):
                        optimizer.zero_grad()
                        y_pred = model(incidence_matrix_pos,
                                       incidence_matrix[:, itern * epoch_size:(itern + 1) * epoch_size])
                        loss = crossentropyloss(y_pred, y[itern * epoch_size:(itern + 1) * epoch_size])
                        loss.backward()
                        optimizer.step()
                        l += loss.item()
                    l = l / 10
                    print(f'epoch:{_}, loss:{l}')

                score[:, i] = test(incidence_matrix_pos, incidence_matrix_cand, model)[:, 1]
            score_df = pd.DataFrame(data=score.detach().numpy(), index=rxn_pool_df.columns)
            bigg_rxns = set([item.id for item in bigg_pool.reactions])
            common_rxns = list(bigg_rxns & set(score_df.index))
            common_score_df = score_df.T[common_rxns].T
            common_score_df.to_csv('./results/fba_result/' + sample[:-4] + '.csv')
    print('Done with prediction!')


if __name__ == '__main__':
    predict()