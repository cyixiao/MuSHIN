import os
import torch
import torch.nn as nn
import config
import pandas as pd
import numpy as np
import copy
from MuSHIN import MuSHIN
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
)

from utils import set_random_seed, create_neg_rxns, getGipKernel
from algorithms.smiles2vec import smiles_to_chemberta_vector_gpu
from algorithms.smiles2fp import smiles_to_fingerprint

import json

device = "cuda" if torch.cuda.is_available() else "cpu"


# Function to evaluate the model on test data
def test_pre(feature, incidence_matrix, model):
    model.eval()
    with torch.no_grad():
        y_pred = model.predict(feature, incidence_matrix)
    return torch.squeeze(y_pred)


# Function to train the model with the given data and hyperparameters
def train(
    args,
    X_smiles,
    reaction_smiles,
    train_incidence_pos,
    incidence_train,
    incidence_valid,
    y_train,
    y_valid,
):
    node_num, _ = incidence_train.shape
    reaction_feature_t = None

    # Generate features based on the selected algorithm
    if args.algorithm == "similarity":
        X_smiles_t = smiles_to_fingerprint(smiles=X_smiles, radius=2, nBits=1024)
        extra_feature_t = getGipKernel(X_smiles_t, False, args.g_lambda).to(device)

    if args.algorithm == "smiles2vec":
        extra_feature_t = smiles_to_chemberta_vector_gpu(
            smiles=X_smiles, batch_size=args.s2m_batch_size
        ).to(device)

    # Generate reaction fingerprints if enabled
    if args.enable_reaction_fp:
        reaction_feature_t = smiles_to_chemberta_vector_gpu(
            smiles=reaction_smiles, batch_size=args.s2m_batch_size
        ).to(device)

    # Initialize the model and optimizer
    model = MuSHIN(
        input_num=node_num,
        input_feature_num=train_incidence_pos.shape[1],
        extra_feature=extra_feature_t,
        reaction_feature=reaction_feature_t,
        emb_dim=args.emb_dim,
        conv_dim=args.conv_dim,
        head=args.head,
        p=args.p,
        L=args.L,
        use_attention=True,
        enable_hygnn=args.enable_hygnn,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    crossentropyloss = nn.CrossEntropyLoss()
    max_valid_f1 = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Training loop
    for _ in tqdm(range(args.epoch)):
        model.train()
        epoch_size = incidence_train.shape[1] // args.batch_size
        for e in range(epoch_size):
            optimizer.zero_grad()
            y_pred = model(
                train_incidence_pos,
                incidence_train[:, e * args.batch_size : (e + 1) * args.batch_size],
            )
            loss = crossentropyloss(
                y_pred, y_train[e * args.batch_size : (e + 1) * args.batch_size]
            )
            loss.backward()
            optimizer.step()

        # Validate the model and track the best weights
        valid_score = test_pre(train_incidence_pos, incidence_valid, model)
        true_valid_score = valid_score.cpu().numpy()[:, 1]
        b_score = [int(s >= 0.5) for s in true_valid_score]
        auc_score = roc_auc_score(y_valid, true_valid_score)
        pr = precision_score(y_valid, b_score, zero_division=0)
        re = recall_score(y_valid, b_score)
        f1 = f1_score(y_valid, b_score)
        aupr = average_precision_score(y_valid, true_valid_score)
        if max_valid_f1 < f1:
            max_valid_f1 = f1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(
                f"\nvalid, epoch:{_}, f1:{f1},pr:{pr},recall:{re},auc:{auc_score},aupr:{aupr}"
            )
    model.load_state_dict(best_model_wts)
    return model


# Function to predict test results using the trained model
def predict(model, train_incidence_pos, incidence_test, y_test):
    y_pred = test_pre(train_incidence_pos, incidence_test, model)
    score_t = torch.squeeze(y_pred)

    true_test_score = score_t.cpu().numpy()[:, 1]
    ground_truth = y_test.detach().cpu().numpy()
    return true_test_score, ground_truth


if __name__ == "__main__":
    # Parse arguments and set up the environment
    args = config.parse()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    set_random_seed(args.seed)
    print(f"the seed is {args.seed}")
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    testing_results = []
    recover_results = []

    # Main loop for multiple iterations
    for i in range(args.iteration):
        reaction_smiles = []
        if args.create_negative:
            # Create negative samples and split data into train, validation, and test sets
            rxn_df, neg_df, name_to_smiles, label2rxn_df, reaction_smiles = (
                create_neg_rxns(args)
            )
            rxn_df[rxn_df != 0] = 1
            neg_df[neg_df != 0] = 1

            train_split = args.train_split if args.recover else 0.6
            valid_split = train_split + 0.1 if args.recover else 0.8

            train_pos_df, valid_pos_df, test_pos_df = np.split(
                rxn_df.sample(frac=1, axis=1, random_state=args.seed),
                [int(train_split * len(rxn_df.T)), int(valid_split * len(rxn_df.T))],
                axis=1,
            )
            train_neg_df, valid_neg_df, test_neg_df = np.split(
                neg_df.sample(frac=1, axis=1, random_state=args.seed),
                [int(train_split * len(neg_df.T)), int(valid_split * len(neg_df.T))],
                axis=1,
            )

            train_df = pd.concat([train_pos_df, train_neg_df], axis=1).sample(
                frac=1, axis=1
            )
            valid_df = pd.concat([valid_pos_df, valid_neg_df], axis=1).sample(
                frac=1, axis=1
            )

            if args.recover:
                test_df = test_pos_df
            else:
                test_df = pd.concat([test_pos_df, test_neg_df], axis=1).sample(
                    frac=1, axis=1
                )

        # Prepare data for training and testing
        y_train = (
            torch.tensor(["p" in c for c in train_df.columns], dtype=torch.long)
            .view(-1)
            .to(device)
        )
        y_test = torch.tensor(
            ["p" in c for c in test_df.columns], dtype=torch.float
        ).view(-1)
        y_valid = torch.tensor(
            ["p" in c for c in valid_df.columns], dtype=torch.float
        ).view(-1)

        train_incidence_pos = torch.tensor(
            train_pos_df.to_numpy(), dtype=torch.float
        ).to(device)
        incidence_train = torch.tensor(train_df.to_numpy(), dtype=torch.float).to(
            device
        )
        incidence_test = torch.tensor(test_df.to_numpy(), dtype=torch.float).to(device)
        incidence_valid = torch.tensor(valid_df.to_numpy(), dtype=torch.float).to(
            device
        )
        X_smiles = [
            name_to_smiles[name_to_smiles["name"] == name].smiles.values[0]
            for name in train_df.index
        ]

        # Train the model and evaluate on the test set
        model = train(
            args,
            X_smiles,
            reaction_smiles,
            train_incidence_pos,
            incidence_train,
            incidence_valid,
            y_train,
            y_valid,
        )

        true_test_score, ground_truth = predict(
            model, train_incidence_pos, incidence_test, y_test
        )

        testing_results.append(
            {"raw_pred": true_test_score.tolist(), "raw_gt": ground_truth.tolist()}
        )

    # Save results to file
    if args.recover:
        recover_dict = {
            "model": args.train,
            "algorithm": args.baseline,
            "remove": args.remove,
            "recover": recover_results,
        }

        with open("recover.jsonl", "a") as f:
            f.write(json.dumps(recover_dict) + "\n")

    else:
        result_dict = {
            "model": args.train,
            "algorithm": args.baseline,
            "results": testing_results,
        }

        output_dir = os.path.dirname(args.raw_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if not os.path.isfile(args.raw_path):
            open(args.raw_path, "w").close()

        with open(args.raw_path, "a") as f:
            f.write(json.dumps(result_dict) + "\n")
