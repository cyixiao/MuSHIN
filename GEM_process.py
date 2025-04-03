import argparse
import os
import pandas as pd
import cobra
from tqdm import tqdm

from process_data import change_metaid_to_metaname, change_arrow
import numpy as np
from rdkit import Chem
from rdkit.Chem.Lipinski import HeavyAtomCount
from BiggDatabase import BiggDatabase as BiggDB


def parse():
    # Parse command-line arguments to get the GEM name.
    parser = argparse.ArgumentParser()
    parser.add_argument("--GEM_name", type=str, default="iMM904")
    return parser.parse_args()


def get_filenames(path):
    # Get a sorted list of filenames in the specified directory.
    return sorted(os.listdir(path))


def remove_rxn(model, name_list):
    # Remove reactions from the model that do not involve metabolites in the given name list.
    remove_list = []
    for i in range(len(model.metabolites)):
        meta = model.metabolites[i]
        if meta.name in name_list:
            continue
        remove_list.append(meta)
    model.remove_metabolites(remove_list, destructive=True)
    print(f"remove_rxn:{len(remove_list)}")


def get_data(path, sample):
    # Load a metabolic model from an SBML file.
    model = cobra.io.read_sbml_model(path + "/" + sample)
    return model


def remove_right_empty(path="../iMM904", output_rxn_file=None, output_meta_file=None):
    # Process metabolic models to remove reactions with empty right-hand sides.
    namelist = get_filenames(path)
    rxns_list = []
    for sample in namelist:
        if sample.endswith("xml"):
            model = get_data(path, sample)
            rxn_equation = np.array([rxn for rxn in model.reactions])
            for rxn in rxn_equation:
                id, rxn = str(rxn).split(": ")
                if "<--" in rxn:
                    left, right = rxn.split("<--")
                    rxn = right + " => " + left
                tem_rxn = rxn.replace("-->", "=>")
                dir = "=>"
                if "<=>" in tem_rxn:
                    dir = "<=>"
                print(tem_rxn)
                left, right = tem_rxn.split(dir)
                if right == " ":
                    continue
                tem_rxn = tem_rxn.strip()
                rxns_list.append(tem_rxn)
            rxn_equation_list_df = pd.DataFrame({"rxn_equation": rxns_list})

            # Extract metabolite information and save to files if specified.
            metas_id = np.array([meta.id for meta in model.metabolites])
            metas_id_df = pd.DataFrame({"name_id": metas_id})
            metas = np.array([meta.name for meta in model.metabolites])
            metas_name_df = pd.DataFrame(data=metas[0:], columns=["name"])
            metas_links = np.array([meta.annotation for meta in model.metabolites])
            metas_links_df = pd.DataFrame({"links": metas_links})
            metas = pd.concat([metas_id_df, metas_name_df, metas_links_df], axis=1)

            if output_rxn_file is not None:
                rxn_equation_list_df.to_csv(output_rxn_file, index=False)
            if output_meta_file is not None:
                metas.to_csv(output_meta_file, index=False)
    return rxns_list, metas


def get_chebi_link(metas, output_meta_chebi_file=None):
    # Extract ChEBI links from metabolite annotations.
    valid_metas = metas[metas["links"].apply(lambda x: "chebi" in x)]
    valid_metas.reset_index()
    valid_metas_re = valid_metas.reset_index(drop=True)
    l = []
    for i in range(valid_metas_re.shape[0]):
        p = valid_metas_re.links[i]["chebi"]
        if not isinstance(p, list):
            l.append([p])
        else:
            l.append(p)
    valid_metas_re["chebi"] = l
    if output_meta_chebi_file is not None:
        valid_metas_re.to_csv(output_meta_chebi_file, index=False)
    return valid_metas_re


def get_smiles(all_metas, output_smiles_file=None):
    # Map ChEBI IDs to SMILES strings using a pre-cleaned ChEBI dataset.
    data = pd.read_csv("./data/pool/cleaned_chebi.csv")
    smiles = {"name": [], "smiles": [], "name_id": []}
    for i, row in all_metas.iterrows():
        name = row["name"]
        name_id = row["name_id"]
        s = ""
        for che in row["chebi"]:
            try:
                s = data[int(che.split(":")[-1]) == data["ChEBI_ID"]].smiles.values[0]
                break
            except Exception:
                pass
        smiles["name"].append(name)
        smiles["smiles"].append(s)
        smiles["name_id"].append(name_id)
    meta_smiles = pd.DataFrame(smiles)
    meta_smiles.drop_duplicates(inplace=True, ignore_index=True)
    meta_smiles = meta_smiles[meta_smiles["smiles"] != ""]
    if output_smiles_file is not None:
        meta_smiles.to_csv(output_smiles_file, index=False)
    return meta_smiles


def get_smiles_from_db(all_metas, output_smiles_file=None):
    # Retrieve SMILES strings for metabolites using the BiggDatabase.
    smiles = {"name": [], "smiles": [], "name_id": []}
    db = BiggDB()
    for i, row in tqdm(all_metas.iterrows(), total=len(all_metas)):
        name = row["name"]
        name_id = row["name_id"]
        s = db.find_by_name(name)
        if not s:
            print(f"can not find the smiles for {name}")
        smiles["name"].append(name)
        smiles["smiles"].append(s)
        smiles["name_id"].append(name_id)
    meta_smiles = pd.DataFrame(smiles)
    meta_smiles.drop_duplicates(inplace=True, ignore_index=True)
    meta_smiles = meta_smiles[meta_smiles["smiles"] != ""]
    if output_smiles_file is not None:
        meta_smiles.to_csv(output_smiles_file, index=False)
    return meta_smiles


def cout_atom_number(metas, output_meta_count_file=None):
    # Count the number of heavy atoms in each metabolite's SMILES string.
    results = pd.DataFrame([], columns=["name", "smiles", "count"])
    smiles = []
    count = []
    metas["mol"] = metas["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
    metas = metas[metas["mol"].isna() == False]
    metas.loc[:, "smiles"] = metas["mol"].apply(lambda x: Chem.MolToSmiles(x))
    metas.loc[:, "count"] = metas["mol"].apply(lambda x: HeavyAtomCount(x))
    metas = metas.drop(columns=["mol"])
    metas_remove_dup = metas.drop_duplicates("name", ignore_index=True)
    if output_meta_count_file is not None:
        metas_remove_dup.to_csv(output_meta_count_file, index=False)
    return metas_remove_dup


if __name__ == "__main__":
    # Main script to process GEM data and generate outputs.
    args = parse()
    gem_name = args.GEM_name
    print(gem_name)
    rxn_list_no_empty, all_metas = remove_right_empty(
        path=f"./data/{gem_name}",
        output_rxn_file=f"./data/{gem_name}/{gem_name}_rxn_no_empty.csv",
    )
    all_metas_name = all_metas.loc[:, ["name"]]
    all_metas_remove_dup = all_metas_name.drop_duplicates()

    print(f"{all_metas_name}")
    print(f"{all_metas_remove_dup}")
    meta_smiles = get_smiles_from_db(all_metas_remove_dup)
    meta_smiles_count = cout_atom_number(
        meta_smiles,
        output_meta_count_file=f"./data/{gem_name}/{gem_name}_meta_count.csv",
    )

    rxn_list_no_empty_clean = pd.read_csv(
        f"./data/{gem_name}/{gem_name}_rxn_no_empty.csv"
    )["rxn_equation"].tolist()
    pos_rxns_names = change_metaid_to_metaname(rxn_list_no_empty_clean, all_metas)
    rxn_all_name_sample = pd.DataFrame({"rxn_names": pos_rxns_names})
    rxn_all_name_sample_clean = rxn_all_name_sample.drop_duplicates()

    pos_rxns = change_arrow(
        pos_rxns_names,
        filter_name=meta_smiles_count["name"].tolist(),
        save_file=f"./data/{gem_name}/{gem_name}_rxn_name_list.txt",
    )
