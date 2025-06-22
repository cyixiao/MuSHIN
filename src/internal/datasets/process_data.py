import re
import math
import random

from tqdm import tqdm


def get_coefficient_and_reactant(reactions):
    # Parses reaction strings to extract coefficients, reactants, products, and reaction direction.
    reactions_index = []
    reactions_metas = []
    reactants_nums = []
    direction = []
    for rxn in reactions:
        if rxn.startswith("=>"):
            continue
        rxn_index, rxn_metas, rxn_direction = [], [], '=>'
        if '<=>' in rxn:
            rxn_direction = '<=>'
        tem_rxn = rxn.replace(' ' + rxn_direction + ' ', ' + ')
        metas = tem_rxn.split(' + ')
        for m in metas:
            a = re.findall('\d+\.?\d*', m)
            aa = re.findall(r"^([\\+|-]?\\d+(.{0}|.\\d+))[Ee]{1}([\\+|-]?\\d+)$", m)
            b = m.split(' ')
            if len(a) and a[0] == b[0]:
                rxn_index.append(a[0])
                rxn_metas.append(' '.join(b[1:]))
            else:
                rxn_metas.append(m)
                rxn_index.append('1')
        reactant, product = rxn.split(' ' + rxn_direction + ' ')
        reactants = reactant.split(' + ')
        products = product.split(' + ')
        reactants_nums.append(len(reactants))
        reactions_index.append(rxn_index)
        reactions_metas.append(rxn_metas)
        direction.append(rxn_direction)
    return reactions_index, reactions_metas, reactants_nums, direction


def combine_meta_rxns(reactions_index, reactions_metas, reactants_nums, reactants_direction):
    # Combines reaction metadata into a single reaction string.
    re_index = reactions_index[:reactants_nums]
    pro_index = reactions_index[reactants_nums:]
    metas = reactions_metas
    rxn = str(re_index[0]) + ' ' + metas[0]
    for j in range(1, len(re_index)):
        rxn = rxn + ' + ' + str(re_index[j]) + ' ' + metas[j]
    rxn = rxn + ' ' + reactants_direction + ' ' + str(pro_index[0]) + ' ' + metas[len(re_index)]
    for j in range(1, len(pro_index)):
        rxn = rxn + ' + ' + str(pro_index[j]) + ' ' + metas[len(re_index) + j]
    return rxn


def create_neg_rxn(pos_rxn, pos_data_soucre, neg_data_soucre, negative_ratio=1, atom_ratio=0.5):
    # Generates negative reactions by replacing reactants/products with alternatives from a negative dataset.
    reactions_index, reactions_metas, reactants_nums, reactants_direction = get_coefficient_and_reactant(pos_rxn)
    neg_rxn_name_list = []
    assert negative_ratio >= 1 and isinstance(negative_ratio, int)
    for i in tqdm(range(len(reactions_index))):
        for j in range(negative_ratio):
            selected_atoms = math.floor(len(reactions_metas[i]) * atom_ratio)
            assert selected_atoms > 0, "The number of selected atoms is zero"
            index_value = random.sample(list(enumerate(reactions_metas[i])), selected_atoms)
            for index, meta in index_value:
                dup = True
                count = pos_data_soucre[pos_data_soucre['name'] == meta]['count'].values[0]
                while dup:
                    found_chebi_metas = neg_data_soucre[neg_data_soucre['count'] == count].sample(1)['name'].values[0]
                    if found_chebi_metas not in reactions_metas[i]:
                        dup = False
                neg_metas = reactions_metas[i].copy()
                neg_metas[index] = found_chebi_metas
            neg_rxns = combine_meta_rxns(reactions_index[i], neg_metas, reactants_nums[i],
                                         reactants_direction[i])
            neg_rxn_name_list.append(neg_rxns)
    return neg_rxn_name_list


def change_metabolites_to_smiles(rxns, df_metas_smiles):
    # Converts metabolite names in reactions to their corresponding SMILES representations.
    reactions_index, reactions_metas, reactants_nums, reactants_direction = get_coefficient_and_reactant(rxns)
    reactions_smiles = []
    for metas in reactions_metas:
        reactions_smiles.append([df_metas_smiles[df_metas_smiles['name'] == meta].smiles.values[0] for meta in metas])
    rxn_index = [[math.ceil(float(x)) for x in rxn_index] for rxn_index in reactions_index]
    rxns_smiles = []
    for i in tqdm(range(len(rxn_index))):
        smiles = reactions_smiles[i]
        re_index = rxn_index[i][:reactants_nums[i]]
        pro_index = rxn_index[i][reactants_nums[i]:]
        rxn_smiles = '.'.join([smiles[0]] * re_index[0])
        for j in range(1, len(re_index)):
            rxn_smiles = rxn_smiles + '.' + '.'.join([smiles[j]] * re_index[j])
        rxn_smiles = rxn_smiles + '=>' + '.'.join([smiles[len(re_index)]] * pro_index[0])
        for j in range(1, len(pro_index)):
            rxn_smiles = rxn_smiles + '.' + '.'.join([smiles[len(re_index) + j]] * pro_index[j])
        rxns_smiles.append(rxn_smiles)
    return rxns_smiles


def change_metabolites_to_smiles2(rxns, df_metas_smiles):
    # Alternative implementation to convert metabolite names to SMILES, using a different approach.
    reactions_index, reactions_metas, reactants_nums, reactants_direction = get_coefficient_and_reactant(rxns)
    rxn_smiles = []
    for i in range(len(reactions_metas)):
        metas = reactions_metas[i]
        print(f'metas: {metas}')
        metas_new = []
        for m in metas:
            metas_new.append(df_metas_smiles[df_metas_smiles['name'] == m].smiles.values[0])
        rxn_name = combine_meta_rxns(reactions_index[i], metas_new, reactants_nums[i], reactants_direction[i])
        rxn_smiles.append(rxn_name)
    return rxn_smiles


def change_metaid_to_metaname(rxns, df_metas_names):
    # Converts metabolite IDs in reactions to their corresponding names.
    reactions_index, reactions_metas, reactants_nums, reactants_direction = get_coefficient_and_reactant(rxns)
    rxn_names = []
    for i in range(len(reactions_metas)):
        metas = reactions_metas[i]
        metas_new = []
        for m in metas:
            print(m)
            m = m.split(' ')[-1] if ' ' in m else m
            metas_new.append(df_metas_names[df_metas_names['name_id'] == m].name.values[0])
        rxn_name = combine_meta_rxns(reactions_index[i], metas_new, reactants_nums[i], reactants_direction[i])
        rxn_names.append(rxn_name)
    return rxn_names


def change_arrow(rxns, filter_name=None, save_file=None):
    # Normalizes reaction arrows and optionally filters reactions by metabolite names or saves them to a file.
    results = []
    for rxn in rxns:
        if '=> ' in rxn:
            l, r = rxn.split('=> ')
            l, r = l.strip(), r.strip()
            if len(r) == 0:
                continue
            if '<=>' in rxn:
                l, r = rxn.split('<=>')
                l, r = l.strip(), r.strip()
                reaction = l + ' => ' + r
                if reaction not in results:
                    results.append(reaction)
                reaction = r + ' => ' + l
                if reaction not in results:
                    results.append(reaction)
            else:
                if rxn not in results:
                    results.append(rxn)
    if filter_name is not None:
        # Filters reactions to include only those with specified metabolite names.
        filter_rxn = []
        for rxn in results:
            rxn_index, rxn_metas = [], []
            tem_rxn = rxn.replace(' => ', ' + ')
            metas = tem_rxn.split(' + ')
            for m in metas:
                a = re.findall('\d+\.?\d*', m)
                b = m.split(' ')
                if len(a) and a[0] == b[0]:
                    rxn_metas.append(' '.join(b[1:]))
                else:
                    rxn_metas.append(m)
            temp = True
            for meta in rxn_metas:
                if meta not in filter_name:
                    temp = False
                    break
            if not temp:
                continue
            filter_rxn.append(rxn)
        results = filter_rxn
    if save_file is not None:
        # Saves the filtered reactions to a file.
        with open(save_file, 'w') as f:
            for r in results:
                f.write(r + '\n')
    return results