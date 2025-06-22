import os
import json
import sqlite3
import requests
from rdkit import Chem
import pubchempy as pcp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
from requests.exceptions import RequestException
import concurrent
from bs4 import BeautifulSoup
from xml.etree import ElementTree
import pandas as pd


def get_smiles_from_pubchem(inchikey, timeout):
    pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(pubchem_url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception as e:
        print(f"Caught error when fetching smiles of {inchikey} by InChI Key: {e}")
    return None


def get_smiles_from_kegg(kegg_id, timeout):
    kegg_url = f"http://rest.kegg.jp/get/cpd:{kegg_id}/mol"
    try:
        response = requests.get(kegg_url, timeout=timeout)
        response.raise_for_status()
        mol_data = response.text
        mol = Chem.MolFromMolBlock(mol_data)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Caught error when fetching smiles of {kegg_id} from KEGG: {e}")
    return None


def get_smiles_from_human_hmdb(hmdb_id, timeout):
    hmdb_url = f"http://www.hmdb.ca/metabolites/{hmdb_id}"
    try:
        response = requests.get(hmdb_url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        element = soup.find("th", string="SMILES").find_next("td")
        return element.find("div").text.strip()
    except Exception as e:
        print(f"Caught error when fetching smiles of {hmdb_id} from hmdb: {e}")
    return None


def get_smiles_from_biocyc(biocyc_id, timeout):
    biocyc_url = f"https://biocyc.org/getxml?META:{biocyc_id}"
    try:
        response = requests.get(biocyc_url, timeout=timeout)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        element = root.find(".//string[@title='smiles']")
        return element.text.strip()
    except Exception as e:
        print(f"Caught error when fetching smiles of {biocyc_id} from biocyc: {e}")
    return None


def get_smiles_from_mnx(mnx_id, timeout):
    mnx_url = f"https://www.metanetx.org/chem_info/{mnx_id}"
    try:
        response = requests.get(mnx_url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        element = soup.find("td", class_="smiles")
        return element.text.strip()
    except Exception as e:
        print(f"Caught error when fetching smiles of {mnx_id} from mnx: {e}")
    return None


def get_smiles_from_seed(seed_id, timeout):
    seed_url = f"https://modelseed.org/solr/compounds/select?wt=json&q=id:{seed_id}"
    try:
        response = requests.get(seed_url, timeout=timeout)
        response.raise_for_status()
        json_obj = response.json()
        return json_obj["smiles"].strip()
    except Exception as e:
        print(f"Caught error when fetching smiles of {seed_id} from seed: {e}")
    return None


def get_smiles_from_chebi(chebi_id, timeout):
    chebi_url = f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={chebi_id}"
    try:
        response = requests.get(chebi_url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        element = soup.find("td", string="SMILES").find_next_sibling("td")
        return element.text.strip()
    except Exception as e:
        print(f"Caught error when fetching smiles of {chebi_id} from chebi: {e}")
    return None


class BiggDatabase:
    def __init__(self, db_path, catched=True):
        self.db_path = db_path
        self.catched = catched
        self.file_conn = None
        self.memory_conn = None

    def _create_tables(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metabolites (
                id TEXT PRIMARY KEY,
                name TEXT,
                smiles TEXT
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metabolite_id TEXT,
                source TEXT,
                url TEXT,
                FOREIGN KEY (metabolite_id) REFERENCES metabolites (id)
            )
            """
        )

    def _get_smiles(
        self, metabolite_id, annotations, max_retries=1, retry_delay=0.1, timeout=5
    ):
        for annotation in annotations:
            source, url = annotation
            print(f"{source}, {url}")
            try:
                smiles = None
                if source == "InChI Key":
                    inchikey = url.split("/")[-1]
                    smiles = get_smiles_from_pubchem(timeout, url, inchikey)
                elif source == "KEGG Compound":
                    kegg_id = url.split("/")[-1]
                    smiles = get_smiles_from_kegg(timeout, kegg_id)
                elif source == "Human Metabolome Database":
                    hmdb_id = url.split("/")[-1]
                    smiles = get_smiles_from_human_hmdb(metabolite_id, timeout, hmdb_id)
                elif source == "BioCyc":
                    biocyc_id = url.split("/")[-1].split(":")[-1]
                    smiles = get_smiles_from_biocyc(biocyc_id, timeout)
                elif source == "MetaNetX (MNX) Chemical":
                    mnx_id = url.split("/")[-1]
                    smiles = get_smiles_from_mnx(mnx_id, timeout)
                elif source == "SEED Compound":
                    seed_id = url.split("/")[-1]
                    smiles = get_smiles_from_seed(seed_id, timeout)
                elif source == "CHEBI":
                    chebi_id = url.split("/")[-1].split(":")[-1]
                    smiles = get_smiles_from_chebi(chebi_id, timeout)
                if smiles:
                    return smiles
            except Exception as e:
                print(f"Something wrong when fetching smiles of {url} by {source}: {e}")

        return None

    def _get_smiles_thread(self, metabolite):
        metabolite_id = metabolite["id"]
        annotations = metabolite.get("annotation", [])
        smiles = self._get_smiles(metabolite_id, annotations)
        return metabolite_id, smiles

    def load_cleaned_chebi(self, start_index=0):
        data = pd.read_csv("data/pool/cleaned_chebi.csv")
        if start_index >= data.shape[0]:
            return
        count = 0
        for row in tqdm(
            data.iloc[start_index:].itertuples(index=False),
            total=data.shape[0] - start_index,
        ):
            obj = {
                "id": "chebi_id_" + str(row.ChEBI_ID),
                "name": row.name,
                "type": "metabolite",
                "drug_bank": None,
                "smiles": None if pd.isna(row.smiles) else row.smiles,
                "formula": None,
                "cas_number": None,
                "drug_groups": None,
                "inchikey": None,
                "inchi": None,
                "kegg_compound_id": None,
                "kegg_drug_id": None,
                "pubchem_compound_id": None,
                "pubchem_substance_id": None,
                "chebi_id": row.ChEBI_ID,
                "chembl_id": None,
                "het_id": None,
                "chemspider_id": None,
                "bindingdb_id": None,
                "hmd_id": None,
                "biocyc_id": None,
                "mnx_id": None,
                "seed_id": None,
                "reactome_id": None,
            }
            self.update_or_insert_data(obj)
            if self.catched or count == 32:
                self.conn.commit()
                count = 0
            count += 1
        self.conn.commit()

    def load_data(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        metabolites = data["metabolites"]
        smiles_results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._get_smiles_thread, m): m["id"]
                for m in metabolites
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Loading metabolites",
            ):
                try:
                    metabolite_id, smiles = future.result()
                    smiles_results[metabolite_id] = smiles
                except Exception as e:
                    print(f"Error fetching SMILES: {e}")
        for metabolite in metabolites:
            metabolite_id = metabolite["id"]
            name = metabolite["name"]
            annotations = metabolite.get("annotation", [])
            smiles = smiles_results.get(metabolite_id)
            self.cursor.execute(
                "INSERT OR IGNORE INTO metabolites (id, name, smiles) VALUES (?, ?, ?, ?)",
                (metabolite_id, name, smiles),
            )
            for annotation in annotations:
                source = annotation[0]
                url = annotation[1]
                self.cursor.execute(
                    "INSERT INTO annotations (metabolite_id, source, url) VALUES (?, ?, ?)",
                    (metabolite_id, source, url),
                )
        self.conn.commit()

    def load_chgnn_datasets(self, file_path):
        data = pd.read_csv(file_path)
        count = 0
        for i, row in tqdm(data.iterrows(), total=data.shape[0]):
            obj = {
                "id": row["DrugBank ID"],
                "name": row["Name"],
                "type": "drug",
                "drug_bank": row["DrugBank ID"],
                "smiles": row["SMILES"],
                "formula": row["Formula"],
                "cas_number": row["CAS Number"],
                "drug_groups": row["Drug Groups"],
                "inchikey": row["InChIKey"],
                "inchi": row["InChI"],
                "kegg_compound_id": row["KEGG Compound ID"],
                "kegg_drug_id": row["KEGG Drug ID"],
                "pubchem_compound_id": row["PubChem Compound ID"],
                "pubchem_substance_id": row["PubChem Substance ID"],
                "chebi_id": row["ChEBI ID"],
                "chembl_id": row["ChEMBL ID"],
                "het_id": row["HET ID"],
                "chemspider_id": row["ChemSpider ID"],
                "bindingdb_id": row["BindingDB ID"],
                "hmd_id": None,
                "biocyc_id": None,
                "mnx_id": None,
                "seed_id": None,
                "reactome_id": None,
            }
            self.update_or_insert_data(obj)
            if count == 32:
                self.conn.commit()
                count = 0
            count += 1
        self.conn.commit()

    def find_by_id(self, metabolite_id):
        self.cursor.execute("SELECT * FROM metabolites WHERE id = ?", (metabolite_id,))
        metabolite = self.cursor.fetchone()
        if metabolite:
            self.cursor.execute(
                "SELECT source, url FROM annotations WHERE metabolite_id = ?",
                (metabolite_id,),
            )
            annotations = self.cursor.fetchall()
            return {
                "id": metabolite[0],
                "name": metabolite[1],
                "smiles": metabolite[2],
                "annotations": [{"source": a[0], "url": a[1]} for a in annotations],
            }
        return None

    def find_by_name(self, name):
        self.cursor.execute("SELECT * FROM metabolites WHERE name = ?", (name,))
        metabolites = self.cursor.fetchall()
        results = []
        for metabolite in metabolites:
            self.cursor.execute(
                "SELECT source, url FROM annotations WHERE metabolite_id = ?",
                (metabolite[0],),
            )
            annotations = self.cursor.fetchall()
            results.append(
                {
                    "id": metabolite[0],
                    "name": metabolite[1],
                    "smiles": metabolite[2],
                    "annotations": [{"source": a[0], "url": a[1]} for a in annotations],
                }
            )
        return results

    def get_smiles_from_inchikey(self, metabolite_id):
        self.cursor.execute(
            "SELECT url FROM annotations WHERE metabolite_id = ? AND source = ?",
            (metabolite_id, "InChI Key"),
        )
        result = self.cursor.fetchone()
        if result:
            inchikey_url = result[0]
            inchikey = inchikey_url.split("/")[-1]
            try:
                compound = pcp.get_compounds(inchikey, namespace="inchikey")
                if compound:
                    return compound[0].canonical_smiles
            except Exception as e:
                print("PubChemPy error:", e)
        else:
            print("No InChI Key annotation found.")
        return None

    def get_smiles_from_kegg(self, metabolite_id):
        self.cursor.execute(
            "SELECT url FROM annotations WHERE metabolite_id = ? AND source = ?",
            (metabolite_id, "KEGG Compound"),
        )
        result = self.cursor.fetchone()
        if result:
            kegg_id = result[0].split("/")[-1]
            kegg_url = f"http://rest.kegg.jp/get/cpd:{kegg_id}/mol"
            response = requests.get(kegg_url)
            if response.status_code == 200:
                mol_data = response.text
                mol = Chem.MolFromMolBlock(mol_data)
                smiles = Chem.MolToSmiles(mol)
                return smiles
            else:
                print("Unable to fetch MOL data from KEGG:", response.status_code)
        print("No KEGG Compound annotation found.")
        return None

    def get_smiles_from_name(self, name):
        metabolites = self.find_by_name(name)
        if not metabolites:
            print("No matching metabolite name found.")
            return None
        for metabolite in metabolites:
            smiles = self.get_smiles_from_inchikey(metabolite["id"])
            if smiles:
                print(f"SMILES obtained via InChI Key: {smiles}")
                return smiles
        for metabolite in metabolites:
            smiles = self.get_smiles_from_kegg(metabolite["id"])
            if smiles:
                print(f"SMILES obtained via KEGG: {smiles}")
                return smiles
        print("Unable to obtain SMILES via InChI Key or KEGG.")
        return None

    def get_names_with_non_null_smiles(self):
        self.cursor.execute("SELECT name FROM metabolites WHERE smiles IS NOT NULL")
        rows = self.cursor.fetchall()
        return {row[0] for row in rows}

    def get_names_and_smiles(self):
        self.cursor.execute(
            "SELECT name, smiles FROM metabolites WHERE smiles IS NOT NULL"
        )
        rows = self.cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    def update_smiles(self, metabolite_id):
        self.cursor.execute(
            "select id, name from metabolites where id = ?", (metabolite_id,)
        )
        metabolites = self.cursor.fetchall()
        if not metabolites:
            print(f"No record found with id {metabolite_id}.")
            return
        for metabolite in tqdm(metabolites, desc="updating smiles"):
            metabolite_id, name = metabolite
            self.cursor.execute(
                "select source, url from annotations where metabolite_id = ?",
                (metabolite_id,),
            )
            annotations = self.cursor.fetchall()
            smiles = self._get_smiles(metabolite_id, annotations)
            if smiles:
                self.cursor.execute(
                    "update metabolites set smiles = ? where id = ?",
                    (smiles, metabolite_id),
                )
            else:
                print(
                    f"Unable to fetch SMILES for metabolite {name} (id: {metabolite_id})."
                )
        self.conn.commit()

    def update_null_smiles(self):
        self.cursor.execute("select id, name from metabolites where smiles is null")
        metabolites = self.cursor.fetchall()
        if not metabolites:
            print("All records already have SMILES filled.")
            return
        for metabolite in tqdm(metabolites, desc="updating smiles"):
            metabolite_id, name = metabolite
            self.cursor.execute(
                "select source, url from annotations where metabolite_id = ?",
                (metabolite_id,),
            )
            annotations = self.cursor.fetchall()
            smiles = self._get_smiles(metabolite_id, annotations)
            if smiles:
                self.cursor.execute(
                    "update metabolites set smiles = ? where id = ?",
                    (smiles, metabolite_id),
                )
                print(f"Found SMILES for {name}, {metabolite_id}")
            else:
                print(
                    f"Unable to fetch SMILES for metabolite {name} (id: {metabolite_id})."
                )
            self.conn.commit()

    def update_or_insert_data(self, data):
        name_to_check = data["name"].lower()
        id_to_check = data["id"].lower()
        self.cursor.execute(
            "SELECT COUNT(*) FROM metabolites WHERE LOWER(name) = ? and LOWER(id) = ?",
            (
                name_to_check,
                id_to_check,
            ),
        )
        record_exists = self.cursor.fetchone()[0] > 0
        if record_exists:
            update_query = """
                UPDATE metabolites
                SET
                    id = :id,
                    type = :type,
                    drug_bank = :drug_bank,
                    smiles = :smiles,
                    formula = :formula,
                    cas_number = :cas_number,
                    drug_groups = :drug_groups,
                    inchikey = :inchikey,
                    inchi = :inchi,
                    kegg_compound_id = :kegg_compound_id,
                    kegg_drug_id = :kegg_drug_id,
                    pubchem_compound_id = :pubchem_compound_id,
                    pubchem_substance_id = :pubchem_substance_id,
                    chebi_id = :chebi_id,
                    chembl_id = :chembl_id,
                    het_id = :het_id,
                    chemspider_id = :chemspider_id,
                    bindingdb_id = :bindingdb_id,
                    hmd_id = :hmd_id,
                    biocyc_id = :biocyc_id,
                    mnx_id = :mnx_id,
                    seed_id = :seed_id,
                    reactome_id = :reactome_id
                WHERE LOWER(name) = :name
            """
            self.cursor.execute(update_query, data)
        else:
            insert_query = """
                INSERT INTO metabolites (
                    id, name, type, drug_bank, smiles, formula, cas_number, drug_groups, inchikey, inchi,
                    kegg_compound_id, kegg_drug_id, pubchem_compound_id, pubchem_substance_id, chebi_id,
                    chembl_id, het_id, chemspider_id, bindingdb_id, hmd_id, biocyc_id, mnx_id, seed_id, reactome_id
                ) VALUES (
                    :id, :name, :type, :drug_bank, :smiles, :formula, :cas_number, :drug_groups, :inchikey, :inchi,
                    :kegg_compound_id, :kegg_drug_id, :pubchem_compound_id, :pubchem_substance_id, :chebi_id,
                    :chembl_id, :het_id, :chemspider_id, :bindingdb_id, :hmd_id, :biocyc_id, :mnx_id, :seed_id, :reactome_id
                )
            """
            self.cursor.execute(insert_query, data)

    def open(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.file_conn = sqlite3.connect(self.db_path)
        self.conn = self.file_conn
        if self.catched:
            self.memory_conn = sqlite3.connect(":memory:")
            with self.file_conn:
                self.file_conn.backup(self.memory_conn)
            self.conn = self.memory_conn
        self.file_conn.execute("PRAGMA journal_mode=WAL;")
        self.cursor = self.conn.cursor()
        self._create_tables()

    def close(self):
        if self.catched and self.memory_conn and self.file_conn:
            with self.memory_conn:
                self.memory_conn.backup(self.file_conn)
            self.memory_conn.close()
        self.file_conn.close()


if __name__ == "__main__":
    sqlite_db = "data/bigg/bigg_db.db"

    db = BiggDatabase(sqlite_db, catched=True)
    db.open()

    try:
        db.load_cleaned_chebi(12391)
    finally:
        db.close()
