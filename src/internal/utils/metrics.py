import os
import json
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--raw_path", type=str, required=True, help="Path to input directory containing raw .jsonl result files")
parser.add_argument("--output_file", type=str, required=True, help="Path to save processed results .jsonl")
parser.add_argument("--threshold_mode", type=str, choices=["value", "mean", "median"], default="value", help="Thresholding strategy: value, mean, or median")
parser.add_argument("--threshold_value", type=float, default=0.5, help="Specific threshold value used if threshold_mode is 'value'")
args = parser.parse_args()

raw_path = args.raw_path
output_file = args.output_file

os.makedirs(os.path.dirname(output_file), exist_ok=True)

jsonl_files = glob.glob(os.path.join(raw_path, "*.jsonl"))

with open(output_file, "w") as out_f:
    for jsonl_file in jsonl_files:
        tqdm.write(f"Processing {jsonl_file}...")
        
        with open(jsonl_file, "r") as f:
            lines = f.readlines()

        with tqdm(total=len(lines), desc="Processing lines", unit="line", leave=True) as pbar:
            for line in lines:
                data = json.loads(line)

                model = data["model"]
                algorithm = data["algorithm"]
                results = data["results"]

                processed_results = []

                for iteration in results:
                    raw_pred = np.array(iteration["raw_pred"])
                    raw_gt = np.array(iteration["raw_gt"])
                    
                    if args.threshold_mode == "value":
                        threshold = args.threshold_value
                    elif args.threshold_mode == "mean":
                        threshold = np.mean(raw_pred)
                    elif args.threshold_mode == "median":
                        threshold = np.median(raw_pred)
                    
                    b_score = np.array([int(s >= threshold) for s in raw_pred])

                    f1 = f1_score(raw_gt, b_score, zero_division=0)
                    precision = precision_score(raw_gt, b_score, zero_division=0)
                    recall = recall_score(raw_gt, b_score, zero_division=0)
                    accuracy = accuracy_score(raw_gt, b_score)

                    try:
                        auroc = roc_auc_score(raw_gt, raw_pred)
                        aupr = average_precision_score(raw_gt, raw_pred)
                    except ValueError:
                        auroc = None
                        aupr = None

                    processed_results.append({
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "accuracy": accuracy,
                        "auroc": auroc if auroc is not None else "N/A",
                        "aupr": aupr if aupr is not None else "N/A"
                    })

                pbar.update(1)

                result_dict = {
                    "model": model,
                    "algorithm": algorithm,
                    "results": processed_results
                }

                out_f.write(json.dumps(result_dict) + "\n")

print(f"\nProcessed results saved to {output_file}")