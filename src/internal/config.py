import argparse


def parse():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument(
        "--train", type=str, default="iAF692", help="Training dataset name"
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default="./results.jsonl",
        help="Output file path for raw JSON results",
    )
    parser.add_argument("--seed", type=int, default=2, help="Random seed value")
    parser.add_argument(
        "--iteration", type=int, default=10, help="Number of model runs"
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="Maximum number of epochs"
    )

    # Algorithm settings
    parser.add_argument(
        "--algorithm",
        type=str,
        default="similarity",
        help="Algorithm to use: similarity, smiles2vec",
    )
    parser.add_argument(
        "--create_negative",
        type=bool,
        default=False,
        help="Generate negative samples under specific conditions",
    )
    parser.add_argument(
        "--atom_ratio",
        type=float,
        default=0.5,
        help="Ratio of replaced atoms for negative reactions",
    )
    parser.add_argument(
        "--negative_ratio",
        type=int,
        default=1,
        help="Negative-to-positive sample ratio",
    )

    # Model settings
    parser.add_argument(
        "--emb_dim", type=int, default=64, help="Input embedding dimension"
    )
    parser.add_argument(
        "--conv_dim", type=int, default=128, help="HypergraphConv output dimension"
    )
    parser.add_argument(
        "--head", type=int, default=6, help="Number of HypergraphConv heads"
    )
    parser.add_argument(
        "--L", type=int, default=2, help="Number of HypergraphConv layers"
    )
    parser.add_argument("--p", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--g_lambda", type=float, default=1, help="Gaussian kernel parameter"
    )

    # Training settings
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--batch_size", type=float, default=256, help="Batch size")
    parser.add_argument(
        "--s2m_batch_size", type=int, default=32, help="Batch size for s2m"
    )

    # Feature toggles
    parser.add_argument(
        "--enable_hygnn", action="store_true", help="Enable HyGNN processing"
    )
    parser.add_argument(
        "--disable_hygnn",
        action="store_false",
        dest="enable_hygnn",
        help="Disable HyGNN processing",
    )
    parser.add_argument(
        "--enable_reaction_fp",
        action="store_true",
        help="Enable reaction fingerprint generation",
    )
    parser.add_argument(
        "--disable_reaction_fp",
        action="store_false",
        dest="enable_reaction_fp",
        help="Disable reaction fingerprint generation",
    )

    return parser.parse_args()
