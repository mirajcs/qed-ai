# data/prepare_data.py
# QED AI — Download and prepare training data
# Run this ONCE on HPC to get training data

import json
import os
from tqdm import tqdm


def extract_leandojo_data(output_path="data/lean4_tactics.json"):
    """
    Extract training data from Mathlib4 via LeanDojo.
    This gives us 259k (proof_state → tactic) pairs.
    """
    print("=" * 50)
    print("  QED AI — Data Preparation")
    print("=" * 50)

    from lean_dojo import LeanGitRepo, trace

    print("\n📥 Downloading Mathlib4 via LeanDojo...")
    print("   This takes 2-4 hours — submit as SLURM job!")

    repo = LeanGitRepo("https://github.com/leanprover-community/mathlib4", "v4.3.0")
    traced = trace(repo)

    print("\n🔄 Extracting tactics...")
    data = []
    for theorem in tqdm(traced.get_theorems()):
        for tactic in theorem.traced_tactics:
            data.append(
                {
                    "state_before": str(tactic.state_before),
                    "tactic": str(tactic.tactic),
                    "state_after": str(tactic.state_after),
                    "theorem_name": str(theorem.full_name),
                }
            )

    # Save
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"\n✅ Saved {len(data):,} tactic examples")
    print(f"   Location: {output_path}")
    return data


def load_data(path="data/lean4_tactics.json"):
    with open(path) as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data):,} examples")
    return data


def prepare_for_training(data, tokenizer, max_len=512):
    """
    Convert raw data into training format.
    Input:  "STATE: ⊢ n + 0 = n TACTIC:"
    Output: "simp"
    """
    from torch.utils.data import Dataset
    import torch

    class TacticDataset(Dataset):
        def __init__(self, data, tokenizer, max_len):
            self.samples = []
            print("Preparing dataset...")
            for item in tqdm(data):
                text = f"STATE: {item['state_before']} TACTIC: {item['tactic']}"
                ids = tokenizer.encode(text)
                ids = ids[:max_len]
                # Pad to max_len
                ids += [0] * (max_len - len(ids))
                self.samples.append(ids)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            ids = torch.tensor(self.samples[idx], dtype=torch.long)
            return ids[:-1], ids[1:]  # input, target

    return TacticDataset(data, tokenizer, max_len)


if __name__ == "__main__":
    extract_leandojo_data()
