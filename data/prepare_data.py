# data/prepare_data.py
# QED AI — Download and prepare training data
# Run this ONCE on HPC to get training data

import json
import os
from tqdm import tqdm


BENCHMARK_URL = "https://zenodo.org/records/12740403/files/leandojo_benchmark_4.tar.gz"
BENCHMARK_TGZ = "data/leandojo_benchmark_4.tar.gz"
BENCHMARK_DIR = "data/leandojo_benchmark_4"


def extract_leandojo_data(output_path="data/lean4_tactics.json"):
    """
    Build training data from the pre-extracted LeanDojo Benchmark 4 dataset
    (Zenodo). Downloads ~68 MB, then flattens theorem-level traces into
    (state_before, tactic, state_after, theorem_name) records.
    """
    import tarfile
    import urllib.request

    print("=" * 50)
    print("  QED AI — Data Preparation")
    print("=" * 50)

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(BENCHMARK_TGZ):
        print(f"\n📥 Downloading {BENCHMARK_URL}")
        urllib.request.urlretrieve(BENCHMARK_URL, BENCHMARK_TGZ)
        print(f"   Saved to {BENCHMARK_TGZ}")

    if not os.path.isdir(BENCHMARK_DIR):
        print(f"\n📂 Extracting {BENCHMARK_TGZ}")
        with tarfile.open(BENCHMARK_TGZ) as tar:
            tar.extractall("data")

    splits = [
        os.path.join(BENCHMARK_DIR, "random", name)
        for name in ("train.json", "val.json", "test.json")
    ]

    print("\n🔄 Flattening tactics...")
    data = []
    for split_path in splits:
        with open(split_path) as f:
            theorems = json.load(f)
        for thm in tqdm(theorems, desc=os.path.basename(split_path)):
            name = thm.get("full_name") or thm.get("name") or ""
            for t in thm.get("traced_tactics", []):
                data.append(
                    {
                        "state_before": t.get("state_before", ""),
                        "tactic": t.get("tactic", ""),
                        "state_after": t.get("state_after", ""),
                        "theorem_name": name,
                    }
                )

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
