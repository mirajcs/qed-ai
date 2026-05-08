# model/evaluate.py
# Top-1 / Top-k exact-match accuracy and per-token perplexity
# on the LeanDojo Benchmark 4 random/test split.

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import QEDAI
from model.tokenizer import QEDTokenizer

DEFAULT_TEST_PATH = "data/leandojo_benchmark_4/random/test.json"
MAX_PROMPT_TOKENS = 448
MAX_NEW_TOKENS = 64


def load_test_pairs(path, limit=None):
    with open(path) as f:
        theorems = json.load(f)
    pairs = []
    for thm in theorems:
        for t in thm.get("traced_tactics", []):
            sb = (t.get("state_before") or "").strip()
            tc = (t.get("tactic") or "").strip()
            if sb and tc:
                pairs.append((sb, tc))
    if limit:
        pairs = pairs[:limit]
    return pairs


def _step_logits(model, ids):
    return model(ids[:, -model.cfg.max_seq_len :])[:, -1, :]


def greedy_decode(model, prompt_ids, device, max_new=MAX_NEW_TOKENS, eos_id=2):
    ids = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        for _ in range(max_new):
            next_id = _step_logits(model, ids).argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == eos_id:
                break
    return ids[0, len(prompt_ids):].tolist()


def sample_decode(model, prompt_ids, device, max_new=MAX_NEW_TOKENS, temperature=0.7, eos_id=2):
    ids = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        for _ in range(max_new):
            logits = _step_logits(model, ids) / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == eos_id:
                break
    return ids[0, len(prompt_ids):].tolist()


def teacher_forced_nll(model, prompt_ids, target_ids, device):
    full = torch.tensor([prompt_ids + target_ids], device=device)
    if full.shape[1] > model.cfg.max_seq_len:
        full = full[:, -model.cfg.max_seq_len :]
        prompt_len = full.shape[1] - len(target_ids)
    else:
        prompt_len = len(prompt_ids)
    with torch.no_grad():
        logits = model(full)
    pred = logits[0, prompt_len - 1 : prompt_len - 1 + len(target_ids), :]
    target = torch.tensor(target_ids, device=device)
    log_probs = F.log_softmax(pred, dim=-1)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return nll.mean().item()


def evaluate(model_path="saved_model", test_path=DEFAULT_TEST_PATH,
             limit=500, k=5, temperature=0.7, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {model_path}/")

    tok = QEDTokenizer.load(model_path)
    model = QEDAI.load(model_path).to(device).eval()

    pairs = load_test_pairs(test_path, limit=limit)
    print(f"Evaluating {len(pairs):,} (state, tactic) pairs...\n")

    top1_hits = 0
    topk_hits = 0
    nll_sum = 0.0
    nll_n = 0

    for sb, gold in tqdm(pairs):
        prompt = f"STATE: {sb} TACTIC: "
        prompt_ids = tok.encode(prompt)[:MAX_PROMPT_TOKENS]
        gold_ids = tok.encode(gold, add_special=False)
        gold_str = gold.strip()

        # Top-1: greedy
        greedy_ids = greedy_decode(model, prompt_ids, device)
        greedy_str = tok.decode(greedy_ids).strip()
        if greedy_str == gold_str:
            top1_hits += 1

        # Top-k: greedy + (k-1) samples
        candidates = {greedy_str}
        for _ in range(k - 1):
            sampled = sample_decode(model, prompt_ids, device, temperature=temperature)
            candidates.add(tok.decode(sampled).strip())
        if gold_str in candidates:
            topk_hits += 1

        # Perplexity
        if gold_ids:
            nll = teacher_forced_nll(model, prompt_ids, gold_ids, device)
            nll_sum += nll
            nll_n += 1

    n = len(pairs)
    top1 = top1_hits / n
    topk = topk_hits / n
    avg_nll = nll_sum / max(nll_n, 1)
    ppl = float(torch.exp(torch.tensor(avg_nll)).item())

    print()
    print("=" * 55)
    print(f"  Examples evaluated:   {n:,}")
    print(f"  Top-1 exact match:    {top1:.3%}")
    print(f"  Top-{k} exact match:    {topk:.3%}")
    print(f"  Mean NLL/token:       {avg_nll:.4f}")
    print(f"  Perplexity:           {ppl:.2f}")
    print("=" * 55)

    return {
        "n": n,
        "top1": top1,
        f"top{k}": topk,
        "nll_per_token": avg_nll,
        "perplexity": ppl,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate QED AI on a held-out test split.")
    p.add_argument("--model-path", default="saved_model")
    p.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    p.add_argument("--limit", type=int, default=500,
                   help="Cap on (state, tactic) pairs evaluated. None for full test split.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="Optional path to write JSON metrics.")
    args = p.parse_args()

    metrics = evaluate(
        model_path=args.model_path,
        test_path=args.test_path,
        limit=args.limit,
        k=args.k,
        temperature=args.temperature,
        device=args.device,
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics written to {args.out}")
