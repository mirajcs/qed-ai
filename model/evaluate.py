# model/evaluate.py
# Top-1 / Top-k exact-match accuracy and per-token perplexity
# on the LeanDojo Benchmark 4 random/test split.

import argparse
import json
import os
import re
import sys
from collections import Counter

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


_WS = re.compile(r"\s+")


def normalize(s):
    return _WS.sub(" ", s.strip().lower())


def head_token(s):
    s = s.strip()
    if not s:
        return ""
    return s.split()[0].rstrip(",;[(<")


def teacher_forced_nll(model, prompt_ids, target_ids, device):
    max_len = model.cfg.max_seq_len
    target_ids = target_ids[: max_len - 1]
    prompt_ids = prompt_ids[-(max_len - len(target_ids)) :]
    full = torch.tensor([prompt_ids + target_ids], device=device)
    with torch.no_grad():
        logits = model(full)
    prompt_len = len(prompt_ids)
    pred = logits[0, prompt_len - 1 : prompt_len - 1 + len(target_ids), :]
    target = torch.tensor(target_ids, device=device)
    log_probs = F.log_softmax(pred, dim=-1)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return nll.mean().item()


def evaluate(model_path="saved_model", test_path=DEFAULT_TEST_PATH,
             limit=500, k=5, temperature=0.7, device=None,
             debug=0, examples_path=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {model_path}/")

    tok = QEDTokenizer.load(model_path)
    model = QEDAI.load(model_path).to(device).eval()

    pairs = load_test_pairs(test_path, limit=limit)
    print(f"Evaluating {len(pairs):,} (state, tactic) pairs...\n")

    top1_exact = 0
    topk_exact = 0
    top1_lenient = 0
    topk_lenient = 0
    head_hits = 0
    nll_sum = 0.0
    nll_n = 0
    gold_heads = Counter()
    pred_heads = Counter()
    debug_rows = []

    for i, (sb, gold) in enumerate(tqdm(pairs)):
        prompt = f"STATE: {sb} TACTIC: "
        prompt_ids = tok.encode(prompt)[:MAX_PROMPT_TOKENS]
        gold_ids = tok.encode(gold, add_special=False)
        gold_str = gold.strip()
        gold_norm = normalize(gold_str)
        gold_head = head_token(gold_str)
        gold_heads[gold_head] += 1

        # Top-1: greedy
        greedy_ids = greedy_decode(model, prompt_ids, device)
        greedy_str = tok.decode(greedy_ids).strip()
        greedy_norm = normalize(greedy_str)
        greedy_head = head_token(greedy_str)
        pred_heads[greedy_head] += 1

        if greedy_str == gold_str:
            top1_exact += 1
        if greedy_norm == gold_norm:
            top1_lenient += 1
        if greedy_head and greedy_head == gold_head:
            head_hits += 1

        # Top-k: greedy + (k-1) samples
        exact_set = {greedy_str}
        lenient_set = {greedy_norm}
        for _ in range(k - 1):
            sampled = sample_decode(model, prompt_ids, device, temperature=temperature)
            s = tok.decode(sampled).strip()
            exact_set.add(s)
            lenient_set.add(normalize(s))
        if gold_str in exact_set:
            topk_exact += 1
        if gold_norm in lenient_set:
            topk_lenient += 1

        # Perplexity
        if gold_ids:
            nll = teacher_forced_nll(model, prompt_ids, gold_ids, device)
            nll_sum += nll
            nll_n += 1

        if i < debug:
            debug_rows.append({
                "state_before": sb[:200],
                "gold": gold_str,
                "greedy": greedy_str,
                "gold_head": gold_head,
                "pred_head": greedy_head,
            })

    n = len(pairs)
    avg_nll = nll_sum / max(nll_n, 1)
    ppl = float(torch.exp(torch.tensor(avg_nll)).item())
    metrics = {
        "n": n,
        "top1_exact": top1_exact / n,
        f"top{k}_exact": topk_exact / n,
        "top1_lenient": top1_lenient / n,
        f"top{k}_lenient": topk_lenient / n,
        "head_match": head_hits / n,
        "nll_per_token": avg_nll,
        "perplexity": ppl,
        "top_gold_heads": gold_heads.most_common(15),
        "top_pred_heads": pred_heads.most_common(15),
    }

    print()
    print("=" * 60)
    print(f"  Examples evaluated:        {n:,}")
    print(f"  Top-1 exact match:         {metrics['top1_exact']:.3%}")
    print(f"  Top-{k} exact match:         {metrics[f'top{k}_exact']:.3%}")
    print(f"  Top-1 lenient match:       {metrics['top1_lenient']:.3%}")
    print(f"  Top-{k} lenient match:       {metrics[f'top{k}_lenient']:.3%}")
    print(f"  Head-token match (top-1):  {metrics['head_match']:.3%}")
    print(f"  Mean NLL/token:            {avg_nll:.4f}")
    print(f"  Perplexity:                {ppl:.2f}")
    print("=" * 60)

    print("\n  Top gold tactic heads:")
    for h, c in metrics["top_gold_heads"]:
        print(f"    {c:6d}  {h!r}")
    print("\n  Top predicted heads (greedy):")
    for h, c in metrics["top_pred_heads"]:
        print(f"    {c:6d}  {h!r}")

    if debug_rows:
        print("\n  Debug (first {} examples):".format(len(debug_rows)))
        for r in debug_rows:
            print(f"    state: {r['state_before']!r}")
            print(f"      gold:   {r['gold']!r}  [head={r['gold_head']!r}]")
            print(f"      greedy: {r['greedy']!r}  [head={r['pred_head']!r}]")

    if examples_path:
        os.makedirs(os.path.dirname(examples_path) or ".", exist_ok=True)
        with open(examples_path, "w") as f:
            json.dump(debug_rows, f, indent=2, ensure_ascii=False)
        print(f"\n  Debug examples written to {examples_path}")

    return metrics


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
    p.add_argument("--debug", type=int, default=0,
                   help="Print first N (state, gold, greedy) triples for inspection.")
    p.add_argument("--examples-out", default=None,
                   help="Optional path to write debug examples as JSON.")
    args = p.parse_args()

    metrics = evaluate(
        model_path=args.model_path,
        test_path=args.test_path,
        limit=args.limit,
        k=args.k,
        temperature=args.temperature,
        device=args.device,
        debug=args.debug,
        examples_path=args.examples_out,
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics written to {args.out}")
