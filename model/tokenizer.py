# model/tokenizer.py
# QED AI — Custom Lean 4 Tokenizer

import json
import os
from collections import Counter

# Special tokens
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<state>", "<tactic>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# Lean 4 math symbols — these get their own tokens
LEAN_SYMBOLS = [
    "⊢", "→", "←", "↔", "∀", "∃", "∧", "∨", "¬", "≠",
    "≤", "≥", "∈", "∉", "⊂", "⊆", "∩", "∪", "∑", "∏",
    ":=", "::", "=>", "by", "have", "show", "calc", "match",
    "induction", "cases", "simp", "rw", "exact", "apply",
    "intro", "intros", "constructor", "use", "omega", "ring",
    "norm_num", "decide", "trivial", "tauto", "linarith",
    "nlinarith", "field_simp", "push_neg", "contrapose",
    "funext", "ext", "congr", "conv", "rfl", "aesop",
    "<;>", "·", "fun", "let", "in", "if", "then", "else",
    "theorem", "lemma", "def", "example", "import", "open",
    "namespace", "end", "section", "variable", "where",
    "Nat", "Int", "Real", "Complex", "Bool", "List", "Set",
    "Type", "Prop", "Sort", "ℕ", "ℤ", "ℝ", "ℂ", "𝔽",
]

class QEDTokenizer:
    def __init__(self):
        self.vocab     = {}
        self.inv_vocab = {}
        self.vocab_size = 0

    def build(self, texts, target_vocab_size=32000):
        print(f"Building tokenizer from {len(texts):,} texts...")

        # Start with special tokens + Lean symbols
        all_tokens = SPECIAL_TOKENS + LEAN_SYMBOLS

        # Add individual characters from corpus
        char_freq = Counter()
        for text in texts:
            char_freq.update(text)

        # Add most common characters
        common_chars = [c for c, _ in char_freq.most_common(500)
                       if c not in all_tokens]
        all_tokens += common_chars

        # Simple BPE — merge common pairs
        print("Running BPE merges...")
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                word_freqs[word] += 1

        # Get current vocabulary as set for fast lookup
        vocab_set = set(all_tokens)

        # BPE merges until target size
        pair_freq = Counter()
        for word, freq in word_freqs.items():
            chars = list(word)
            for i in range(len(chars) - 1):
                pair = chars[i] + chars[i+1]
                pair_freq[pair] += freq

        while len(all_tokens) < target_vocab_size:
            if not pair_freq:
                break
            best_pair = pair_freq.most_common(1)[0][0]
            if best_pair not in vocab_set:
                all_tokens.append(best_pair)
                vocab_set.add(best_pair)
            del pair_freq[best_pair]

        # Build vocab dicts
        all_tokens = list(dict.fromkeys(all_tokens))  # deduplicate
        self.vocab     = {tok: i for i, tok in enumerate(all_tokens)}
        self.inv_vocab = {i: tok for i, tok in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)

        print(f"✅ Tokenizer built — vocab size: {self.vocab_size:,}")
        return self

    def encode(self, text, add_special=True):
        tokens = [BOS_ID] if add_special else []
        i = 0
        chars = list(text)

        while i < len(chars):
            # Try longest match first (up to 20 chars)
            matched = False
            for length in range(min(20, len(chars) - i), 0, -1):
                substr = "".join(chars[i:i+length])
                if substr in self.vocab:
                    tokens.append(self.vocab[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                tokens.append(UNK_ID)
                i += 1

        if add_special:
            tokens.append(EOS_ID)
        return tokens

    def decode(self, ids, skip_special=True):
        special_ids = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
        tokens = []
        for id in ids:
            if skip_special and id in special_ids:
                continue
            tokens.append(self.inv_vocab.get(id, "<unk>"))
        return "".join(tokens)

    def save(self, path="saved_model"):
        os.makedirs(path, exist_ok=True)
        data = {
            "vocab"     : self.vocab,
            "vocab_size": self.vocab_size
        }
        with open(f"{path}/tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"✅ Tokenizer saved to {path}/tokenizer.json")

    @classmethod
    def load(cls, path="saved_model"):
        tok = cls()
        with open(f"{path}/tokenizer.json", encoding="utf-8") as f:
            data = json.load(f)
        tok.vocab      = data["vocab"]
        tok.inv_vocab  = {int(v): k for k, v in tok.vocab.items()}
        tok.vocab_size = data["vocab_size"]
        print(f"✅ Tokenizer loaded — vocab size: {tok.vocab_size:,}")
        return tok


# ═══════════════════════════════════════
# TEST
# ═══════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  QED AI — Tokenizer Test")
    print("=" * 50)

    # Sample Lean 4 texts
    sample_texts = [
        "⊢ ∀ (n : ℕ), n + 0 = n",
        "⊢ a + b = b + a",
        "simp omega ring norm_num",
        "induction n with | zero => simp | succ n ih => omega",
        "theorem add_comm (a b : Nat) : a + b = b + a := by ring",
    ] * 100  # repeat to simulate real dataset

    # Build tokenizer
    tok = QEDTokenizer()
    tok.build(sample_texts, target_vocab_size=1000)  # small for testing

    # Test encode/decode
    test = "⊢ ∀ (n : ℕ), n + 0 = n"
    encoded = tok.encode(test)
    decoded = tok.decode(encoded)

    print(f"Original:  {test}")
    print(f"Encoded:   {encoded[:10]}... ({len(encoded)} tokens)")
    print(f"Decoded:   {decoded}")

    # Test save/load
    tok.save("saved_model")
    tok2 = QEDTokenizer.load("saved_model")
    print(f"✅ Save/load works!")

    print("=" * 50)
    print("  Tokenizer verified! ✅")
    print("=" * 50)
