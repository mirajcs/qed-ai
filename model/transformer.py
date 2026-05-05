# model/transformer.py
# QED AI — Model Architecture
# This defines the shape of your model.

import torch
import torch.nn as nn
import math


# ═══════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════
class QEDConfig:
    vocab_size = 32000  # unique tokens
    d_model = 512  # vector size
    n_heads = 8  # attention heads
    n_layers = 6  # transformer blocks
    d_ff = 2048  # feedforward size
    max_seq_len = 1024  # max input length
    dropout = 0.1  # regularization


# ═══════════════════════════════════════
# ATTENTION
# ═══════════════════════════════════════
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ═══════════════════════════════════════
# FEEDFORWARD
# ═══════════════════════════════════════
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(cfg)
        self.feedforward = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x


# ═══════════════════════════════════════
# QED AI — FULL MODEL
# ═══════════════════════════════════════
class QEDAI(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.apply(self._init_weights)
        print(f"✅ QED AI — {self.count_params():,} parameters")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def make_mask(self, T, device):
        mask = torch.tril(torch.ones(T, T, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(T, device=device)
        x = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        mask = self.make_mask(T, device)
        for block in self.blocks:
            x = block(x, mask)
        return self.output_head(self.norm(x))

    def generate(self, input_ids, max_new_tokens=64, temperature=0.8):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ids = input_ids[:, -self.cfg.max_seq_len :]
                logits = self(ids)[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_t = torch.multinomial(probs, 1)
                input_ids = torch.cat([input_ids, next_t], dim=1)
                if next_t.item() == 2:  # end token
                    break
        return input_ids

    def save(self, path="saved_model"):
        import os, json

        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/model_weights.pt")
        cfg_dict = {
            k: v for k, v in vars(self.cfg.__class__).items() if not k.startswith("_")
        }
        with open(f"{path}/config.json", "w") as f:
            json.dump(cfg_dict, f)
        print(f"✅ Model saved to {path}/")

    @classmethod
    def load(cls, path="saved_model"):
        import json

        with open(f"{path}/config.json") as f:
            cfg_dict = json.load(f)
        cfg = QEDConfig()
        for k, v in cfg_dict.items():
            setattr(cfg, k, v)
        model = cls(cfg)
        model.load_state_dict(
            torch.load(f"{path}/model_weights.pt", map_location="cpu")
        )
        return model


# ═══════════════════════════════════════
# TEST
# ═══════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  QED AI — Model Test")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = QEDConfig()
    model = QEDAI(cfg).to(device)

    # Test forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 128)).to(device)
    out = model(x)
    print(f"✅ Input shape:    {x.shape}")
    print(f"✅ Output shape:   {out.shape}")

    # Test generation
    prompt = torch.randint(0, cfg.vocab_size, (1, 10)).to(device)
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"✅ Generated {generated.shape[1] - 10} new tokens")

    # Test save/load
    model.save("saved_model")
    loaded = QEDAI.load("saved_model")
    print(f"✅ Save and load works!")

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"✅ GPU memory: {mem:.3f} GB")

    print("=" * 50)
    print("  QED AI model verified! ✅")
    print("=" * 50)
